"""AWS credential helpers for the ingest layer.

Functions here resolve IAM credentials while deliberately bypassing env
vars — needed when ``set_s3_credentials`` has temporarily overwritten
``AWS_*`` env vars with OPERA-scoped STS credentials for S3 direct
access. Without this, any Zarr/Dask read on our *own* S3 bucket that
happens after the STS injection would pick up those bucket-scoped
tokens and get AccessDenied.

Usage in the S1 ingest flow::

    from tessera_embeddings.providers.aws.credentials import iam_s3_storage_options
    ...
    roi_storage_options = iam_s3_storage_options() if roi_zarr_path.startswith("s3://") else None
    ingest_s1_roi_sar(..., storage_options=roi_storage_options)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import icechunk
from botocore import session as botocore_session

if TYPE_CHECKING:
    from botocore.credentials import ReadOnlyCredentials


def _resolve_iam_credentials() -> ReadOnlyCredentials:
    """Resolve IAM credentials, skipping the botocore ``env`` provider.

    Removing the ``env`` provider from the credential chain forces resolution
    to the IAM role (instance-metadata on EC2, container credentials on ECS,
    ``~/.aws/credentials`` / SSO locally) — even when ``set_s3_credentials``
    has overridden the ``AWS_*`` env vars with OPERA-scoped STS tokens for
    GDAL reads.

    Raises:
        RuntimeError: If no AWS credentials are found outside env vars.
    """
    bc_session = botocore_session.get_session()
    bc_session.get_component("credential_provider").remove("env")
    creds = bc_session.get_credentials()
    if creds is None:
        raise RuntimeError(
            "No AWS credentials found (checked all providers except env vars). "
            "Ensure an IAM role, instance profile, or ~/.aws/credentials is configured."
        )
    return creds.get_frozen_credentials()


def iam_s3_storage_options() -> dict[str, str]:
    """Resolve IAM credentials as plain strings for ``da.from_zarr`` / fsspec.

    Strips the ``env`` provider from the botocore credential chain so the
    result is always the IAM role (instance-metadata on EC2, container
    credentials on ECS, ``~/.aws/credentials`` / SSO locally) — even when
    ``set_s3_credentials`` has overridden ``AWS_*`` env vars with
    OPERA-scoped STS tokens.

    Returns frozen credential strings that Dask can serialize into the
    task graph without re-invoking the credential chain.

    Raises:
        RuntimeError: If no AWS credentials are found outside env vars.
    """
    frozen = _resolve_iam_credentials()
    opts: dict[str, str] = {"key": frozen.access_key, "secret": frozen.secret_key}
    if frozen.token:
        opts["token"] = frozen.token
    return opts


def iam_icechunk_credentials() -> icechunk.S3StaticCredentials:
    """Resolve IAM credentials as ``S3StaticCredentials`` for icechunk's S3 client.

    The icechunk counterpart to :func:`iam_s3_storage_options`: same
    env-provider-stripping logic, but returns the type icechunk's
    ``get_credentials`` callback expects. Registered process-wide via
    ``zarr_store.set_credentials_provider`` so icechunk writes to our own
    store keep using IAM-role creds even after ``set_s3_credentials`` has
    overwritten the ``AWS_*`` env vars with OPERA-scoped STS tokens.

    icechunk re-invokes the callback on each refresh, so IAM-role tokens
    (instance-metadata, ~1hr TTL) never go stale.

    Raises:
        RuntimeError: If no AWS credentials are found outside env vars.
    """
    frozen = _resolve_iam_credentials()
    return icechunk.S3StaticCredentials(
        access_key_id=frozen.access_key,
        secret_access_key=frozen.secret_key,
        session_token=frozen.token,
    )
