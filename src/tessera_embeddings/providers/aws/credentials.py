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

from botocore import session as botocore_session


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
    bc_session = botocore_session.get_session()
    bc_session.get_component("credential_provider").remove("env")
    creds = bc_session.get_credentials()
    if creds is None:
        raise RuntimeError(
            "No AWS credentials found (checked all providers except env vars). "
            "Ensure an IAM role, instance profile, or ~/.aws/credentials is configured."
        )
    frozen = creds.get_frozen_credentials()
    opts: dict[str, str] = {"key": frozen.access_key, "secret": frozen.secret_key}
    if frozen.token:
        opts["token"] = frozen.token
    return opts
