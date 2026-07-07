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

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING

import icechunk
from botocore import session as botocore_session

if TYPE_CHECKING:
    from botocore.credentials import Credentials

# How long icechunk may reuse a returned credential before re-invoking the
# provider. Kept well under any IAM/STS token TTL (instance-metadata ~1h,
# ECS task role ~6h) so each re-invocation lands inside botocore's own
# refresh window — botocore auto-refreshes the underlying token on the next
# get_frozen_credentials() call, so we never serve a stale one.
_ICECHUNK_CRED_TTL = timedelta(minutes=15)


@lru_cache(maxsize=1)
def _resolve_iam_credentials() -> Credentials:
    """Resolve IAM credentials once per process, skipping the ``env`` provider.

    Removing the ``env`` provider from the credential chain forces resolution
    to the IAM role (instance-metadata on EC2, container credentials on ECS,
    ``~/.aws/credentials`` / SSO locally) — even when ``set_s3_credentials``
    has overridden the ``AWS_*`` env vars with OPERA-scoped STS tokens for
    GDAL reads.

    The result is cached for the process lifetime: botocore returns a live
    ``RefreshableCredentials`` for an IAM role, which serves its in-memory
    credential and refreshes itself in the background (with botocore's own IMDS
    retry/backoff). Caching the *session* — rather than building a fresh one per
    call — is what makes that refresh effective: a transient IMDS blip during
    normal operation returns the still-valid cached credential untouched, and
    IMDS is only contacted during botocore's refresh window. A fresh session
    each call would instead do a cold IMDS resolve every time, turning every
    momentary IMDS throttle into a credential failure. ``lru_cache`` does not
    cache exceptions, so a failed cold resolve still retries on the next call.

    Returns the live botocore credentials object (not a frozen snapshot) so
    callers can read the credential expiry off refreshable creds. Call
    ``get_frozen_credentials()`` on the result for a serializable snapshot;
    on refreshable creds that call is what triggers a background refresh.

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
    return creds


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
    frozen = _resolve_iam_credentials().get_frozen_credentials()
    opts: dict[str, str] = {"key": frozen.access_key, "secret": frozen.secret_key}
    if frozen.token:
        opts["token"] = frozen.token
    return opts


def iam_icechunk_credentials() -> icechunk.S3StaticCredentials:
    """Resolve IAM credentials as ``S3StaticCredentials`` for icechunk's S3 client.

    The icechunk counterpart to :func:`iam_s3_storage_options`: same
    env-provider-stripping logic, but returns the type icechunk's
    ``get_credentials`` callback expects. Registered via
    ``zarr_store.credentials_provider`` so icechunk writes to our own
    store keep using IAM-role creds even after ``set_s3_credentials`` has
    overwritten the ``AWS_*`` env vars with OPERA-scoped STS tokens.

    Without ``expires_after`` icechunk treats the static creds as valid
    indefinitely and reuses the first token until S3 returns ``ExpiredToken``
    mid-write. We set a fixed, conservative ``expires_after`` so icechunk
    re-invokes this callback every :data:`_ICECHUNK_CRED_TTL`; each
    re-invocation calls ``get_frozen_credentials()``, which lets botocore
    auto-refresh the underlying IAM/STS token. The window is well under any
    token TTL, so we never serve a stale credential — and we avoid reaching
    into botocore's private ``_expiry_time``.

    Raises:
        RuntimeError: If no AWS credentials are found outside env vars.
    """
    frozen = _resolve_iam_credentials().get_frozen_credentials()
    expires_after = datetime.now(UTC) + _ICECHUNK_CRED_TTL
    return icechunk.S3StaticCredentials(
        access_key_id=frozen.access_key,
        secret_access_key=frozen.secret_key,
        session_token=frozen.token,
        expires_after=expires_after,
    )
