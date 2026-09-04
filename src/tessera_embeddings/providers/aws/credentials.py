"""AWS credential helpers for the ingest layer.

Functions here resolve IAM credentials while deliberately bypassing env vars — needed when
``set_s3_credentials`` has temporarily overwritten ``AWS_*`` with OPERA-scoped STS credentials
for S3 direct access. Without this, any Zarr/Dask read on our *own* S3 bucket after the STS
injection would pick up those bucket-scoped tokens and get AccessDenied.

Usage in the S1 ingest flow::

    from tessera_embeddings.providers.aws.credentials import iam_s3_storage_options
    ...
    roi_storage_options = iam_s3_storage_options() if roi_zarr_path.startswith("s3://") else None
    ingest_s1_roi_sar(..., storage_options=roi_storage_options)
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, final

import icechunk
from botocore import session as botocore_session

if TYPE_CHECKING:
    from botocore.credentials import Credentials

# How long icechunk may reuse a returned credential before re-invoking the provider. Kept well
# under any IAM/STS token TTL (instance-metadata ~1h, ECS task role ~6h) so each re-invocation
# lands inside botocore's own refresh window — botocore auto-refreshes the underlying token on
# the next get_frozen_credentials() call, so we never serve a stale one.
_ICECHUNK_CRED_TTL = timedelta(minutes=15)

_LOG = logging.getLogger(__name__)

#: The same interval for an ASSUMED-ROLE credential, and shorter for a measured reason.
#:
#: A role assumed from a role is *chained*, which AWS caps at one hour whatever the target role's
#: MaxSessionDuration says. Botocore refreshes such a credential advisory at 900s before expiry and
#: MANDATORY at 600s. At a 15-minute cadence the callbacks in a one-hour session land at 0, 15, 30,
#: 45 and 60 minutes: the 45-minute call trips only the ADVISORY refresh, which serves the old
#: still-valid credential if STS hiccups, and the next is at 60 — exactly expiry, so the mandatory
#: window (from minute 50) is stepped over entirely and any clock skew or in-flight request meets
#: `ExpiredToken` mid-write. Five minutes puts calls at 45, 50 and 55, so at least one lands inside
#: the mandatory window and forces a blocking refresh; the cost is twelve in-memory reads an hour
#: rather than four, at a few microseconds each (the STS call itself is botocore's decision).
_ASSUMED_ROLE_CRED_TTL = timedelta(minutes=5)

#: Never promise icechunk a credential for longer than the real token has left, minus this. The
#: interval above is only half the protection — it relies on two independently chosen constants
#: staying in a sensible relationship — whereas capping by the ACTUAL expiry cannot be defeated
#: that way. Two minutes is comfortably more than a refresh round trip and far less than the
#: mandatory window it sits inside.
_EXPIRY_SAFETY_MARGIN = timedelta(minutes=2)


def _expires_after(creds: Credentials, ttl: timedelta) -> datetime:
    """When icechunk should come back: ``now + ttl``, but never past the token's own life.

    Refreshable credentials expose ``_expiry_time``; a static one has none, in which case the
    interval alone is all there is to go on. Reading a private attribute is deliberate and narrow:
    botocore exposes no public expiry, and the alternative is promising validity we cannot support.
    """
    horizon = datetime.now(UTC) + ttl
    expiry = getattr(creds, "_expiry_time", None)
    if expiry is None:
        return horizon
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    # NORMALISED TO datetime.timezone.utc, not merely "some UTC tzinfo". Botocore parses the STS
    # expiry with dateutil, so a refreshable credential's `_expiry_time` carries `tzutc()` — an
    # object that compares equal and formats identically but which icechunk rejects outright with
    # "expected datetime.timezone.utc", failing the first store write whenever the real expiry is
    # the nearer of the two. Only the assumed-role path has an expiry to read.
    return min(horizon, expiry - _EXPIRY_SAFETY_MARGIN).astimezone(UTC)


@lru_cache(maxsize=1)
def _resolve_iam_credentials() -> Credentials:
    """Resolve IAM credentials once per process, skipping the ``env`` provider.

    Removing the ``env`` provider from the credential chain forces resolution to the IAM role
    (instance-metadata on EC2, container credentials on ECS, ``~/.aws/credentials`` / SSO locally)
    — even when ``set_s3_credentials`` has overridden the ``AWS_*`` env vars with OPERA-scoped STS
    tokens for GDAL reads.

    The result is cached for the process lifetime: botocore returns a live
    ``RefreshableCredentials`` for an IAM role, which serves its in-memory credential and refreshes
    itself in the background (with botocore's own IMDS retry/backoff). Caching the *session* is
    what makes that refresh effective — a transient IMDS blip returns the still-valid cached
    credential untouched and IMDS is only contacted during botocore's refresh window, whereas a
    fresh session each call would do a cold IMDS resolve every time, turning every momentary IMDS
    throttle into a credential failure. ``lru_cache`` does not cache exceptions, so a failed cold
    resolve still retries on the next call.

    Returns the live botocore credentials object (not a frozen snapshot) so callers can read the
    credential expiry off refreshable creds. Call ``get_frozen_credentials()`` on the result for a
    serializable snapshot; on refreshable creds that call is what triggers a background refresh.

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


@lru_cache(maxsize=4)
def _assumed_role_credentials(role_arn: str, external_id: str | None = None) -> Credentials:
    """A self-refreshing credential for ``role_arn``, built once per process.

    Mirrors :func:`_resolve_iam_credentials` deliberately: build the credential object ONCE and let
    botocore refresh it underneath, rather than calling ``AssumeRole`` on every access. The reason
    is not the cost of the call — an STS round trip is tens of milliseconds against cells that run
    for hours — but what a fresh resolve does to a transient failure: a cached refreshable
    credential serves its still-valid in-memory token through an STS blip and only reaches out
    inside its own refresh window, while a fresh resolve turns that blip into a failed write.

    The refresh callback re-assumes, so the credential renews indefinitely even though each session
    is capped at one hour by AWS's role-chaining rule. ``lru_cache`` does not cache exceptions, so
    a failed cold assume retries on the next call.
    """
    from botocore.credentials import DeferredRefreshableCredentials

    # The BASE identity does the assuming — the task role, resolved with env stripped for the same
    # reason as everywhere else here: `set_s3_credentials` may have overwritten AWS_* with
    # OPERA-scoped tokens, and those cannot assume anything of ours.
    bc_session = botocore_session.get_session()
    bc_session.get_component("credential_provider").remove("env")
    sts = bc_session.create_client("sts")

    def _assume() -> dict[str, str]:
        kwargs: dict[str, object] = {
            "RoleArn": role_arn,
            # Identifies US in the target account's CloudTrail. A fixed name rather than a random
            # one so their audit log groups our sessions instead of showing one principal per hour.
            "RoleSessionName": "tessera-published-store-writer",
        }
        if external_id:
            kwargs["ExternalId"] = external_id
        creds = sts.assume_role(**kwargs)["Credentials"]
        return {
            "access_key": creds["AccessKeyId"],
            "secret_key": creds["SecretAccessKey"],
            "token": creds["SessionToken"],
            "expiry_time": creds["Expiration"].isoformat(),
        }

    # DEFERRED so constructing this costs no network call: the first assume happens on first use,
    # keeping a process that never touches the published store from needing the permission at all.
    return DeferredRefreshableCredentials(refresh_using=_assume, method="sts-assume-role")


@final
@dataclass(frozen=True)
class AssumedRoleIcechunkCredentials:
    """An icechunk credential callback that writes as ``role_arn``.

    For the published store, whose bucket belongs to the delivery partner: the write permission
    lives in THEIR account on a role we assume, so this is the only credential that can write it —
    and it cannot read our own buckets, which is why the choice is made per destination rather than
    per process (see :func:`icechunk_credentials_for`).

    **A class rather than a closure because icechunk PICKLES the callback.** It hands it to its
    Rust S3 client and ships it to every process that deserialises a ``Storage``, so a nested
    function fails at construction with "Can't get local object" — before any credential is
    fetched, and identically whether or not the grant works. A frozen dataclass holding just the
    role identifiers pickles by value; the refreshable credential behind it is rebuilt on arrival,
    keyed by those same identifiers, so each process gets its own and refreshes independently. The
    task-role callback survives only because it is a module-level function.

    Calling it is cheap and repeatable: it reads an already-resolved credential while botocore
    refreshes underneath. The expiry it reports is capped by the token's real life
    (:func:`_expires_after`), so icechunk comes back before the credential dies rather than
    discovering the fact as a failed write.
    """

    role_arn: str
    external_id: str | None = None

    def __call__(self) -> icechunk.S3StaticCredentials:
        """A freshly-read credential for the assumed role, with its capped expiry."""
        creds = _assumed_role_credentials(self.role_arn, self.external_id)
        frozen = creds.get_frozen_credentials()
        # Through the shared boundary, like the task-role provider: THIS is the callback that
        # serves the published store, so instrumenting only the other one leaves the credential
        # behind an assembly failure invisible.
        return _serve_icechunk_credential(
            # The FULL arn, not the role name. The same role name in two partner accounts is a
            # real configuration, so collapsing them would both mislabel the log line and merge
            # two providers' rotation history into one map entry, making a rotation on either
            # read as a rotation on both. The arn carries the account id, which is also what
            # makes the line correlatable against the other account's CloudTrail. An external id
            # never disambiguates an arn (one role has one), so it is recorded as presence only
            # and its value is never logged.
            source=f"assumed-role:{self.role_arn}" + ("+external-id" if self.external_id else ""),
            access_key=frozen.access_key,
            secret_key=frozen.secret_key,
            session_token=frozen.token,
            expires_after=_expires_after(creds, _ASSUMED_ROLE_CRED_TTL),
        )


def assumed_role_icechunk_credentials(
    role_arn: str, external_id: str | None = None
) -> Callable[[], icechunk.S3StaticCredentials]:
    """Build the published store's credential callback. See :class:`AssumedRoleIcechunkCredentials`."""
    return AssumedRoleIcechunkCredentials(role_arn, external_id)


#: Env vars the runner's task definition carries when this deployment publishes to a bucket that is
#: not ours: the role to assume, and an optional external id if the owner requires one.
#: Configuration rather than an argument because the destination decides, and every process that
#: touches the store — flow runner, each assembly worker — must reach the same conclusion untold.
_WRITER_ROLE_ENV = "PUBLISHED_STORE_WRITER_ROLE_ARN"
_WRITER_EXTERNAL_ID_ENV = "PUBLISHED_STORE_WRITER_EXTERNAL_ID"


def published_store_writer_role() -> tuple[str, str | None] | None:
    """The role to assume for the published store, or ``None`` when there is nothing to assume."""
    arn = (os.environ.get(_WRITER_ROLE_ENV) or "").strip()
    if not arn:
        return None
    return arn, (os.environ.get(_WRITER_EXTERNAL_ID_ENV) or "").strip() or None


def icechunk_credentials_for(
    uri: str, default: Callable[[], icechunk.S3StaticCredentials] | None = None
) -> Callable[[], icechunk.S3StaticCredentials]:
    """The credential callback for ``uri`` — assumed role for the published store, task role elsewhere.

    **Chosen from the destination, never from the caller.** One fill legitimately spans two
    accounts: it reads mosaics and staged tiles from our buckets and writes embeddings to the
    partner's. The assumed role is scoped to two prefixes in their bucket and cannot read ours, so
    handing it to everything breaks every other access — and the task role cannot write the
    published store. Deriving the answer from where the bytes are going is what makes both work
    without any call site having to know which case it is in. Same shape as the registry mistake
    one layer down (``context_docs/storage/writing-to-the-global-store.md``), where an artifact took
    its location from a default rather than the resolved target and published beside the wrong
    store; here the resolved target is the argument.

    Returns the ordinary task-role provider whenever no writer role is configured, which is every
    deployment whose store lives in our own buckets — so dev behaviour is unchanged by construction.
    """
    configured = published_store_writer_role()
    if configured and _is_published_store(uri):
        return assumed_role_icechunk_credentials(*configured)
    # The caller's own provider is PRESERVED rather than replaced: it is the right credential for
    # our own buckets, and in tests it is a double — substituting the real task-role provider here
    # would quietly undo both.
    return default or iam_icechunk_credentials


def _is_published_store(uri: str) -> bool:
    """Whether ``uri`` addresses the externally-owned published store or its registry sibling.

    Keyed on the BUCKET, not on a prefix list. The store and its registry are siblings under one
    externally-owned bucket, and a prefix list would have to be kept in step with
    ``BucketPaths.optical_registry`` — which derives one location from the other precisely so the
    two cannot drift. Anything we are asked to write in that bucket needs the assumed role; there
    is nothing else of ours there.
    """
    if not uri.startswith("s3://"):
        return False
    bucket = uri[len("s3://") :].split("/", 1)[0]
    return bucket == PARTNER_DELIVERY_BUCKET


#: The externally-owned bucket the campaign publishes into. Named here rather than imported from the
#: infrastructure definition because that lives in a separate uv project the library cannot import;
#: the two are kept in hand-lockstep, and the test below pins the pairing.
PARTNER_DELIVERY_BUCKET = "tessera-embeddings"


def iam_s3_storage_options() -> dict[str, str]:
    """Resolve IAM credentials as plain strings for ``da.from_zarr`` / fsspec.

    Strips the ``env`` provider from the botocore credential chain so the result is always the IAM
    role (instance-metadata on EC2, container credentials on ECS, ``~/.aws/credentials`` / SSO
    locally) — even when ``set_s3_credentials`` has overridden ``AWS_*`` env vars with OPERA-scoped
    STS tokens. Returns frozen credential strings, which Dask can serialize into a task graph.

    **Pass this function to the ingest, not its result.** What it returns is a snapshot, and the
    role credential behind it expires (instance-metadata ~1h, ECS task role ~6h) — a leg can run
    longer, so options resolved once at leg entry stop working partway through, on a bucket the
    role can always read. Every consumer accepts a callable for exactly this reason (see
    :func:`tessera_embeddings.ingest.roi.resolve_storage_options`) and re-invokes it at each read.
    That is cheap: :func:`_resolve_iam_credentials` caches a live refreshable credential, so each
    call is a frozen copy of an already-resolved one and botocore refreshes underneath.

    Raises:
        RuntimeError: If no AWS credentials are found outside env vars.
    """
    frozen = _resolve_iam_credentials().get_frozen_credentials()
    opts: dict[str, str] = {"key": frozen.access_key, "secret": frozen.secret_key}
    if frozen.token:
        opts["token"] = frozen.token
    return opts


#: Access keys this process has ANNOUNCED, per source. Keyed by source because one process
#: legitimately uses both providers -- a fill reads our buckets with the task role and writes the
#: partner's with an assumed role -- and one slot would read the alternation as constant rotation.
#: A set, not a sequence: see :func:`_serve_icechunk_credential` on why no ordering claim made here
#: can be trusted. Bounded by the number of real rotations a process lives through.
_announced_access_keys: dict[str, set[str]] = {}
#: Stamped inside the lock, so it totally orders the SERVE DECISIONS. Records are emitted outside
#: the lock and can reach the stream out of order; this recovers that order. It is NOT the order
#: the credentials were used: a thread can take its number and be descheduled before returning,
#: while a later one returns and its credential is used first.
_announce_sequence = itertools.count(1)
_ANNOUNCE_LOCK = threading.Lock()


def _serve_icechunk_credential(
    *,
    source: str,
    access_key: str,
    secret_key: str,
    session_token: str | None,
    expires_after: datetime,
) -> icechunk.S3StaticCredentials:
    """Build the credential icechunk will use, and announce it once per process.

    The single construction point for every icechunk credential in this module, so BOTH providers
    are visible -- the task role and the published store's assumed role; instrumenting either alone
    leaves the other's failures unexplained. Records the access-key id, which is an identifier
    rather than a secret and is what CloudTrail indexes on, plus the pid and a sequence number.
    **Never the secret key or the session token.** Background:
    ``context_docs/storage/writing-to-the-global-store.md``.

    Announces NOVELTY, never a transition: nothing reaching here records when a credential was
    ACQUIRED (the expiry is computed after it is frozen), so a thread descheduled past a refresh
    arrives with an old key bearing a LATER expiry, and any "which is newer" test is wrong in the
    one case it exists for. The INFO lines for one pid in ``seq`` order are therefore the
    credentials this process served in the order it DECIDED to serve them -- not necessarily the
    order they were USED. A late callback costs at most one extra line naming a credential the
    process genuinely served.

    ``source`` must be fully qualified (the whole role arn): a role NAME is not unique -- the same
    name in two partner accounts is a real configuration -- and merging them would report one
    provider's rotation as the other's.

    The timestamp is icechunk's REUSE deadline, not the token's validity limit -- the task-role
    path passes ``now + TTL`` outright, and only the assumed-role path caps it by the real expiry.
    Labelled "reuse until" so a responder does not read it as when the credential died.

    Levels: a process's first credential and each further distinct one are INFO, the events a
    signature failure is lined up against; routine re-serves are DEBUG, because icechunk asks per
    request once its cached credential lapses (icechunk#2077) and a campaign runs on the order of a
    hundred clients. Where a caller scatters, a spawned worker inherits a credential without asking
    and its first call lands only after that lapses -- so a child's line names that possibility
    rather than implying a start.
    """
    with _ANNOUNCE_LOCK:  # bookkeeping only: logging in here would serialise every request
        announced = _announced_access_keys.setdefault(source, set())
        novel, first = access_key not in announced, not announced
        announced.add(access_key)
        order = next(_announce_sequence)

    if not novel:
        level, label = logging.DEBUG, "served"
    elif not first:
        level, label = logging.INFO, "NEW"
    elif multiprocessing.parent_process() is None:
        level, label = logging.INFO, "FIRST"
    else:
        # `parent_process()` says only that this is a multiprocessing child -- a Dask nanny's worker
        # is one too, and its storage may never have scattered. So the line names the possibility
        # rather than asserting inheritance, which would be wrong on those paths.
        level, label = (
            logging.INFO,
            (
                "FIRST in this child process (if it inherited a scattered credential the parent "
                "announced that one, making this a change and not a start)"
            ),
        )
    _LOG.log(
        level,
        "icechunk credential [%s] pid=%d seq=%d %s: %s, reuse until %s",
        source,
        os.getpid(),
        order,
        label,
        access_key,
        expires_after.isoformat(timespec="seconds"),
    )
    return icechunk.S3StaticCredentials(
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token=session_token,
        expires_after=expires_after,
    )


def iam_icechunk_credentials() -> icechunk.S3StaticCredentials:
    """Resolve IAM credentials as ``S3StaticCredentials`` for icechunk's S3 client.

    The icechunk counterpart to :func:`iam_s3_storage_options`: same env-provider-stripping logic,
    but returns the type icechunk's ``get_credentials`` callback expects. Registered via
    ``zarr_store.credentials_provider`` so icechunk writes to our own store keep using IAM-role
    creds even after ``set_s3_credentials`` has overwritten the ``AWS_*`` env vars with OPERA-scoped
    STS tokens.

    Without ``expires_after`` icechunk treats the static creds as valid indefinitely and reuses the
    first token until S3 returns ``ExpiredToken`` mid-write. A fixed, conservative ``expires_after``
    makes icechunk re-invoke this callback every :data:`_ICECHUNK_CRED_TTL`; each re-invocation
    calls ``get_frozen_credentials()``, which lets botocore auto-refresh the underlying IAM/STS
    token. The window is well under any token TTL, so we never serve a stale credential — and this
    path needs no read of botocore's private ``_expiry_time`` (only the assumed-role path, capped at
    an hour by role chaining, reads it, via :func:`_expires_after`).

    Raises:
        RuntimeError: If no AWS credentials are found outside env vars.
    """
    frozen = _resolve_iam_credentials().get_frozen_credentials()
    return _serve_icechunk_credential(
        source="task-role",
        access_key=frozen.access_key,
        secret_key=frozen.secret_key,
        session_token=frozen.token,
        expires_after=datetime.now(UTC) + _ICECHUNK_CRED_TTL,
    )
