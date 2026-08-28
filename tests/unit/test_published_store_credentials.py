"""Credentials for the published store, which lives in a bucket that is not ours.

The delivery partner grants write through a role in THEIR account rather than through their bucket
policy, so writing the published store means assuming that role. One fill spans both accounts — it
reads mosaics and staged tiles from our buckets and writes embeddings to theirs — so the credential
has to be chosen from the destination. These tests pin that choice and the two properties that keep a
long assembly from dying on an expired token.

Integration is not testable here: assuming the role needs our deployed policy AND the partner's trust
policy naming our runner role, and AWS returns the same denial whichever half is missing. So what is
covered is the arithmetic and the branching, which is where the mistakes actually live.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import icechunk
import pytest
from dateutil.tz import tzutc

from tessera_embeddings.providers.aws import credentials as cred
from tessera_embeddings.providers.aws import credentials as creds_mod

_OURS = "s3://global-tessera-embeddings/global-tessera/global/tessera.icechunk"
_THEIRS = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
_THEIR_REGISTRY = "s3://tessera-embeddings/v1.1/dclimate.registry/parts/zone=16S/year=2021/r.parquet"
_ROLE = "arn:aws:iam::601791338954:role/TesseraDClimateWriterRole"


class _Refreshable:
    """Stands in for botocore's refreshable credentials — only the expiry matters here."""

    def __init__(self, minutes_left: float) -> None:
        self._expiry_time = datetime.now(UTC) + timedelta(minutes=minutes_left)


class _Static:
    """A credential with no expiry at all, as a static one has."""


class TestTheExpiryWePromiseIcechunk:
    """Icechunk reuses a credential until the moment we said it expires, so that promise must never
    outlast the token. Before `expires_after` existed at all it reused the FIRST token until S3
    answered `ExpiredToken` mid-write, which is the failure this whole mechanism exists to stop.
    """

    def test_the_interval_binds_while_the_token_has_plenty_left(self) -> None:
        out = cred._expires_after(_Refreshable(55), cred._ASSUMED_ROLE_CRED_TTL)
        assert 4.5 < (out - datetime.now(UTC)).total_seconds() / 60 < 5.5

    def test_the_real_expiry_binds_when_it_is_sooner(self) -> None:
        """The structural half. The interval alone relies on two independently chosen constants
        staying in a sensible relationship; this cannot be defeated that way.
        """
        out = cred._expires_after(_Refreshable(6), cred._ASSUMED_ROLE_CRED_TTL)
        left = (out - datetime.now(UTC)).total_seconds() / 60
        assert 3.5 < left < 4.5, "6 minutes left, minus a 2-minute margin"

    def test_we_never_promise_past_the_token_even_when_it_is_nearly_dead(self) -> None:
        out = cred._expires_after(_Refreshable(1), cred._ASSUMED_ROLE_CRED_TTL)
        assert out < datetime.now(UTC), "already inside the margin, so icechunk must come back at once"

    def test_a_naive_expiry_is_not_compared_against_an_aware_one(self) -> None:
        """A naive datetime raises on comparison with an aware one, which would surface as a crash in
        the credential path rather than anywhere near the cause.
        """
        c = _Refreshable(30)
        c._expiry_time = c._expiry_time.replace(tzinfo=None)
        assert cred._expires_after(c, cred._ASSUMED_ROLE_CRED_TTL) is not None

    def test_a_credential_with_no_expiry_falls_back_to_the_interval(self) -> None:
        out = cred._expires_after(_Static(), cred._ASSUMED_ROLE_CRED_TTL)
        assert 4.5 < (out - datetime.now(UTC)).total_seconds() / 60 < 5.5


def test_the_interval_lands_inside_botocores_mandatory_refresh_window() -> None:
    """The measured reason the assumed-role interval is shorter than the task-role one.

    A chained role session is capped at one hour by AWS. Botocore refreshes advisory at 900s before
    expiry and MANDATORY at 600s — mandatory being the blocking one that guarantees a fresh token. At
    the 15-minute task-role cadence the calls in a one-hour session land at 0/15/30/45/60, so the
    45-minute call trips only the advisory refresh (which serves the old token if STS hiccups) and the
    next call is at 60: exactly expiry. The mandatory window is stepped over entirely.

    Asserted against botocore's own constant rather than a copied number, so if botocore moves its
    threshold this fails instead of quietly becoming wrong.
    """
    from botocore.credentials import RefreshableCredentials

    mandatory = timedelta(seconds=RefreshableCredentials._mandatory_refresh_timeout)
    assert mandatory > cred._ASSUMED_ROLE_CRED_TTL, (
        "at least one callback must land inside the mandatory window, so the interval must be "
        "shorter than the window itself"
    )
    assert mandatory <= cred._ICECHUNK_CRED_TTL, (
        "documents the contrast: the task-role interval is NOT shorter than the window, which is "
        "safe only because that credential lives ~6h rather than 1h"
    )
    assert cred._EXPIRY_SAFETY_MARGIN < cred._ASSUMED_ROLE_CRED_TTL, (
        "the margin must not exceed the interval, or every promise would already be in the past"
    )


class TestWhichCredentialAUriGets:
    """Chosen from the destination, never from the caller. The assumed role is scoped to two prefixes
    in the partner's bucket and cannot read ours, so a process-wide swap breaks every other access.
    """

    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch) -> None:
        monkeypatch.delenv(cred._WRITER_ROLE_ENV, raising=False)
        monkeypatch.delenv(cred._WRITER_EXTERNAL_ID_ENV, raising=False)
        cred._assumed_role_credentials.cache_clear()

    def test_with_nothing_configured_everything_uses_the_task_role(self) -> None:
        """Which is every deployment whose store is in our own buckets. Dev is unchanged by
        construction rather than by a separate code path.
        """
        for uri in (_OURS, _THEIRS, _THEIR_REGISTRY, "/local/path.icechunk"):
            assert cred.icechunk_credentials_for(uri) is cred.iam_icechunk_credentials

    def test_only_the_partners_bucket_gets_the_assumed_role(self, monkeypatch) -> None:
        monkeypatch.setenv(cred._WRITER_ROLE_ENV, _ROLE)
        assert cred.icechunk_credentials_for(_OURS) is cred.iam_icechunk_credentials
        assert cred.icechunk_credentials_for(_THEIRS) is not cred.iam_icechunk_credentials
        assert cred.icechunk_credentials_for(_THEIR_REGISTRY) is not cred.iam_icechunk_credentials

    def test_a_local_path_never_assumes_anything(self, monkeypatch) -> None:
        """Tests and local runs address a filesystem path, and there is nothing to assume for one."""
        monkeypatch.setenv(cred._WRITER_ROLE_ENV, _ROLE)
        assert cred.icechunk_credentials_for("/tmp/x.icechunk") is cred.iam_icechunk_credentials

    def test_a_bucket_whose_name_merely_contains_theirs_is_not_theirs(self, monkeypatch) -> None:
        """`tessera-embeddings` is a prefix of our own `tessera-embeddings-dev`-shaped names, so a
        substring test would hand our bucket a credential that cannot read it.
        """
        monkeypatch.setenv(cred._WRITER_ROLE_ENV, _ROLE)
        for uri in (
            "s3://tessera-embeddings-dev/v1.1/x",
            "s3://not-tessera-embeddings/v1.1/x",
            "s3://tessera-embeddings2/v1.1/x",
        ):
            assert cred.icechunk_credentials_for(uri) is cred.iam_icechunk_credentials, uri

    def test_the_external_id_is_carried_when_the_owner_requires_one(self, monkeypatch) -> None:
        monkeypatch.setenv(cred._WRITER_ROLE_ENV, _ROLE)
        monkeypatch.setenv(cred._WRITER_EXTERNAL_ID_ENV, "shared-secret")
        assert cred.published_store_writer_role() == (_ROLE, "shared-secret")

    def test_an_empty_external_id_is_absent_rather_than_blank(self, monkeypatch) -> None:
        """An empty string would be sent to STS as an ExternalId of "", which fails against a trust
        policy that requires none — different from not sending the field at all.
        """
        monkeypatch.setenv(cred._WRITER_ROLE_ENV, _ROLE)
        monkeypatch.setenv(cred._WRITER_EXTERNAL_ID_ENV, "   ")
        assert cred.published_store_writer_role() == (_ROLE, None)

    def test_a_blank_role_reads_as_unconfigured(self, monkeypatch) -> None:
        """The task definition may carry the variable with an empty value on a deployment that does
        not publish externally, and that must mean "no role" rather than an attempt to assume "".
        """
        monkeypatch.setenv(cred._WRITER_ROLE_ENV, "  ")
        assert cred.published_store_writer_role() is None
        assert cred.icechunk_credentials_for(_THEIRS) is cred.iam_icechunk_credentials


class TestTheAssumedRoleProviderItself:
    """Built once per process and refreshed underneath, rather than re-assumed per call."""

    @pytest.fixture(autouse=True)
    def _clear(self) -> None:
        cred._assumed_role_credentials.cache_clear()

    def test_building_the_callback_makes_no_network_call(self) -> None:
        """Deferred on purpose: a process that never touches the published store must not need the
        permission, and building a provider must not be able to fail.
        """
        assert callable(cred.assumed_role_icechunk_credentials(_ROLE))

    def test_the_callback_assumes_and_reports_a_capped_expiry(self, monkeypatch) -> None:
        calls: list[dict] = []
        expiry = datetime.now(UTC) + timedelta(hours=1)

        class _Sts:
            def assume_role(self, **kwargs):
                calls.append(kwargs)
                return {
                    "Credentials": {
                        "AccessKeyId": "AKIA_TEST",
                        "SecretAccessKey": "secret",
                        "SessionToken": "token",
                        "Expiration": expiry,
                    }
                }

        class _Session:
            def get_component(self, _name):
                return type("P", (), {"remove": lambda self, _n: None})()

            def create_client(self, _name):
                return _Sts()

        monkeypatch.setattr(cred.botocore_session, "get_session", lambda: _Session())

        out = cred.assumed_role_icechunk_credentials(_ROLE)()
        assert out.access_key_id == "AKIA_TEST"
        assert out.session_token == "token"
        assert len(calls) == 1, "assumed once"
        assert calls[0]["RoleArn"] == _ROLE
        assert "ExternalId" not in calls[0], "omitted, not sent empty"
        assert out.expires_after < expiry, "never promised past the token's own life"

    def test_repeated_calls_do_not_re_assume(self, monkeypatch) -> None:
        """The whole reason the credential object is cached: a fresh resolve per call turns a
        transient STS failure into a failed write, where a cached refreshable one rides it out.
        """
        calls: list[dict] = []

        class _Sts:
            def assume_role(self, **kwargs):
                calls.append(kwargs)
                return {
                    "Credentials": {
                        "AccessKeyId": "AKIA_TEST",
                        "SecretAccessKey": "secret",
                        "SessionToken": "token",
                        "Expiration": datetime.now(UTC) + timedelta(hours=1),
                    }
                }

        class _Session:
            def get_component(self, _name):
                return type("P", (), {"remove": lambda self, _n: None})()

            def create_client(self, _name):
                return _Sts()

        monkeypatch.setattr(cred.botocore_session, "get_session", lambda: _Session())
        get = cred.assumed_role_icechunk_credentials(_ROLE)
        for _ in range(5):
            get()
        assert len(calls) == 1, f"assumed {len(calls)} times for 5 reads"


class TestIcechunkCanActuallyUseTheCallback:
    """The two properties icechunk enforces at CONSTRUCTION, before any credential is fetched.

    Both were violated by the first version of this provider, and neither was reachable by a test that
    checked what the callback returns: they are about the callback OBJECT and about the tzinfo TYPE of
    the instant it reports. Because the grant itself was blocked, nothing exercised either one — a
    passing suite and a working grant would still have failed on the first store write.
    """

    def test_the_callback_survives_the_pickling_icechunk_does_to_it(self) -> None:
        """A closure fails here with "Can't get local object" — before any credential is fetched.

        icechunk pickles the callback to reach its Rust S3 client and to ship it to every process that
        deserialises a Storage, so the provider has to be picklable by value.
        """
        import pickle

        provider = cred.assumed_role_icechunk_credentials(_ROLE)
        revived = pickle.loads(pickle.dumps(provider))

        assert revived == provider, "a frozen dataclass round-trips by value"
        assert revived.role_arn == _ROLE

    def test_icechunk_accepts_the_provider_when_building_storage(self) -> None:
        """The end-to-end acceptance criterion, and no network: construction alone rejects a closure."""
        storage = icechunk.s3_storage(
            bucket="tessera-embeddings",
            prefix="v1.1/dclimate.icechunk/",
            region="us-west-2",
            get_credentials=cred.assumed_role_icechunk_credentials(_ROLE),
        )
        assert storage is not None

    def test_the_expiry_is_datetime_timezone_utc_even_when_botocore_says_tzutc(self) -> None:
        """Not merely "a UTC offset" — icechunk requires that exact tzinfo object.

        botocore parses the STS expiry with dateutil, so a refreshable credential's ``_expiry_time``
        carries ``tzutc()``: equal to and indistinguishable from ``timezone.utc`` in comparisons and
        formatting, and rejected by icechunk. It only bites when the real expiry is the nearer of the
        two bounds, which is exactly the case the cap exists to produce.
        """

        class _DateutilRefreshable:
            _expiry_time = datetime.now(tzutc()) + timedelta(minutes=45)

        expires = cred._expires_after(_DateutilRefreshable(), timedelta(minutes=60))

        assert expires.tzinfo is UTC, f"icechunk requires datetime.timezone.utc, got {expires.tzinfo!r}"
        # The real proof: the value is one icechunk will take.
        icechunk.S3StaticCredentials(access_key_id="a", secret_access_key="b", session_token="c", expires_after=expires)

    def test_the_cap_still_bites_after_normalisation(self) -> None:
        """Normalising the tzinfo must not quietly stop the expiry from bounding the promise."""

        class _DateutilRefreshable:
            _expiry_time = datetime.now(tzutc()) + timedelta(minutes=10)

        expires = cred._expires_after(_DateutilRefreshable(), timedelta(minutes=60))
        assert expires < datetime.now(UTC) + timedelta(minutes=10), "the token's own life still wins"


class TestTheWiringReachesTheStore:
    """The selector existing is not the same as anything using it.

    `icechunk_credentials_for` was built, tested and then called from nowhere in runtime code,
    so with a writer role configured the seed failed with AccessDenied before the store could be
    created, the fill failed on its first store write, and the registry publication failed into
    a handler that suppresses it — leaving a campaign that completes green having published no
    registry at all. These pin that the production call sites reach the selector.
    """

    @staticmethod
    def _configured(monkeypatch):
        monkeypatch.setenv("PUBLISHED_STORE_WRITER_ROLE_ARN", "arn:aws:iam::111122223333:role/writer")
        monkeypatch.setattr(cred, "assumed_role_icechunk_credentials", lambda *a, **k: lambda: "ASSUMED", raising=True)

    def test_the_runner_picks_the_store_credential_from_the_store_path(self, monkeypatch) -> None:
        """The runner resolves per call site rather than taking a second parameter, so this is
        what proves a store open gets the assumed role while our buckets keep the task role.
        """
        from tessera_embeddings.orchestration.runners import zone_fill

        self._configured(monkeypatch)
        ours = lambda: "TASK_ROLE"  # noqa: E731 — a stand-in callback, identity is what is asserted

        assert zone_fill._store_credentials(_THEIRS, ours)() == "ASSUMED"
        assert zone_fill._store_credentials(_OURS, ours) is ours, "our own buckets keep the task role"
        assert zone_fill._store_credentials("/tmp/local.icechunk", ours) is ours, "and so do local paths"

    def test_an_unconfigured_deployment_is_untouched(self, monkeypatch) -> None:
        """Every deployment whose store is in our own buckets must behave exactly as before."""
        from tessera_embeddings.orchestration.runners import zone_fill

        monkeypatch.delenv("PUBLISHED_STORE_WRITER_ROLE_ARN", raising=False)
        ours = lambda: "TASK_ROLE"  # noqa: E731

        for uri in (_OURS, _THEIRS, _THEIR_REGISTRY):
            assert zone_fill._store_credentials(uri, ours) is ours, uri

    def test_the_registry_write_is_given_the_stores_credential(self) -> None:
        """The registry is a sibling of the store in the SAME bucket, so it needs the same role.

        Asserted on the signature rather than by writing: `publish_registry_part` builds its
        filesystem from `plain_zarr_storage_options(uri, get_credentials, s3_region)`, and before
        this it called `_fs_for(uri)` bare — falling back to fsspec's ambient chain, which is the
        task role and cannot write there.
        """
        import inspect

        from tessera_embeddings.inference.assembly import ZarrWriter

        params = inspect.signature(ZarrWriter.publish_registry_part).parameters
        assert "get_credentials" in params and "s3_region" in params

        src = inspect.getsource(ZarrWriter.publish_registry_part)
        assert "plain_zarr_storage_options(uri, get_credentials, s3_region)" in src
        assert "_fs_for(uri)" not in src, "a bare _fs_for falls back to the ambient chain"


# ---------------------------------------------------------------------------
# Making the credential icechunk uses observable.
# ---------------------------------------------------------------------------


class TestIcechunkCredentialVisibility:
    """Why this exists: on 2026-08-28 the fleet took 516 `SignatureDoesNotMatch` refusals
    across 1h45m — 494 on ingest, 22 on assembly, the latter costing four cells.

    The providers proved CORRECT on inspection, so the gap was never a defect to fix: nothing
    recorded WHICH credential was in play, so a recurrence could only be inferred from the
    shape of the failure rather than lined up against a rotation.
    """

    @staticmethod
    def _reset() -> None:
        creds_mod._announced_access_keys.clear()

    def _serve(self, source: str, key: str, expires_after: datetime | None = None) -> None:
        creds_mod._serve_icechunk_credential(
            source=source,
            access_key=key,
            secret_key="s",
            session_token="t",
            expires_after=expires_after or datetime(2026, 8, 28, 12, tzinfo=UTC),
        )

    def test_the_first_credential_a_process_serves_is_info(self, caplog) -> None:
        """A process that fails before it ever rotates — the common case for these failures —
        would otherwise carry no access-key id at production log level, and could not be
        correlated against CloudTrail.
        """
        self._reset()
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            self._serve("task-role", "ASIAFIRST")
        assert [r.levelname for r in caplog.records] == ["INFO"]
        assert "FIRST" in caplog.records[0].message

    def test_a_steady_reserve_drops_to_debug(self, caplog) -> None:
        """Icechunk re-invokes the callback every TTL per client and a campaign runs on the
        order of a hundred clients, so this is the one that would be a drip saying nothing.
        """
        self._reset()
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            self._serve("task-role", "ASIASTEADY")
            self._serve("task-role", "ASIASTEADY")
            self._serve("task-role", "ASIASTEADY")
        assert [r.levelname for r in caplog.records] == ["INFO", "DEBUG", "DEBUG"]

    def test_a_rotation_is_info(self, caplog) -> None:
        self._reset()
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            self._serve("task-role", "ASIAOLD")
            self._serve("task-role", "ASIANEW")
        rot = [r for r in caplog.records if "ROTATED" in r.message]
        assert len(rot) == 1 and rot[0].levelname == "INFO"
        assert "ASIAOLD" in rot[0].getMessage() and "ASIANEW" in rot[0].getMessage()

    def test_two_providers_in_one_process_do_not_read_as_rotation(self, caplog) -> None:
        """A fill legitimately uses BOTH — the task role to read our buckets, an assumed role
        to write the partner's. A single last-seen slot would read those alternating calls as
        constant rotation and bury the real event.
        """
        self._reset()
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            for _ in range(3):
                self._serve("task-role", "ASIATASK")
                self._serve("assumed-role:writer", "ASIAWRITER")
        assert not [r for r in caplog.records if "ROTATED" in r.message]

    def test_the_same_role_name_in_two_accounts_is_not_one_provider(self, caplog, monkeypatch) -> None:
        """The identity is the full arn, because a role NAME is not unique.

        Two partner accounts may each hold a `Writer` role, and those are different credentials
        that rotate independently. Keyed on the name alone they share one last-seen slot, so
        every alternating call reads as a rotation — burying the real event in noise, which is
        the one thing this instrumentation exists to make findable.
        """
        self._reset()
        monkeypatch.setattr(
            creds_mod,
            "_assumed_role_credentials",
            lambda arn, ext: SimpleNamespace(
                get_frozen_credentials=lambda: SimpleNamespace(
                    # A distinct key per account, so a shared slot WOULD show as rotation.
                    access_key="ASIA" + arn.split("::")[1][:3],
                    secret_key="s",
                    token="t",
                )
            ),
        )
        monkeypatch.setattr(creds_mod, "_expires_after", lambda creds, ttl: datetime(2026, 8, 28, 12, tzinfo=UTC))
        first = creds_mod.AssumedRoleIcechunkCredentials("arn:aws:iam::111111111111:role/Writer")
        second = creds_mod.AssumedRoleIcechunkCredentials("arn:aws:iam::222222222222:role/Writer")

        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            for _ in range(3):
                first()
                second()

        assert not [r for r in caplog.records if "ROTATED" in r.message]
        assert len([r for r in caplog.records if "FIRST" in r.message]) == 2
        text = " ".join(r.getMessage() for r in caplog.records)
        # The account id is in the line, which is also what makes it correlatable against the
        # other account's CloudTrail.
        assert "111111111111" in text and "222222222222" in text

    def test_the_bookkeeping_is_locked_and_the_log_is_not(self, caplog, monkeypatch) -> None:
        """Pins BOTH halves of the concurrency design, deterministically.

        icechunk invokes this callback from its own Rust threads. An unguarded read-then-write
        lets two of them both find a key novel and both announce it — but a racing test can only
        fail probabilistically, so the invariant is asserted directly instead. The second half
        matters just as much: holding the lock across the log call would serialise every
        icechunk request in the process behind it, and per-request callback volume is the very
        thing being observed.
        """
        self._reset()
        lock = creds_mod._LAST_ACCESS_KEY_LOCK

        class _Watched(dict):
            def __init__(self) -> None:
                super().__init__()
                self.held: list[bool] = []

            def setdefault(self, key, default=None):  # type: ignore[override]
                self.held.append(lock.locked())
                return super().setdefault(key, default)

        watched = _Watched()
        monkeypatch.setattr(creds_mod, "_announced_access_keys", watched)
        logged_under_lock: list[bool] = []
        monkeypatch.setattr(creds_mod._LOG, "info", lambda *a, **k: logged_under_lock.append(lock.locked()))

        self._serve("task-role", "ASIALOCKED")

        assert watched.held == [True], "the novelty check must happen under the lock"
        assert logged_under_lock == [False], "the log must not run inside the lock"

    def test_concurrent_serves_announce_the_first_exactly_once(self, caplog) -> None:
        """The realistic companion: eight threads racing one new key produce one FIRST line."""
        self._reset()
        barrier = threading.Barrier(8)

        def serve() -> None:
            barrier.wait()
            self._serve("task-role", "ASIACONCURRENT")

        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            threads = [threading.Thread(target=serve) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len([r for r in caplog.records if "FIRST" in r.message]) == 1
        assert len(caplog.records) == 8

    def test_every_credential_event_carries_the_process_id(self, caplog) -> None:
        """The state this decides against is process-local; the log stream is not.

        An assembly's spawned workers and their parent share one container stream, and the
        formatter carries no process field, so without a pid a rotation seen in one worker
        reads as though it belonged to whichever process reported the adjacent failure. All
        three levels carry it, because the level that matters is whichever one is next to the
        failure.
        """
        self._reset()
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            self._serve("task-role", "ASIAONE")  # FIRST
            self._serve("task-role", "ASIATWO")  # ROTATED
            self._serve("task-role", "ASIATWO")  # steady, DEBUG
        assert [r.levelname for r in caplog.records] == ["INFO", "INFO", "DEBUG"]
        for record in caplog.records:
            assert f"pid={os.getpid()}" in record.getMessage(), record.getMessage()

    def test_a_late_callback_carrying_the_old_credential_is_not_a_rotation(self, caplog) -> None:
        """Acquisition happens before this call and outside the lock, so a thread can freeze the
        old credential, be descheduled past a refresh, and arrive after a faster thread recorded
        the new one. Announcing unconditionally would put the old key back and manufacture TWO
        false rotations — one backwards, one forwards again — in the one signal a real failure
        has to be lined up against.

        **The straggler here carries a LATER expiry than the key already recorded**, which is
        the case an order-based test gets wrong: the expiry is computed AFTER the credential is
        frozen, so it dates callback completion, not acquisition. Novelty is decidable from the
        state; order is not.
        """
        self._reset()
        base = datetime(2026, 8, 28, 12, tzinfo=UTC)
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            self._serve("task-role", "ASIAOLD", base)
            self._serve("task-role", "ASIANEW", base + timedelta(minutes=5))
            # The straggler — old key, and a LATER expiry than the new key was given.
            self._serve("task-role", "ASIAOLD", base + timedelta(minutes=10))
            # The state it must not have clobbered: a later new-key serve stays steady, which
            # asserts the straggler changed nothing rather than merely that it stayed quiet.
            self._serve("task-role", "ASIANEW", base + timedelta(minutes=11))

        assert len([r for r in caplog.records if "ROTATED" in r.message]) == 1
        assert [r.levelname for r in caplog.records] == ["INFO", "INFO", "DEBUG", "DEBUG"]
        assert "stale" in caplog.records[2].getMessage()
        assert "served" in caplog.records[3].getMessage()

    def test_every_event_carries_an_in_lock_sequence_number(self, caplog) -> None:
        """The records are emitted OUTSIDE the lock, so two threads can emit out of order and a
        ROTATED can reach the stream before the FIRST it followed. The sequence number is
        stamped while the lock is held, so a reader can always put the stream back in order.
        """
        self._reset()
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            self._serve("task-role", "ASIASEQA")
            self._serve("task-role", "ASIASEQB")
            self._serve("assumed-role:arn:aws:iam::1:role/W", "ASIASEQC")
        seqs = [int(re.search(r"seq=(\d+)", r.getMessage()).group(1)) for r in caplog.records]
        assert len(seqs) == 3
        # Process-wide and strictly increasing, so it orders ACROSS sources too — a fill uses
        # both providers and the interleaving between them is what a correlation reads.
        assert seqs == sorted(seqs) and len(set(seqs)) == 3

    def test_it_never_logs_the_secret_or_the_token(self, caplog) -> None:
        """The access-key id is an identifier and is what CloudTrail indexes on. The secret key
        and the session token are credentials and must never reach a log.
        """
        self._reset()
        with caplog.at_level(logging.DEBUG, logger=creds_mod.__name__):
            creds_mod._serve_icechunk_credential(
                source="task-role",
                access_key="ASIAVISIBLE",
                secret_key="SECRETVALUE",
                session_token="TOKENVALUE",
                expires_after=datetime(2026, 8, 28, 12, tzinfo=UTC),
            )
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "ASIAVISIBLE" in text
        assert "SECRETVALUE" not in text
        assert "TOKENVALUE" not in text

    def test_both_providers_are_instrumented(self) -> None:
        """The one that matters is the ASSUMED-ROLE callback: it is what serves the published
        store, whose writes motivated this. Instrumenting only the task-role provider would
        have left the credential behind the real failures invisible — which is exactly the
        defect review caught in the first version of this change.
        """
        import inspect

        task_src = inspect.getsource(creds_mod.iam_icechunk_credentials)
        role_src = inspect.getsource(creds_mod.AssumedRoleIcechunkCredentials.__call__)
        for name, src in (("task-role", task_src), ("assumed-role", role_src)):
            assert "_serve_icechunk_credential(" in src, f"{name} bypasses the boundary"
            assert "icechunk.S3StaticCredentials(" not in src, (
                f"{name} constructs its own credential, so it can drift from the boundary"
            )
