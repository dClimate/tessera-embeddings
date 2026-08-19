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

from datetime import UTC, datetime, timedelta

import pytest

from tessera_embeddings.providers.aws import credentials as cred

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
