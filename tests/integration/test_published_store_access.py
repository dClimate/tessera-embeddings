"""The published store is readable by an outside consumer — checked against the live store.

**Why this test reaches a live service when the tier says cassettes always.** The rule exists so
the suite stays deterministic and fast, and it is the right rule for a third-party API: what we
care about there is that our parsing still works, and a recording proves that. Here the subject
under test IS the live artifact. The claim is "somebody with no AWS account can open the published
store and get data out of it", and a cassette of our own recording cannot fail in the way that
claim can — the bucket policy could be tightened, the branch could be reset, a future
reorganisation could move the zone groups, and every one of those would leave a replayed cassette
green.

So it is opt-in TWICE: the ``integration`` marker the default invocation deselects, and an
environment variable on top, because a marker alone has previously been enough for a test to end
up running where it was not wanted. Nothing in CI sets it::

    TESSERA_TEST_PUBLISHED_STORE=1 uv run pytest -m integration tests/integration/test_published_store_access.py

It reads no credentials, on purpose. If it passes with an AWS profile in the environment but fails
without one, that is the finding, not a flaw in the test — so it clears every AWS variable first.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from tessera_embeddings.storage import published_store
from tessera_embeddings.storage.global_store import open_global_repo

PUBLISHED_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
PUBLISHED_REGION = "us-west-2"
#: Nine annual timesteps, preallocated at seed (ADR 008 D1).
EXPECTED_YEARS = tuple(range(2017, 2026))
#: 60 six-degree UTM zones, north and south.
EXPECTED_GROUPS = 120
#: A small zone (a couple of dozen live shards) so a read of it stays quick.
SAMPLE_ZONE = "16S"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TESSERA_TEST_PUBLISHED_STORE"),
        reason="set TESSERA_TEST_PUBLISHED_STORE=1 to read the live published store",
    ),
]

_AWS_ENV = (
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_PROFILE",
)


@pytest.fixture(scope="module")
def anonymous_root(tmp_path_factory):
    """The published store's root group, opened with every AWS variable cleared.

    Module-scoped because opening it is the expensive part and every test here wants the same
    snapshot; a per-test open would also make the suite's runtime a function of its test count.
    """
    saved = {name: os.environ.pop(name, None) for name in _AWS_ENV}
    # Point the shared-credentials and config files at an empty directory as well, so a profile in
    # ~/.aws cannot answer for the anonymous path and make this test pass for the wrong reason.
    empty = tmp_path_factory.mktemp("no-aws")
    saved["AWS_SHARED_CREDENTIALS_FILE"] = os.environ.get("AWS_SHARED_CREDENTIALS_FILE")
    saved["AWS_CONFIG_FILE"] = os.environ.get("AWS_CONFIG_FILE")
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(empty / "credentials")
    os.environ["AWS_CONFIG_FILE"] = str(empty / "config")
    try:
        # Prove the scrubbing worked before trusting anything this fixture yields. Without this,
        # an ambient profile would make every test below pass while saying nothing about the
        # anonymous path — the exact failure the environment clearing exists to prevent.
        import botocore.session

        assert botocore.session.get_session().get_credentials() is None, (
            "credentials are still resolvable, so this fixture is not testing anonymous access"
        )
        repo = open_global_repo(PUBLISHED_URI, region=PUBLISHED_REGION, anonymous=True)
        session = repo.readonly_session(branch="main")
        yield repo, session, zarr.open_group(session.store, mode="r")
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestAnonymousAccess:
    """What an outside consumer can do with no account."""

    def test_the_store_opens_and_holds_every_utm_zone(self, anonymous_root):
        _, _, root = anonymous_root
        groups = sorted(name for name, _ in root.groups())
        assert len(groups) == EXPECTED_GROUPS
        assert groups[0] == "01N"
        assert groups[-1] == "60S"

    def test_the_reader_inherits_the_writers_manifest_tuning(self, anonymous_root):
        # Saved into the store with `save_config`, so a consumer who passes no configuration of
        # their own still reads with the splitting and preload the writer chose. If this ever
        # returns None, readers are silently on icechunk's defaults.
        repo, _, _ = anonymous_root
        manifest = repo.config.manifest
        assert manifest is not None
        assert manifest.splitting is not None
        assert manifest.preload is not None
        assert manifest.preload.max_total_refs > 0

    def test_a_zone_group_conforms_to_the_declared_layout(self, anonymous_root):
        _, _, root = anonymous_root
        assert published_store.layout_departures(root[SAMPLE_ZONE]) == []

    def test_a_zone_declares_its_crs_and_complete_years(self, anonymous_root):
        _, _, root = anonymous_root
        attrs = dict(root[SAMPLE_ZONE].attrs)
        assert attrs["crs"] == "EPSG:32716"
        assert tuple(attrs["years_complete"]) == EXPECTED_YEARS

    def test_the_time_axis_is_the_nine_preallocated_calendar_years(self, anonymous_root):
        _, _, root = anonymous_root
        stamps = np.asarray(root[SAMPLE_ZONE]["time"][:]).astype("datetime64[ns]")
        years = tuple(int(y) for y in stamps.astype("datetime64[Y]").astype(int) + 1970)
        assert years == EXPECTED_YEARS

    def test_reading_a_live_pixel_returns_embeddings_rather_than_fill(self, anonymous_root):
        _, session, root = anonymous_root
        group = root[SAMPLE_ZONE]
        coverage = published_store.live_shards(session, SAMPLE_ZONE)
        last_year_index = len(EXPECTED_YEARS) - 1
        assert coverage[last_year_index], f"{SAMPLE_ZONE} has no live shards in {EXPECTED_YEARS[-1]}"
        pixels = published_store.sample_live_pixels(group, last_year_index, coverage[last_year_index], 1, seed=0)
        assert pixels, "no sampled pixel held data"
        y, x = pixels[0]
        vector = np.asarray(group["embeddings"][last_year_index, y, x, :])
        assert vector.shape == (128,)
        assert vector.dtype == np.dtype("int8")
        # int8 fill is 0 and a real embedding is not all-zero, which is exactly why `scales` and
        # its NaN fill is what `sample_live_pixels` filters on rather than this array.
        assert np.any(vector != 0)

    def test_every_zone_year_marked_complete_is_also_tagged(self, anonymous_root):
        # The two are written in separate commits, so they can disagree. Checked over the whole
        # store because a single zone would not exercise the reconciliation at all.
        repo, _, root = anonymous_root
        tagged = {
            (parts[1], int(parts[2]))
            for parts in (tag.split("-") for tag in repo.list_tags())
            if len(parts) == 3 and parts[0] == "zone" and parts[2].isdigit()
        }
        marked = {(zone, int(year)) for zone, group in root.groups() for year in group.attrs.get("years_complete", [])}
        assert marked == tagged
        assert len(marked) == EXPECTED_GROUPS * len(EXPECTED_YEARS) - 14, (
            "the campaign left 14 zone-years unfilled; a different count means coverage moved"
        )
