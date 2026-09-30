"""The published store is readable by an outside consumer — checked against the live store.

Reaches a live service where the tier says cassettes always, because here the subject under test IS
the live artifact: the claim is "somebody with no AWS account can open the published store and get
data out of it", and a replayed recording of our own request stays green through every way that
claim can break. See `tests/integration/README.md` for the full reasoning.

Opt-in twice — the ``integration`` marker plus ``TESSERA_TEST_PUBLISHED_STORE=1``, which nothing in
CI sets::

    TESSERA_TEST_PUBLISHED_STORE=1 uv run pytest -m integration tests/integration/test_published_store_access.py

It reads no credentials on purpose, and clears every AWS variable first: passing with a profile in
the environment but failing without one would be the finding, not a flaw in the test.
"""

from __future__ import annotations

import os

import icechunk
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

#: Every way botocore can find a credential, not just the ones a laptop uses. On an EC2 or ECS host
#: — where this is most likely to be run — instance metadata and the container credential endpoint
#: answer even with every key and profile variable cleared, and a web-identity pair answers in CI,
#: so clearing only the profile and static-key variables would turn a correct environment red.
_AWS_ENV = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)


@pytest.fixture(scope="module")
def anonymous_root(tmp_path_factory):
    """The published store's root group, opened with every AWS variable cleared.

    Module-scoped because opening it is the expensive part and every test here wants the same
    snapshot; a per-test open would also make the suite's runtime a function of its test count.
    """
    saved = {name: os.environ.pop(name, None) for name in _AWS_ENV}
    # The shared-credentials and config files too, so a profile in ~/.aws cannot answer for the
    # anonymous path and make this pass for the wrong reason.
    empty = tmp_path_factory.mktemp("no-aws")
    for name in ("AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE", "AWS_EC2_METADATA_DISABLED"):
        saved[name] = os.environ.get(name)
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(empty / "credentials")
    os.environ["AWS_CONFIG_FILE"] = str(empty / "config")
    # Instance metadata is reachable on any EC2 host and is not an environment variable to unset.
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    try:
        # Prove the scrubbing worked before trusting anything this fixture yields: an ambient
        # profile would make every test below pass while saying nothing about the anonymous path.
        import botocore.session

        assert botocore.session.get_session().get_credentials() is None, (
            "credentials are still resolvable, so this fixture is not testing anonymous access"
        )
        repo = open_global_repo(PUBLISHED_URI, region=PUBLISHED_REGION, anonymous=True)
        session = repo.readonly_session(branch="main")
        # The config SAVED IN THE STORE, fetched separately, so the tests below can compare it
        # against the handle rather than reading the handle and calling that the store's state.
        saved_config = icechunk.Repository.fetch_config(
            icechunk.s3_storage(
                bucket=PUBLISHED_URI.removeprefix("s3://").split("/", 1)[0],
                prefix=PUBLISHED_URI.removeprefix("s3://").split("/", 1)[1],
                region=PUBLISHED_REGION,
                anonymous=True,
            )
        )
        yield repo, session, zarr.open_group(session.store, mode="r"), saved_config
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestAnonymousAccess:
    """What an outside consumer can do with no account."""

    def test_the_store_opens_and_holds_every_utm_zone(self, anonymous_root):
        _, _, root, _ = anonymous_root
        groups = sorted(name for name, _ in root.groups())
        assert len(groups) == EXPECTED_GROUPS
        assert groups[0] == "01N"
        assert groups[-1] == "60S"

    def test_the_store_is_tuned_for_readers_rather_than_for_the_fill(self, anonymous_root):
        # The campaign's last step switches the saved manifest preload off — BOTH counters, which
        # is what `set_saved_manifest_preload` writes. A re-enabled preload costs every consumer
        # about 2.5 s per open and buys them nothing, and the figures in
        # context_docs/storage/reading-the-published-store.md would stop applying. Splitting must
        # survive that switch, because it describes the manifests already on disk.
        _, _, _, saved_config = anonymous_root
        assert saved_config is not None, "the store has no saved repository config"
        manifest = saved_config.manifest
        assert manifest.splitting is not None, "the manifest splitting configuration was dropped"
        assert (manifest.preload.max_total_refs, manifest.preload.max_arrays_to_scan) == (0, 0)

    def test_opening_inherits_the_saved_config_rather_than_replacing_it(self, anonymous_root):
        # A config handed to `Repository.open` REPLACES the saved one, which is how readers used to
        # get the writer's preload back through this package — 2,245 ms against 852 inheriting.
        repo, _, _, saved_config = anonymous_root
        assert repr(repo.config) == repr(saved_config)

    def test_a_zone_group_conforms_to_the_declared_layout(self, anonymous_root):
        _, _, root, _ = anonymous_root
        assert published_store.layout_departures(root[SAMPLE_ZONE]) == []

    def test_a_zone_declares_its_crs_and_complete_years(self, anonymous_root):
        _, _, root, _ = anonymous_root
        attrs = dict(root[SAMPLE_ZONE].attrs)
        assert attrs["crs"] == "EPSG:32716"
        assert tuple(attrs["years_complete"]) == EXPECTED_YEARS

    def test_the_time_axis_is_the_nine_preallocated_calendar_years(self, anonymous_root):
        _, _, root, _ = anonymous_root
        assert tuple(published_store.calendar_years(root[SAMPLE_ZONE])) == EXPECTED_YEARS

    def test_reading_a_live_pixel_returns_embeddings_rather_than_fill(self, anonymous_root):
        _, session, root, _ = anonymous_root
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
        # int8 fill is 0 and a real embedding is not all-zero, which is why `sample_live_pixels`
        # filters on `scales` and its NaN fill rather than on this array.
        assert np.any(vector != 0)

    def test_every_zone_year_marked_complete_is_also_tagged(self, anonymous_root):
        # The two are written in separate commits, so they can disagree. Over the whole store,
        # because a single zone would not exercise the reconciliation at all.
        repo, _, root, _ = anonymous_root
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
