"""Overriding where the global store lives, for publishing outside our own buckets.

The campaign's published store is normally derived from the outputs bucket
(``{outputs}/global/tessera.icechunk``). Publishing to an AWS Open Data bucket needs a location the
derivation cannot produce — a different bucket, no ``global/`` segment, a ``.zarr`` suffix on what is
still an Icechunk repo — so ``BucketPaths`` carries an optional whole URI that wins.

**Why one field rather than a parameter threaded through nineteen call sites.** Every producer and
consumer of the campaign store asks ``paths.global_store(...)``: the driver, both fills, the
assembler, the seeder, the validator, the closing sweep, the monitoring checks and the repair tools.
One field is therefore inherited by all of them at once, and — the part that matters — there is no
arrangement in which one tool writes to the override while another reads the derived path.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.storage.registry import part_uri

#: The real target, so the test says what this exists for.
OPEN_DATA = "s3://tessera-embeddings/v1.1/dclimate.icechunk"


def _paths(**kw) -> BucketPaths:
    return BucketPaths(inputs="s3://in", outputs="s3://out", **kw)


def test_without_an_override_the_path_is_derived_exactly_as_before() -> None:
    """The default has to be untouched: every dev run, and every account that is not publishing,
    depends on it — and a change here would silently redirect a campaign.
    """
    assert _paths().global_store() == "s3://out/global/tessera.icechunk"
    assert _paths().global_store("other") == "s3://out/global/other.icechunk"


def test_an_override_is_returned_verbatim() -> None:
    """Verbatim, with nothing joined onto it. The point of the override is that the shape is not
    ours to decide: no `global/` segment, and a `.zarr` suffix on an Icechunk repo because that is
    the convention of the bucket we are publishing into.
    """
    assert _paths(global_store_uri=OPEN_DATA).global_store() == OPEN_DATA


def test_an_override_survives_serialisation() -> None:
    """`BucketPaths` reaches a flow as deployment parameters — `model_dump()` on one side and a
    rebuild on the other. An override that did not round-trip would apply on the laptop that
    dispatched the run and nowhere else.
    """
    rebuilt = BucketPaths(**_paths(global_store_uri=OPEN_DATA).model_dump())
    assert rebuilt.global_store() == OPEN_DATA


def test_asking_for_a_store_name_alongside_an_override_raises() -> None:
    """The guard. With an override the name has nowhere to go, and silently ignoring it is how a
    caller ends up certain it addressed a store that does not exist — the failure would surface as
    an empty or missing repo somewhere far from the call.
    """
    with pytest.raises(ValueError, match="the override is the whole URI"):
        _paths(global_store_uri=OPEN_DATA).global_store("something-else")


def test_the_default_name_is_accepted_alongside_an_override() -> None:
    """Because four call sites pass the name explicitly — the seeder, both fills and the driver —
    all of them defaulted. If the guard refused any name at all, the override would be unreachable
    by every real caller.
    """
    assert _paths(global_store_uri=OPEN_DATA).global_store("tessera") == OPEN_DATA


def test_an_override_leaves_every_other_path_alone() -> None:
    """It renames one thing. Mosaics, ROI masks and the land mask stay in our own buckets, and a
    change that moved them would move the campaign's INPUTS into a public bucket.
    """
    p = _paths(global_store_uri=OPEN_DATA)
    assert p.land_mask_store() == "s3://in/masks/global.icechunk"
    assert p.zone_roi_store("33N") == "s3://in/rois/zarrs/zone_33N.zarr"
    assert p.store_for("33N", "reflectance") == "s3://in/mosaics/33N/reflectance.zarr"
    assert p.store_for("33N", "embeddings") == "s3://out/embeddings/33N.zarr"


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_override_derives_rather_than_returning_nothing(empty) -> None:
    """An empty string is what an unset environment variable or a blank deployment parameter looks
    like. Treating it as an override would ask Icechunk to open ``""``.
    """
    assert _paths(global_store_uri=empty).global_store() == "s3://out/global/tessera.icechunk"


class TestTheRegistrySitsBesideWhicheverStoreIsInUse:
    """The registry is derived from the store's own location, never from ``outputs``.

    Production publishes to a bucket that is not ours, so a registry built from ``outputs`` would
    sit in our bucket while the store sat in the public one — and every tool would still work, which
    is how a mask written where nothing reads it looks like success.
    """

    def test_a_derived_store_gets_a_derived_registry_beside_it(self):
        paths = BucketPaths(inputs="s3://in", outputs="s3://out")
        assert paths.global_store() == "s3://out/global/tessera.icechunk"
        assert paths.optical_registry() == "s3://out/global/tessera.registry"

    def test_an_overridden_store_takes_its_registry_with_it(self):
        """The two prefixes named in the Open Data access request, and the reason they are two:
        Icechunk owns every key under its own prefix, so the registry cannot live inside it.
        """
        paths = BucketPaths(
            inputs="s3://in",
            outputs="s3://out",
            global_store_uri="s3://tessera-embeddings/v1.1/dclimate.icechunk",
        )
        assert paths.optical_registry() == "s3://tessera-embeddings/v1.1/dclimate.registry"

    def test_a_relative_store_uri_keeps_its_registry_beside_it(self):
        """A bare relative path is what local runs and tests use, and it has no separator.

        `rpartition("/")` then leaves `base` empty, so joining with "/" named
        `/tessera.registry` at the FILESYSTEM ROOT rather than a sibling — failing on
        permissions if you are lucky and writing somewhere unrelated to the configured store
        if you are not.
        """
        paths = BucketPaths(inputs="s3://in", outputs="s3://out", global_store_uri="tessera.icechunk")
        assert paths.optical_registry() == "tessera.registry"
        assert not paths.optical_registry().startswith("/")

    def test_the_registry_is_never_inside_the_store(self):
        """Icechunk's garbage collection enumerates its own prefix and reconciles it against its
        manifests, so a parquet in there is at best unrecognised and at worst collected.
        """
        for paths in (
            BucketPaths(inputs="s3://in", outputs="s3://out"),
            BucketPaths(inputs="s3://in", outputs="s3://out", global_store_uri="s3://pub/v1/x.icechunk"),
        ):
            assert not paths.optical_registry().startswith(paths.global_store() + "/")

    def test_registry_parts_follow_the_store_not_our_outputs(self):
        """The mistake this class exists to prevent, made once and caught here.

        The per-tile refusal records were first placed under ``outputs``, which works in dev — where
        the store is derived from ``outputs`` too — and quietly splits in production, where the store
        is published to a bucket that is not ours. Every tool still works; the records simply sit
        somewhere no consumer of the store will look.
        """
        prod = BucketPaths(
            inputs="s3://in",
            outputs="s3://ours",
            global_store_uri="s3://tessera-embeddings/v1.1/dclimate.icechunk",
        )
        detail = part_uri(prod.optical_registry(), "32S", 2021, "run-1")

        assert detail.startswith(prod.optical_registry() + "/"), "inside the registry beside the store"
        assert "s3://ours" not in detail, "and never in our own bucket when the store is published"
        assert not detail.startswith(prod.global_store() + "/"), "still never inside the Icechunk prefix"

    def test_registry_parts_are_partitioned_for_a_parquet_dataset(self):
        """The registry is specified as Parquet indexing what each shard holds. The parts are JSON
        for now — pyarrow is not in the fill's runtime — so the LAYOUT has to be the one a later
        compaction wants, or every path changes when the format does.
        """
        paths = BucketPaths(inputs="s3://in", outputs="s3://out")
        detail = part_uri(paths.optical_registry(), "09S", 2024, "run-1")
        assert "zone=09S" in detail and "year=2024" in detail
