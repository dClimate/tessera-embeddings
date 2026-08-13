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

#: The real target, so the test says what this exists for.
OPEN_DATA = "s3://tessera-embeddings/v1.1/dclimate.zarr"


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
