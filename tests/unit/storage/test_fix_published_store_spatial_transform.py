"""The published-store repair script, exercised against a local store in the store's broken shape.

The script edits the one artifact nobody can rebuild, so what matters is not only that it writes
the right transform but that it writes NOTHING else and refuses anything it does not recognise.
Every test here seeds a real Icechunk store, regresses it to the exact state the published store
shipped in, and runs the script's own ``main`` — the entry point an operator invokes — rather than
its internals.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import zarr

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage import global_store
from tessera_embeddings.storage.zone_grid import PIXEL_M, ZoneSpec

_SCRIPT = Path(__file__).parents[3] / "scripts" / "maintenance" / "fix_published_store_spatial_transform.py"

#: Two zones, so a partial run and a per-group refusal have something to be partial about.
_ZONES = (
    ZoneSpec("32601", "N", 1, (0.0, 2 * SHARD_PX * PIXEL_M), (0.0, 2 * SHARD_PX * PIXEL_M)),
    ZoneSpec("32701", "S", 1, (0.0, 2 * SHARD_PX * PIXEL_M), (0.0, 2 * SHARD_PX * PIXEL_M)),
)
_YEARS = (2024,)

#: The dead registration URLs the published store shipped with — a tag neither convention cut.
_STALE_SPATIAL_URL = "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json"


def _load_script() -> Any:
    """Import the maintenance script by path — it is not part of the installed package."""
    spec = importlib.util.spec_from_file_location("fix_published_store_spatial_transform", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _regress_to_published_state(path: str, *, zones: tuple[str, ...], urls: bool = True) -> None:
    """Put a freshly seeded store back into the shape the published store shipped in.

    The fixed source seeds the CORRECT corner origin, so the bug has to be reintroduced to be
    repaired. Done by moving the origin half a pixel forward along each axis — the inverse of the
    fix — rather than by writing literals, so the regression stays valid at any grid.
    """
    session = global_store.open_global_repo(path).writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    for zone in zones:
        attrs = dict(root[zone].attrs)
        a, b, c, d, e, f = (float(v) for v in attrs["spatial:transform"])
        attrs["spatial:transform"] = [a, b, c + a / 2, d, e, f + e / 2]
        if urls:
            attrs["zarr_conventions"] = [
                {**entry, "schema_url": _STALE_SPATIAL_URL} if entry.get("name") == "spatial:" else entry
                for entry in attrs["zarr_conventions"]
            ]
        root[zone].attrs.update(attrs)
    session.commit("regress to the published store's shipped state")


def _attrs_of(path: str, zone: str) -> dict:
    root = zarr.open_group(global_store.open_global_repo(path).readonly_session(branch="main").store, mode="r")
    return dict(root[zone].attrs)


@pytest.fixture
def seeded(tmp_path: Path) -> str:
    """A two-zone global store with correct, freshly written convention attrs."""
    path = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(path)
    global_store.seed_zone_groups(repo, list(_ZONES), years=_YEARS)
    return path


@pytest.fixture
def broken(seeded: str) -> str:
    """The same store, regressed to the centre-origin, dead-URL state that was published."""
    _regress_to_published_state(seeded, zones=("01N", "01S"))
    return seeded


def _run(path: str, *args: str) -> int:
    """Invoke the script the way an operator does, with the published store's guards disabled."""
    return script.main(["--uri", path, "--expect-tags", "0", "--expect-groups", "0", *args])


class TestTheSeedIsAlreadyCorrect:
    """A store written by the fixed source needs no repair — the script must say so and stop."""

    def test_a_fresh_store_is_a_clean_no_op(self, seeded: str, capsys: pytest.CaptureFixture) -> None:
        assert _run(seeded, "--apply") == 0
        assert "nothing to do" in capsys.readouterr().out

    def test_a_fresh_store_gains_no_commit(self, seeded: str) -> None:
        """Not merely "no change": no snapshot at all, so the history stays readable."""
        before = global_store.open_global_repo(seeded).lookup_branch("main")
        _run(seeded, "--apply")
        assert global_store.open_global_repo(seeded).lookup_branch("main") == before


class TestTheRepair:
    """What a real run does to a store in the published state."""

    def test_dry_run_changes_nothing(self, broken: str) -> None:
        before = _attrs_of(broken, "01N")
        assert _run(broken) == 0
        assert _attrs_of(broken, "01N") == before

    def test_the_origin_moves_to_the_bbox_corner(self, broken: str) -> None:
        assert _run(broken, "--apply") == 0
        for zone in ("01N", "01S"):
            attrs = _attrs_of(broken, zone)
            transform, bbox = attrs["spatial:transform"], attrs["spatial:bbox"]
            assert transform[2] == pytest.approx(bbox[0], abs=1e-6, rel=0.0)
            assert transform[5] == pytest.approx(bbox[3], abs=1e-6, rel=0.0)

    def test_the_registration_urls_are_repinned(self, broken: str) -> None:
        assert _run(broken, "--apply") == 0
        registered = {c["name"]: c for c in _attrs_of(broken, "01N")["zarr_conventions"]}
        assert registered["spatial:"]["schema_url"] != _STALE_SPATIAL_URL
        assert "/v0.1/" in registered["spatial:"]["schema_url"]
        assert "/v0.1/" in registered["proj:"]["spec_url"]

    def test_nothing_but_the_two_attrs_changes(self, broken: str) -> None:
        """The guarantee the store depends on: provenance, run records and the depth rule survive.

        Compared key by key over the WHOLE attrs dict, not over a list of things worth checking —
        a list would omit whatever the next change adds.
        """
        before = _attrs_of(broken, "01N")
        assert _run(broken, "--apply") == 0
        after = _attrs_of(broken, "01N")
        assert set(before) == set(after)
        changed = {k for k in before if before[k] != after[k]}
        assert changed == {"spatial:transform", "zarr_conventions"}

    def test_the_repair_restores_what_the_seed_would_have_written(self, seeded: str) -> None:
        """The repaired value equals what the fixed source writes from scratch, to the float.

        The strongest available statement that the script and the source agree: it compares the
        script's output against the seeder's rather than against the script's own arithmetic.
        """
        pristine = {zone: _attrs_of(seeded, zone) for zone in ("01N", "01S")}
        _regress_to_published_state(seeded, zones=("01N", "01S"))
        assert _run(seeded, "--apply") == 0
        for zone, expected in pristine.items():
            assert _attrs_of(seeded, zone) == expected

    def test_rerunning_is_a_no_op(self, broken: str, capsys: pytest.CaptureFixture) -> None:
        assert _run(broken, "--apply") == 0
        capsys.readouterr()
        assert _run(broken, "--apply") == 0
        assert "nothing to do" in capsys.readouterr().out

    def test_one_zone_may_be_rehearsed_alone(self, broken: str) -> None:
        untouched = _attrs_of(broken, "01S")
        assert _run(broken, "--apply", "--zone", "01N") == 0
        assert _attrs_of(broken, "01N")["spatial:transform"][2] == pytest.approx(
            _attrs_of(broken, "01N")["spatial:bbox"][0], abs=1e-6, rel=0.0
        )
        assert _attrs_of(broken, "01S") == untouched

    def test_the_record_file_captures_before_and_after(self, broken: str, tmp_path: Path) -> None:
        record = tmp_path / "record.json"
        assert _run(broken, "--record", str(record)) == 0
        payload = json.loads(record.read_text())
        assert {g["zone"] for g in payload["groups"]} == {"01N", "01S"}
        entry = next(g for g in payload["groups"] if g["zone"] == "01N")
        assert entry["shift"] == [-PIXEL_M / 2, PIXEL_M / 2]
        assert entry["before"] != entry["after"]


class TestRefusals:
    """States the script must decline rather than guess at. Each leaves the store untouched."""

    @staticmethod
    def _corrupt(path: str, zone: str, **attrs: Any) -> None:
        session = global_store.open_global_repo(path).writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        root[zone].attrs.update({**dict(root[zone].attrs), **attrs})
        session.commit(f"corrupt {zone}")

    def test_an_unknown_origin_is_not_overwritten(self, broken: str) -> None:
        """Neither the centre nor the corner means somebody else wrote it; leave it alone."""
        transform = list(_attrs_of(broken, "01N")["spatial:transform"])
        transform[2] += 137.0
        self._corrupt(broken, "01N", **{"spatial:transform": transform})
        before = _attrs_of(broken, "01S")
        assert _run(broken, "--apply") == 1
        assert _attrs_of(broken, "01S") == before, "a refusal on one group must not half-repair another"

    def test_node_registration_is_refused(self, broken: str) -> None:
        """Under node registration the coordinates ARE the edges, so no half-pixel is owed."""
        self._corrupt(broken, "01N", **{"spatial:registration": "node"})
        assert _run(broken, "--apply") == 1

    def test_a_rotated_transform_is_refused(self, broken: str) -> None:
        transform = list(_attrs_of(broken, "01N")["spatial:transform"])
        transform[1] = 0.5
        self._corrupt(broken, "01N", **{"spatial:transform": transform})
        assert _run(broken, "--apply") == 1

    def test_a_bbox_that_does_not_corroborate_is_refused(self, broken: str) -> None:
        """The cross-check is load-bearing: without agreement there is no second opinion."""
        bbox = list(_attrs_of(broken, "01N")["spatial:bbox"])
        bbox[0] += 25.0
        self._corrupt(broken, "01N", **{"spatial:bbox": bbox})
        assert _run(broken, "--apply") == 1

    def test_a_shape_disagreeing_with_the_arrays_is_refused(self, broken: str) -> None:
        self._corrupt(broken, "01N", **{"spatial:shape": [7, 7]})
        assert _run(broken, "--apply") == 1

    def test_an_unexpected_group_count_is_refused(self, broken: str) -> None:
        """The published store has 120 groups; a different number is a store nobody tested this on."""
        assert script.main(["--uri", broken, "--expect-tags", "0", "--expect-groups", "120", "--apply"]) == 1

    def test_an_unknown_zone_is_refused(self, broken: str) -> None:
        assert _run(broken, "--apply", "--zone", "99N") == 1
