"""Optical-source preflight: refusal must be a positive finding; everything else passes through.

The preflight's failure modes are asymmetric: a wasted pass-through costs one late
coverage-gate failure (today's behaviour), while a wrong refusal silently loses a
buildable cell from the campaign. These tests pin the boundary that matters most —
"confirmed absent" versus "could not determine" — and the two constructions the verdict
rests on: the solar-day padding of the probe window, and the blocks jointly covering
every live tile.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.ingest.land_mask import _live_tile_bbox_wgs84, live_tile_block_bboxes_wgs84
from tessera_embeddings.ingest.source_coverage import BLOCK_TILES, SourceFinding, preflight_optical_source
from tessera_embeddings.storage import zone_grid
from tessera_embeddings.storage.zone_grid import PIXEL_M

from .coverage_repo import make_coverage

WINDOW = parse_time_window("December 2021")

#: A zone with land in it and a small tile grid footprint in the fixture repo. The
#: preflight's logic is zone-agnostic; the tests only need SOME real ZoneSpec.
ZONE = "31S"

#: Two live tiles in one block plus one far away, so the sweep has TWO blocks to
#: answer, of unequal weight (2 vs 1) — which also pins the densest-first order.
LIVE_TILES = [(10, 5), (10, 6), (200, 20)]


@pytest.fixture()
def coverage(tmp_path):
    return make_coverage(tmp_path, ZONE, LIVE_TILES)


class ScriptedProbe:
    """A probe answering from a script, recording every call it receives.

    The script is consumed positionally, in the preflight's documented probe order:
    blocks densest-first. An entry of ``True``/``False`` answers; an exception
    instance raises.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[tuple[tuple[float, float, float, float], str, str]] = []

    def __call__(self, bbox, start, end):
        self.calls.append((bbox, start, end))
        answer = self.script.pop(0) if self.script else False
        if isinstance(answer, Exception):
            raise answer
        return answer


def _preflight(coverage, probe, **kwargs):
    return preflight_optical_source(ZONE, WINDOW, land_mask_path=coverage, probe=probe, **kwargs)


def _fixture_blocks():
    """The block boxes the fixture bitmap yields, densest first — the probe order."""
    spec = zone_grid.zone(ZONE)
    tl = _bitmap(spec, LIVE_TILES)
    return sorted(live_tile_block_bboxes_wgs84(spec, tl, block_tiles=BLOCK_TILES), key=lambda b: -b[1])


def test_every_block_missing_is_confirmed_absent(coverage):
    """The motivating shape: nothing published reaches live land, and every block was
    actually asked — absence is earned, not defaulted.
    """
    probe = ScriptedProbe([False, False])
    result = _preflight(coverage, probe)
    assert result.finding is SourceFinding.CONFIRMED_ABSENT
    assert result.probes == 2
    assert len(probe.calls) == 2


def test_first_block_hit_is_provisional_present_in_one_probe(coverage):
    probe = ScriptedProbe([True])
    result = _preflight(coverage, probe)
    assert result.finding is SourceFinding.PRESENT
    assert result.probes == 1


def test_blocks_are_probed_densest_first(coverage):
    """The common case must settle at the first probe, so the first box asked has to
    be the one carrying the most live land.
    """
    probe = ScriptedProbe([False, False])
    _preflight(coverage, probe)
    expected = _fixture_blocks()
    assert probe.calls[0][0] == expected[0][0]
    assert expected[0][1] > expected[1][1]  # the order is real, not alphabetical luck


def test_a_later_block_hit_still_passes(coverage):
    probe = ScriptedProbe([False, True])
    result = _preflight(coverage, probe)
    assert result.finding is SourceFinding.PRESENT
    assert result.probes == 2


def test_a_failed_block_probe_downgrades_absence_to_inconclusive(coverage):
    """The boundary this module exists for: an unanswered block is NOT a miss.

    All answered blocks miss, one block probe fails — absence over live land is
    plausible but was not positively established, so the cell must pass through.
    """
    probe = ScriptedProbe([RuntimeError("catalogue outage"), False])
    result = _preflight(coverage, probe)
    assert result.finding is SourceFinding.INCONCLUSIVE
    assert "failed" in result.reason


def test_every_probe_failing_is_inconclusive_and_never_raises(coverage):
    probe = ScriptedProbe([RuntimeError("down"), RuntimeError("down")])
    result = _preflight(coverage, probe)
    assert result.finding is SourceFinding.INCONCLUSIVE


def test_probe_window_is_the_date_range_padded_one_utc_day(coverage):
    """Mosaic dates are solar days; a solar day draws on UTC dates one day either side.

    An unpadded probe could refuse a cell whose only imagery is keyed to the UTC day
    adjacent to the window — imagery the fill would legitimately use.
    """
    probe = ScriptedProbe([True])
    _preflight(coverage, probe)
    _bbox, start, end = probe.calls[0]
    assert (start, end) == ("2020-12-31", "2022-01-01")


def test_exhausted_budget_passes_through(coverage):
    probe = ScriptedProbe([True])  # a block WOULD hit, but the budget is spent
    result = _preflight(coverage, probe, budget_s=-1.0)
    assert result.finding is SourceFinding.INCONCLUSIVE
    assert "budget" in result.reason
    assert probe.calls == []  # no probe was started against a spent budget


def test_a_stalled_probe_cannot_outrun_the_budget(coverage):
    """The budget must BOUND a probe, not merely precede one.

    Checking the deadline between probes bounds nothing: the default probe carries connect and read
    timeouts plus a three-attempt retry adapter, so a stalled catalogue makes a single call run for
    minutes and the advertised budget becomes a lower bound on the delay rather than an upper one.
    Every campaign cell would pay that, to learn something this preflight will answer INCONCLUSIVE
    for anyway.

    The probe here blocks far longer than the budget. What is asserted is that the CALL returns
    quickly — a timed-out future cannot stop the thread, and this pins the part that is actually
    promised.
    """
    import threading
    import time as _time

    started = threading.Event()

    def stalling_probe(_bbox, _start, _end):
        started.set()
        _time.sleep(30.0)  # far past the budget below
        return True  # and it WOULD have hit, so a pass-through cannot be luck

    t0 = _time.monotonic()
    result = _preflight(coverage, stalling_probe, budget_s=0.3)
    elapsed = _time.monotonic() - t0

    assert started.is_set(), "the probe really was entered"
    assert result.finding is SourceFinding.INCONCLUSIVE
    assert "budget" in result.reason
    assert elapsed < 5.0, f"returned in {elapsed:.1f}s — the budget must bound the call"
    assert result.probes == 1


def test_unreadable_coverage_repo_is_inconclusive(tmp_path):
    probe = ScriptedProbe([False])
    result = preflight_optical_source(
        ZONE, WINDOW, land_mask_path=str(tmp_path / "does-not-exist.icechunk"), probe=probe
    )
    assert result.finding is SourceFinding.INCONCLUSIVE
    assert result.probes == 0


def test_all_ocean_zone_is_inconclusive_with_zero_probes(tmp_path):
    coverage = make_coverage(tmp_path, ZONE, [])
    probe = ScriptedProbe([False])
    result = preflight_optical_source(ZONE, WINDOW, land_mask_path=coverage, probe=probe)
    assert result.finding is SourceFinding.INCONCLUSIVE
    assert result.probes == 0 and probe.calls == []


# ---------------------------------------------------------------------------
# Block geometry: the decisiveness of "all blocks missed" rests entirely on the
# blocks jointly covering every live tile.
# ---------------------------------------------------------------------------


def _tile_centre_lonlat(spec, r: int, c: int) -> tuple[float, float]:
    tile_m = SHARD_PX * PIXEL_M
    e = spec.easting[0] + (c + 0.5) * tile_m
    n = spec.northing[1] - (r + 0.5) * tile_m
    lon, lat = Transformer.from_crs(int(spec.epsg), 4326, always_xy=True).transform(e, n)
    return float(lon), float(lat)


def _bitmap(spec, live):
    tl = np.zeros((spec.height // SHARD_PX, spec.width // SHARD_PX), dtype=bool)
    for r, c in live:
        tl[r, c] = True
    return tl


def test_blocks_cover_every_live_tile_and_only_live_blocks_are_emitted():
    spec = zone_grid.zone("33N")
    live = [(3, 2), (3, 3), (12, 30), (200, 15), (201, 15)]
    tl = _bitmap(spec, live)
    blocks = live_tile_block_bboxes_wgs84(spec, tl, block_tiles=8)

    assert sum(w for _bbox, w in blocks) == len(live)
    for r, c in live:
        lon, lat = _tile_centre_lonlat(spec, r, c)
        assert any(minx <= lon <= maxx and miny <= lat <= maxy for (minx, miny, maxx, maxy), _w in blocks), (
            f"live tile ({r},{c}) centre is outside every block bbox"
        )
    # Distant clusters land in distinct blocks — the sweep is finer than the envelope.
    assert len(blocks) >= 3


def test_ingest_query_envelope_contains_every_block():
    """The ingest searches the live-tile envelope; the blocks must sit inside it, or a
    block could report items the ingest's own query would never see.
    """
    spec = zone_grid.zone("33N")
    tl = _bitmap(spec, [(3, 2), (200, 15)])
    minx, miny, maxx, maxy = _live_tile_bbox_wgs84(spec, tl)
    for (bminx, bminy, bmaxx, bmaxy), _w in live_tile_block_bboxes_wgs84(spec, tl, block_tiles=8):
        assert minx <= bminx and bmaxx <= maxx and miny <= bminy and bmaxy <= maxy


def test_single_tile_blocks_are_tight_to_their_tile():
    spec = zone_grid.zone("33N")
    # A mid-latitude tile: near the pole a 20 km tile legitimately spans degrees of
    # longitude, which would make any fixed threshold measure latitude, not tightness.
    row = (spec.height // SHARD_PX) // 2
    tl = _bitmap(spec, [(row, 10)])
    blocks = live_tile_block_bboxes_wgs84(spec, tl, block_tiles=1)
    assert len(blocks) == 1
    (bbox, weight) = blocks[0]
    assert weight == 1
    lon, lat = _tile_centre_lonlat(spec, row, 10)
    minx, miny, maxx, maxy = bbox
    assert minx <= lon <= maxx and miny <= lat <= maxy
    # One 2048-px tile is ~20 km; its box must be tile-sized, not block- or zone-sized.
    assert (maxx - minx) < 1.0 and (maxy - miny) < 1.0


def test_block_tiles_must_be_positive():
    spec = zone_grid.zone("33N")
    with pytest.raises(ValueError):
        live_tile_block_bboxes_wgs84(spec, _bitmap(spec, [(0, 0)]), block_tiles=0)
