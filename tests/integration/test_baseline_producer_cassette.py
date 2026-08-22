"""The producer split, recorded from the live catalogue.

**This cassette exists because no other fixture in the repo covers the raw-producer case.**
Every item in ``test_s2_roi_parity.yaml`` has its read bands in Element 84's COG bucket — the
416 ESA references in it are *extra* assets — so the parity suite can exercise the exemption
and not the correction. That asymmetry is exactly how a dead code path survives a green suite.

**Two windows are recorded, because no single one carries every case.**

The first is MGRS-33TWM in December 2017, where Element 84's 2026-08 backfill indexed items
whose primary band keys point at ESA's archive. Two things make this window the right choice:

* it contains items from **both** producers, which is the coverage that was missing, and
* it contains **2017-12-19**, a real solar day that fuses a harmonised COG at baseline 05.00
  with a raw item at 02.06 — kept apart because their acquisition instants are 209 s apart, so
  duplicate selection treats them as distinct acquisitions and both reach the loader.

That date is the regression this file most needs to pin. An early version refused any
mixed-producer date, which would have discarded it even though nothing in it is owed a correction.
The offset is now decided per reflectance asset (``boa_offset.source_decision``), so a mixed day
needs no date-wide answer at all — but the recorded items are still the only real evidence that
the two producers are told apart correctly, which is what these assertions check.

The second window is MGRS-33NVB in February 2022, four days of it, and it closes a gap this file
used to admit in a docstring. Every raw item in the 2017 window predates version 04.00, so all of
them are exempt on version alone and **the subtraction was never exercised on a real image** — it
was covered by rewriting the version on an otherwise real one, with a note saying the combination
"does not exist upstream".

**It does.** A census of seven zone-months on 2026-08-22 found 680 reflectance sources served from
ESA's bucket at version 04.00 or later, across 68 real images. This window holds one of them beside
one of Element 84's, **both declaring 04.00 exactly** — so the version cannot be what separates
them, and the pair pins the threshold as inclusive at the same time.

Run::

    uv run pytest tests/integration/test_baseline_producer_cassette.py -m integration

To re-record (rare — see tests/fixtures/stac_cassettes/README.md)::

    uv run pytest tests/integration/test_baseline_producer_cassette.py \
        -m integration --record-mode=once
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import pytest

from tessera_embeddings.config.satellites import S2_BASELINE_THRESHOLD
from tessera_embeddings.ingest.asset_locations import (
    REFLECTANCE_ASSET_KEYS,
    Harmonisation,
    asset_bucket,
    asset_href,
    item_harmonisation,
)
from tessera_embeddings.ingest.boa_offset import OffsetDecision, source_decision
from tessera_embeddings.ingest.duplicates import (
    acquisition_identity,
    acquisition_instant,
    select_preferred_duplicates,
)
from tessera_embeddings.ingest.item_baselines import processing_baseline
from tessera_embeddings.ingest.solar_days import normalize_to_solar_day
from tessera_embeddings.ingest.stac import extract_baselines

REPO_ROOT = Path(__file__).resolve().parents[2]
CASSETTE_DIR = REPO_ROOT / "tests" / "fixtures" / "stac_cassettes"

CATALOGUE = "https://earth-search.aws.element84.com/v1"
TILE = "MGRS-33TWM"
# December alone: 6 items, both producers, and the mixed day. pytest-recording names a
# cassette per TEST, so a wide window would record the same query five times over.
WINDOW = ("2017-12-01T00:00:00Z", "2018-01-01T00:00:00Z")
#: The mixed-producer day this fixture exists to pin. Real, and owed no correction.
MIXED_DAY = "2017-12-19"

#: The second window, and the two real items in it whose pixels still carry the +1000 offset.
TILE_AT_THRESHOLD = "MGRS-33NVB"
#: Four days, which is the whole month's worth of evidence: it returns exactly one image from
#: each producer, both at 04.00. A wider window adds five more harmonised images that say the
#: same thing and triples the recording.
WINDOW_AT_THRESHOLD = ("2022-02-06T00:00:00Z", "2022-02-10T00:00:00Z")
NEEDS_SUBTRACTION_IDS = frozenset({"S2B_33NVB_20220207_0_L2A"})


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """Centralise under tests/fixtures/stac_cassettes/ so the safety guard sees it."""
    return str(CASSETTE_DIR)


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    """Filter credentials out of the recording. Belt-and-braces with the safety guard."""
    return {
        "filter_headers": ["authorization", "x-amz-security-token", "cookie"],
        "filter_query_parameters": ["X-Amz-Signature", "X-Amz-Security-Token"],
        "decode_compressed_response": True,
    }


def _search(tile: str, window: tuple[str, str]) -> list[Any]:
    """Every item for one tile in one window, from the catalogue or the cassette."""
    pystac_client = pytest.importorskip("pystac_client")
    client = pystac_client.Client.open(CATALOGUE)
    search = client.search(
        collections=["sentinel-2-l2a"],
        query={"grid:code": {"eq": tile}},
        datetime=f"{window[0]}/{window[1]}",
    )
    return list(search.items())


def _items() -> list[Any]:
    """Every 2017 item for the tile, from the catalogue or the cassette."""
    return _search(TILE, WINDOW)


#: UTM zone 33's central meridian. `normalize_to_solar_day` needs the ROI's longitude, and every
#: item in this fixture is on one tile, so the tile's own zone is the right answer.
TILE_MID_LONGITUDE = 15.0


def _normalised() -> list[Any]:
    """The fixture's items, through the chokepoint the real pipeline puts them through.

    The correction path derives the solar day with `solar_day_of`, which refuses an
    item that has not been normalised rather than deriving a plausible-looking wrong day from its
    UTC stamp. The ingest normalises at the catalogue chokepoint, so a test that skips it is
    testing a call the pipeline never makes — and this guard is what caught these two doing it.
    """
    items = _items()
    normalize_to_solar_day(items, mid_longitude=TILE_MID_LONGITUDE)
    return items


def _decisions(items: list) -> set[OffsetDecision]:
    """What is owed to every reflectance source of these recorded items."""
    out: set[OffsetDecision] = set()
    for item in items:
        baseline = processing_baseline(item)
        for key in REFLECTANCE_ASSET_KEYS:
            href = asset_href(item.assets.get(key))
            out.add(source_decision(asset_bucket(href) if href else None, baseline, S2_BASELINE_THRESHOLD))
    return out


@pytest.mark.integration
@pytest.mark.vcr
class TestRecordedProducerSplit:
    """Assertions against real recorded items, not hand-built fakes."""

    def test_the_cassette_covers_both_producers(self) -> None:
        """THE COVERAGE THIS FILE EXISTS FOR, asserted so it cannot quietly lapse.

        If a re-recording ever returns only one producer, every test below still passes while
        measuring nothing — the same way the parity suite passes today without covering the raw
        case at all. This fails instead.
        """
        kinds = collections.Counter(item_harmonisation(item) for item in _items())
        assert kinds[Harmonisation.RAW] > 0, f"no raw-producer items recorded: {dict(kinds)}"
        assert kinds[Harmonisation.HARMONISED] > 0, f"no harmonised items recorded: {dict(kinds)}"
        assert kinds[Harmonisation.MIXED] == 0, "an item straddling producers would need its own handling"
        # The claim that makes UNKNOWN safe to refuse on. Real Element 84 items key their assets
        # by the configured band names, so the producer is always determinable and the refusal
        # never fires in production. If this ever trips, the catalogue changed its asset naming
        # and the ingest will stop rather than subtract 1000 from harmonised pixels.
        assert kinds[Harmonisation.UNKNOWN] == 0, (
            "a real item's producer could not be determined; the catalogue may have moved to "
            "native asset keys, which the harmonisation check cannot resolve"
        )

    def test_raw_items_report_old_baselines(self) -> None:
        """Why the correction never fires on today's archive: the backfill is of the OLD
        archive, so every raw item sits below the threshold.

        Scoped to this cassette's window, so it is evidence and not a census — the archive-wide
        version of this claim belongs to a standing check, not to a fixture. It still fails
        loudly if the backfill ever reaches modern baselines here.
        """
        raw = [it for it in _items() if item_harmonisation(it) is Harmonisation.RAW]
        assert raw, "fixture no longer covers the raw producer"
        baselines = {float(it.properties["s2:processing_baseline"]) * 100 for it in raw}
        assert max(baselines) < S2_BASELINE_THRESHOLD, (
            f"a raw item now reports baseline {max(baselines)} >= {S2_BASELINE_THRESHOLD}; the "
            f"correction path is live on real data and needs an end-to-end test"
        )

    def test_the_real_mixed_day_costs_nothing_to_decide(self) -> None:
        """The regression, on recorded items. A harmonised COG at 05.00 beside a raw item at 02.06.

        Nothing on this day is owed a correction — the raw copy predates the offset — so every one
        of its sources must decide EXEMPT. An early version refused every mixed-producer date and
        would have thrown the day away for an ambiguity that was not present.
        """
        on_day = [it for it in _normalised() if it.datetime.strftime("%Y-%m-%d") == MIXED_DAY]
        assert len(on_day) > 1, f"{MIXED_DAY} no longer carries multiple items; re-pick the fixture day"
        assert {item_harmonisation(it) for it in on_day} == {
            Harmonisation.HARMONISED,
            Harmonisation.RAW,
        }, "this day is supposed to be the mixed-producer case"
        assert _decisions(on_day) == {OffsetDecision.EXEMPT}

    def test_the_real_reprocessing_pair_reduces_to_one_copy(self) -> None:
        """The reported MEDIUM, checked against the recorded catalogue rather than a fixture.

        Both items on this day are reprocessings of ONE observation — same sensing time, orbit and
        tile — but their catalogue timestamps are more than three minutes apart, so clustering on
        those timestamps kept both and handed both to the loader to mosaic together. They share a
        datatake, which is what now reduces them.
        """
        on_day = [it for it in _normalised() if it.datetime.strftime("%Y-%m-%d") == MIXED_DAY]
        identities = {acquisition_identity(it) for it in on_day}
        assert identities and None not in identities, f"the recorded items name no datatake: {identities}"
        assert len(identities) == 1, f"{MIXED_DAY} is no longer one observation; re-pick the fixture day"

        skew = max(acquisition_instant(it) for it in on_day) - min(acquisition_instant(it) for it in on_day)
        assert skew.total_seconds() > 120, (
            f"the recorded copies are only {skew.total_seconds()}s apart, so timestamp clustering "
            f"would already have reduced them and this asserts nothing"
        )

        kept, alternates = select_preferred_duplicates(on_day)
        assert len(kept) == 1, f"two reprocessings of one observation were both loaded: {[i.id for i in kept]}"
        assert item_harmonisation(kept[0]) is Harmonisation.HARMONISED, "the harmonised copy is the one to keep"
        assert alternates, "the raw copy must stay on the fallback ladder"

    def test_provenance_records_the_real_baselines(self) -> None:
        """`extract_baselines` reaches the store's `baselines_applied`, so it must report what
        the items declare — including for harmonised items that are exempt from correction.
        """
        on_day = [it for it in _items() if it.datetime.strftime("%Y-%m-%d") == MIXED_DAY]
        declared = {float(it.properties["s2:processing_baseline"]) * 100 for it in on_day}
        assert extract_baselines(on_day)[MIXED_DAY] in declared, "provenance must be a real reported baseline"


@pytest.mark.integration
# No cassette name: pytest-recording writes one file per TEST, named for it, which is the
# convention every recording in this directory follows. Each test here costs a full copy of the
# same 7-item response, so there are deliberately only two.
@pytest.mark.vcr
class TestARealItemIsOwedTheCorrection:
    """A raw item at the version threshold, recorded rather than synthesised.

    This class replaces a test that rewrote the version on an otherwise real item and said so,
    because the combination was believed not to exist upstream. It does — see the module docstring
    for the census — and a fixture standing in for a case the archive actually has proves the
    routing and nothing about the data.
    """

    @staticmethod
    def _at_threshold() -> list[Any]:
        items = _search(TILE_AT_THRESHOLD, WINDOW_AT_THRESHOLD)
        normalize_to_solar_day(items, mid_longitude=TILE_MID_LONGITUDE)
        return items

    def test_the_version_does_not_decide_which_images_keep_the_offset(self) -> None:
        """THE case this file could not previously reach with real data, and its contrast.

        Both images here declare **04.00 exactly**, so the version cannot be what tells them
        apart — only where their bands are served from. One is ESA's and still carries the +1000
        offset; one is Element 84's and has already had it removed. That is the whole claim of
        `source_decision`, made against real images on both sides of it at once.

        Declaring 04.00 exactly pins the comparison as INCLUSIVE at the same time: one version
        lower and the ESA image would read as exempt, keeping an offset its pixels do carry.

        The identity assertion is the coverage guard. If a re-record finds Element 84 has
        re-pointed that href at its own COG, this fails loudly rather than passing while measuring
        nothing — the failure mode that let the subtraction go untested against real data in the
        first place.
        """
        items = self._at_threshold()
        by_producer = collections.defaultdict(set)
        for item in items:
            by_producer[item_harmonisation(item)].add(item.id)

        assert by_producer[Harmonisation.RAW] == NEEDS_SUBTRACTION_IDS, (
            f"the recorded raw items changed: {sorted(by_producer[Harmonisation.RAW])}"
        )
        assert by_producer[Harmonisation.HARMONISED], "the fixture must keep both producers"
        assert len(items) == 2, "the window is deliberately the smallest that carries the contrast"

        for item in items:
            assert processing_baseline(item) == S2_BASELINE_THRESHOLD, f"{item.id} is not at the threshold"
            expected = OffsetDecision.OWED if item.id in NEEDS_SUBTRACTION_IDS else OffsetDecision.EXEMPT
            assert _decisions([item]) == {expected}, f"{item.id} should be {expected.value}"
