"""The producer split, recorded from the live catalogue.

**This cassette exists because no other fixture in the repo covers the raw-producer case.**
Every item in ``test_s2_roi_parity.yaml`` has its read bands in Element 84's COG bucket — the
416 ESA references in it are *extra* assets — so the parity suite can exercise the exemption
and not the correction. That asymmetry is exactly how a dead code path survives a green suite.

Recorded over MGRS-33TWM in December 2017, where Element 84's 2026-08 backfill indexed items
whose primary band keys point at ESA's archive. Two things make this window the right choice:

* it contains items from **both** producers, which is the coverage that was missing, and
* it contains **2017-12-19**, a real solar day that fuses a harmonised COG at baseline 05.00
  with a raw item at 02.06 — kept apart because their acquisition instants are 209 s apart, so
  duplicate selection treats them as distinct acquisitions and both reach the loader.

That date is the regression this file most needs to pin. An earlier version of
``dates_exempt_from_correction`` refused any mixed-producer date, which would have discarded it
even though nothing in it is owed a correction.

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
from tessera_embeddings.ingest.asset_locations import Harmonisation, item_harmonisation
from tessera_embeddings.ingest.duplicates import (
    acquisition_identity,
    acquisition_instant,
    select_preferred_duplicates,
)
from tessera_embeddings.ingest.solar_days import normalize_to_solar_day
from tessera_embeddings.ingest.stac import (
    dates_exempt_from_correction,
    extract_baselines,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CASSETTE_DIR = REPO_ROOT / "tests" / "fixtures" / "stac_cassettes"
CASSETTE_NAME = "test_s2_producer_split_33twm_2017.yaml"

CATALOGUE = "https://earth-search.aws.element84.com/v1"
TILE = "MGRS-33TWM"
# December alone: 6 items, both producers, and the mixed day. pytest-recording names a
# cassette per TEST, so a wide window would record the same query five times over.
WINDOW = ("2017-12-01T00:00:00Z", "2018-01-01T00:00:00Z")
#: The mixed-producer day this fixture exists to pin. Real, and owed no correction.
MIXED_DAY = "2017-12-19"


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


def _items() -> list[Any]:
    """Every 2017 item for the tile, from the catalogue or the cassette."""
    pystac_client = pytest.importorskip("pystac_client")
    client = pystac_client.Client.open(CATALOGUE)
    search = client.search(
        collections=["sentinel-2-l2a"],
        query={"grid:code": {"eq": TILE}},
        datetime=f"{WINDOW[0]}/{WINDOW[1]}",
    )
    return list(search.items())


#: UTM zone 33's central meridian. `normalize_to_solar_day` needs the ROI's longitude, and every
#: item in this fixture is on one tile, so the tile's own zone is the right answer.
TILE_MID_LONGITUDE = 15.0


def _normalised() -> list[Any]:
    """The fixture's items, through the chokepoint the real pipeline puts them through.

    `dates_exempt_from_correction` derives the solar day with `solar_day_of`, which refuses an
    item that has not been normalised rather than deriving a plausible-looking wrong day from its
    UTC stamp. The ingest normalises at the catalogue chokepoint, so a test that skips it is
    testing a call the pipeline never makes — and this guard is what caught these two doing it.
    """
    items = _items()
    normalize_to_solar_day(items, mid_longitude=TILE_MID_LONGITUDE)
    return items


@pytest.mark.integration
@pytest.mark.vcr(CASSETTE_NAME)
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

    def test_the_real_mixed_day_is_exempt_and_not_refused(self) -> None:
        """The regression. This day fuses a harmonised COG at 05.00 with a raw item at 02.06.

        Nothing in it is owed a correction, so it must be EXEMPT. An earlier version refused
        every mixed-producer date and would have thrown this away — a real date lost to an
        ambiguity that was not present.
        """
        on_day = [it for it in _normalised() if it.datetime.strftime("%Y-%m-%d") == MIXED_DAY]
        assert len(on_day) > 1, f"{MIXED_DAY} no longer carries multiple items; re-pick the fixture day"
        assert {item_harmonisation(it) for it in on_day} == {
            Harmonisation.HARMONISED,
            Harmonisation.RAW,
        }, "this day is supposed to be the mixed-producer case"
        assert MIXED_DAY in dates_exempt_from_correction(on_day)

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

    def test_a_raw_item_over_the_threshold_would_be_corrected(self) -> None:
        """The combination the change was built for, which **does not exist upstream**: a raw
        item at baseline >= 04.00. Reached by rewriting the baseline on an otherwise real item,
        which is the only way to exercise it — stated plainly because a synthetic property on
        real structure is evidence about the routing and not about the archive.
        """
        raw = next(it for it in _normalised() if item_harmonisation(it) is Harmonisation.RAW)
        raw.properties["s2:processing_baseline"] = "05.00"
        day = raw.datetime.strftime("%Y-%m-%d")
        assert day not in dates_exempt_from_correction([raw]), "a raw item over the threshold is owed the offset"
        assert extract_baselines([raw])[day] == 500
