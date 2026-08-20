"""Attributing an unreadable read to the object the loader actually gave up on.

The loader names that object in its own log record and raises an exception that does not
carry it. Without the name, the duplicate ladder has to step every duplicated tile-date in
the date, and these tests hold the two properties that makes the difference between:

* **blast radius** — one bad object steps ONE tile-date down, not every tile that happens to
  have an alternate; and
* **termination** — a bad object with no alternate is given up immediately, instead of
  walking every other tile's ladder first at a full re-read of the date per rung.

Both properties are stated against a date shaped like a real wide-ROI one: many duplicated
tiles, one of them broken.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from tessera_embeddings.ingest.duplicates import (
    alternates_for,
    copies_label,
    select_preferred_duplicates,
    step_down_copies,
)
from tessera_embeddings.ingest.loader_failures import (
    collect_aborted_hrefs,
    drain_local,
    href_key,
    implicated_tile_dates,
    install_capture,
    label_objects,
)

#: The message odc's reader emits for the object it gave up on, verbatim from a fleet log.
ABORT_MESSAGE = (
    "Aborting load due to failure while reading: "
    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/34/W/FA/2021/9/"
    "S2B_34WFA_20210908_1_L2A/B02.tif:1"
)


class _Asset:
    def __init__(self, href: str) -> None:
        self.href = href


class _Item:
    """Stands in for a pystac Item, carrying only what attribution reads."""

    def __init__(self, ident: str, tile: str, sequence: int, assets: dict[str, str] | None = None) -> None:
        self.id = ident
        # normalize_to_solar_day stamps noon UTC and solar_day_of raises on anything else.
        self.datetime = datetime(2021, 9, 8, 12, 0, 0, tzinfo=UTC)
        self.properties: dict[str, object] = {"grid:code": tile, "s2:sequence": str(sequence)}
        self.assets = {k: _Asset(v) for k, v in (assets or {}).items()}


def _copy(tile: str, sequence: int) -> _Item:
    """One catalogue copy of one tile-date, with the asset href the loader would open."""
    bare = tile.removeprefix("MGRS-")
    zone, band, square = bare[:-3], bare[-3], bare[-2:]
    ident = f"S2B_{bare}_20210908_{sequence}_L2A"
    return _Item(
        ident,
        tile,
        sequence,
        assets={"B02": f"s3://sentinel-cogs/sentinel-s2-l2a-cogs/{zone}/{band}/{square}/2021/9/{ident}/B02.tif"},
    )


def _wide_date(n_duplicated: int = 40) -> list[_Item]:
    """A date shaped like a wide ROI's: many tiles duplicated, 34WFA among them."""
    items: list[_Item] = [_copy("MGRS-34WFA", 0), _copy("MGRS-34WFA", 1)]
    for i in range(n_duplicated):
        tile = f"MGRS-33T{chr(ord('A') + i % 26)}{chr(ord('A') + i // 26)}"
        items.extend([_copy(tile, 0), _copy(tile, 1)])
    return items


class TestNamingTheObject:
    """Turning the loader's log line into an object identity a caller can match."""

    def test_the_href_survives_the_band_suffix(self) -> None:
        """The loader appends ``:1`` for the band; the object's identity does not include it."""
        assert href_key(ABORT_MESSAGE.split(": ", 1)[1]) == "S2B_34WFA_20210908_1_L2A/B02.tif"

    def test_the_three_url_spellings_agree(self) -> None:
        """A catalogue asset, a virtual-hosted URL and a path-style URL name one object.

        If they did not agree, every comparison between what the loader opened and what the
        catalogue offered would silently find nothing.
        """
        key = "S2B_34WFA_20210908_1_L2A/B02.tif"
        prefix = "sentinel-s2-l2a-cogs/34/W/FA/2021/9"
        assert href_key(f"s3://sentinel-cogs/{prefix}/{key}") == key
        assert href_key(f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/{prefix}/{key}") == key
        assert href_key(f"https://s3.us-west-2.amazonaws.com/sentinel-cogs/{prefix}/{key}") == key

    def test_the_label_names_nothing_when_nothing_was_captured(self) -> None:
        """``unattributed`` is a distinct answer from a list of objects, and must read as one."""
        assert label_objects([]) == "unattributed"

    def test_the_label_caps_and_says_it_capped(self) -> None:
        hrefs = [f"s3://b/p/S2B_33TAA_2021090{i}_0_L2A/B02.tif" for i in range(20)]
        label = label_objects(hrefs)
        assert label.endswith("+12 more")
        assert label.count(",") == 8


class TestCapturingWhatTheLoaderReports:
    """The handler that records the loader's own abort messages."""

    def test_the_capture_records_the_href_the_loader_names(self) -> None:
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        assert [href_key(h) for h in drain_local()] == ["S2B_34WFA_20210908_1_L2A/B02.tif"]

    def test_installing_twice_does_not_double_the_record(self) -> None:
        """The worker plugin's setup runs again on reconnect, and a second handler would
        report every object twice — which reads as two failing objects.
        """
        install_capture()
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        assert len(drain_local()) == 1

    def test_draining_clears(self) -> None:
        """One failure's attribution must not be inherited by the next."""
        install_capture()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        drain_local()
        assert drain_local() == []

    def test_an_unrelated_error_is_not_captured(self) -> None:
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error("Something else entirely went wrong")
        assert drain_local() == []

    def test_collection_without_a_cluster_returns_the_local_record(self) -> None:
        """Serial runs have no client, and attribution must still work there."""
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        assert len(collect_aborted_hrefs(None)) == 1


class TestMappingObjectsBackToCopies:
    """From an aborted href to the ``(tile, solar day)`` whose copy produced it."""

    def test_the_granule_id_in_the_path_is_enough(self) -> None:
        """Covers a catalogue whose asset hrefs are signed, aliased, or simply absent."""
        item = _Item("S2B_34WFA_20210908_1_L2A", "MGRS-34WFA", 1)
        keys = implicated_tile_dates([item], [ABORT_MESSAGE.split(": ", 1)[1]])
        assert keys == {("MGRS-34WFA", "2021-09-08")}

    def test_the_asset_href_matches_across_spellings(self) -> None:
        item = _copy("MGRS-34WFA", 1)
        item.id = "an-id-that-appears-nowhere-in-the-url"
        keys = implicated_tile_dates([item], [ABORT_MESSAGE.split(": ", 1)[1]])
        assert keys == {("MGRS-34WFA", "2021-09-08")}

    def test_an_href_from_another_date_attributes_nothing(self) -> None:
        """A fleet's workers read many dates at once, so a foreign href is expected. Blaming
        a tile for it would step down a copy that read perfectly well.
        """
        assert implicated_tile_dates([_copy("MGRS-34WFA", 1)], ["s3://b/p/S2B_33TAA_20210101_0_L2A/B02.tif"]) == set()

    def test_no_hrefs_attributes_nothing(self) -> None:
        assert implicated_tile_dates([_copy("MGRS-34WFA", 1)], []) == set()


class TestBlastRadius:
    """What a single bad object costs the rest of the date."""

    def test_attribution_steps_down_exactly_the_blamed_tile(self) -> None:
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        blamed = implicated_tile_dates(kept, [ABORT_MESSAGE.split(": ", 1)[1]])

        result = step_down_copies(alternates, kept, only=blamed)
        assert result is not None
        stepped_items, swapped = result
        assert swapped == {("MGRS-34WFA", "2021-09-08")}
        # Every other tile keeps the newer copy it was chosen for.
        downgraded = {i.id for a, i in zip(kept, stepped_items, strict=True) if a is not i}
        assert downgraded == {"S2B_34WFA_20210908_0_L2A"}

    def test_without_attribution_the_whole_date_steps_down(self) -> None:
        """The behaviour to be avoided, asserted so the improvement is measurable rather than
        asserted: 41 duplicated tiles all lose their newer copy for one bad object.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)

        result = step_down_copies(alternates, kept, only=None)
        assert result is not None
        _, swapped = result
        assert len(swapped) == 41

    def test_an_empty_attribution_is_not_a_licence_to_step_everything(self) -> None:
        """Attributed to nothing in this date is a DIFFERENT answer from unattributed, and
        must not fall through to the unattributed behaviour.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        assert step_down_copies(alternates, kept, only=set()) is None

    def test_a_failure_this_date_cannot_claim_is_unattributed_not_attributed_to_nothing(self) -> None:
        """The other half of the pair above, and the one the caller has to get right.

        ``collect_aborted_hrefs`` drains a CLUSTER-WIDE record, so the hrefs it returns can
        belong to another date entirely, or be a stale line from a failure already handled.
        Matching none of them against this date means "could not attribute" — not
        "attributed to nothing". Handing the empty set straight to ``step_down_copies``
        conflates the two, steps down no copy, and records a date with a perfectly good
        older copy as permanently lost. The consume path coerces it to ``None`` so the
        whole-date ladder still runs, which is what it did before attribution existed.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        foreign = ["s3://b/p/S2B_33TAA_20210101_0_L2A/B02.tif"]

        assert implicated_tile_dates(kept, foreign) == set()  # nothing in this date matches
        # What s2_roi._consume passes on, having read that as "unknown":
        blamed = implicated_tile_dates(kept, foreign) or None
        assert step_down_copies(alternates, kept, only=blamed) is not None


class TestTermination:
    """When the date is given up, and how many rungs that takes."""

    def test_a_blamed_tile_with_no_alternate_gives_up_at_once(self) -> None:
        """The expensive case: the broken object is a single-copy tile. Every other tile still
        has alternates, so an unattributed ladder walks them all — each rung a full re-read of
        the date — before reaching the same answer.
        """
        items = [_copy("MGRS-34WFA", 1)]  # one copy only
        for i in range(40):
            tile = f"MGRS-33T{chr(ord('A') + i)}A"
            items.extend([_copy(tile, 0), _copy(tile, 1)])
        kept, alternates = select_preferred_duplicates(items)
        blamed = implicated_tile_dates(kept, [ABORT_MESSAGE.split(": ", 1)[1]])

        assert blamed == {("MGRS-34WFA", "2021-09-08")}
        assert step_down_copies(alternates, kept, only=blamed) is None
        assert step_down_copies(alternates, kept, only=None) is not None

    def test_a_copy_that_failed_is_never_offered_again(self) -> None:
        """The ladder consumes what it hands out, or a retry loop cannot terminate."""
        items = [_copy("MGRS-34WFA", 0), _copy("MGRS-34WFA", 1), _copy("MGRS-34WFA", 2)]
        kept, alternates = select_preferred_duplicates(items)
        blamed = {("MGRS-34WFA", "2021-09-08")}

        first = step_down_copies(alternates, kept, only=blamed)
        assert first is not None
        second = step_down_copies(alternates, first[0], only=blamed)
        assert second is not None
        assert [i.id for i in first[0]] != [i.id for i in second[0]]
        assert step_down_copies(alternates, second[0], only=blamed) is None


class TestWhatTheLogSays:
    """The labels a reader uses to tell what changed."""

    def test_the_label_names_only_what_was_stepped(self) -> None:
        """A whole-date label on a wide ROI runs to tens of thousands of characters and leaves
        the reader to diff two of them to find the one tile that moved.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        blamed = {("MGRS-34WFA", "2021-09-08")}
        result = step_down_copies(alternates, kept, only=blamed)
        assert result is not None
        stepped_items, swapped = result

        assert copies_label(kept, only=swapped) == "MGRS-34WFA#1"
        assert copies_label(stepped_items, only=swapped) == "MGRS-34WFA#0"

    def test_a_whole_date_label_is_capped(self) -> None:
        items = _wide_date()
        kept, _ = select_preferred_duplicates(items)
        assert copies_label(kept).endswith("more")

    def test_narrowing_alternates_is_an_intersection_not_a_replacement(self) -> None:
        """``only`` must not be able to reach a tile-date absent from ``items`` — a date's
        ladder reaching into another date's copies would swap imagery for a day that read.
        """
        items = _wide_date(n_duplicated=1)
        kept, alternates = select_preferred_duplicates(items)
        foreign = ("MGRS-99XXX", "2021-09-08")
        assert foreign not in alternates_for(alternates, kept, only={foreign})
