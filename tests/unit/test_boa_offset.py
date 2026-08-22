"""The BOA offset decided per SOURCE, and applied as each image is read.

Replaces the date-wide decision these tests used to make. The old shape asked "is this solar day
owed the offset", which has no correct answer on a day whose tiles come from different producers or
sit on both sides of the 04.00 threshold — so such days were refused, at a measured cost of 347 days
of one region-year. The question is now asked of one reflectance asset at a time, which has an
answer always, and the correction is applied to each image before anything fuses it with another.

Three layers, tested separately because they fail separately:

* :func:`source_decision` — the whole decision, as a pure function of a bucket and a baseline.
* :class:`BoaOffsetParser` — stamps that decision onto each source at parse time, and refuses.
* :class:`_BoaCorrectingReader` — applies it to pixels, inside a real ``odc.stac.load``.
"""

from __future__ import annotations

import datetime as dt
import pickle
from typing import Any

import numpy as np
import pystac
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from rasterio.transform import from_origin

from tessera_embeddings.config.providers import PROVIDERS, CollectionConfig
from tessera_embeddings.config.satellites import S2_BASELINE_OFFSET, S2_BASELINE_THRESHOLD, S2_L2A_BANDS
from tessera_embeddings.ingest.asset_locations import Harmonisation, SettledProducer
from tessera_embeddings.ingest.boa_offset import OffsetDecision, source_decision
from tessera_embeddings.ingest.stac import (
    BoaOffsetParser,
    HeterogeneousProducerError,
    _BoaCorrectingDriver,
    _reflectance_asset_keys,
    collection_harmonisation,
    load_stac_items,
)

_HARMONISED_BUCKET = "sentinel-cogs"
_RAW_BUCKET = "sentinel-s2-l2a"
_UNLISTED_BUCKET = "nobody-has-classified-this"
_T = S2_BASELINE_THRESHOLD


class TestSourceDecision:
    """The decision itself. One asset, one baseline, one answer."""

    def test_a_harmonised_source_is_never_owed_the_offset(self) -> None:
        """Its producer already removed it, so nothing here can be owed."""
        for baseline in (None, 1, 206, _T, 510):
            assert source_decision(_HARMONISED_BUCKET, baseline, _T) is OffsetDecision.EXEMPT, baseline

    def test_a_harmonised_source_needs_no_readable_baseline(self) -> None:
        """Because no decision rests on it. Refusing here would cost a perfectly good copy."""
        assert source_decision(_HARMONISED_BUCKET, None, _T) is OffsetDecision.EXEMPT

    def test_a_raw_source_at_or_above_the_threshold_is_owed(self) -> None:
        assert source_decision(_RAW_BUCKET, _T, _T) is OffsetDecision.OWED, "the threshold is inclusive"
        assert source_decision(_RAW_BUCKET, 510, _T) is OffsetDecision.OWED

    def test_a_raw_source_below_the_threshold_is_owed_nothing(self) -> None:
        """The offset did not exist before 04.00, so there is nothing to remove."""
        assert source_decision(_RAW_BUCKET, 206, _T) is OffsetDecision.EXEMPT
        assert source_decision(_RAW_BUCKET, _T - 1, _T) is OffsetDecision.EXEMPT

    def test_a_raw_source_with_no_readable_baseline_refuses(self) -> None:
        """Correcting and exempting are wrong by the same amount in opposite directions.

        A missing or malformed value must not read as zero: that exempts a date whose pixels may
        carry the offset, silently.
        """
        assert source_decision(_RAW_BUCKET, None, _T) is OffsetDecision.UNDECIDABLE

    def test_an_unlisted_bucket_is_never_assumed_raw(self) -> None:
        """Membership of the unharmonised set is what makes a correction safe to apply.

        Inferring "raw" from "not harmonised" would correct the first mirror anyone stands up, by
        exactly the offset, with nothing said about it.
        """
        assert source_decision(_UNLISTED_BUCKET, 510, _T) is OffsetDecision.UNDECIDABLE
        assert source_decision(_UNLISTED_BUCKET, None, _T) is OffsetDecision.UNDECIDABLE

    def test_an_unlisted_bucket_below_the_threshold_costs_nothing(self) -> None:
        """No producer changes a pixel there, so the ambiguity has no consequence to refuse over."""
        assert source_decision(_UNLISTED_BUCKET, 206, _T) is OffsetDecision.EXEMPT

    def test_an_unparseable_href_is_treated_as_unclassified(self) -> None:
        """`None` means the href addresses no recognised bucket, which is not evidence of anything."""
        assert source_decision(None, 510, _T) is OffsetDecision.UNDECIDABLE
        assert source_decision(None, 206, _T) is OffsetDecision.EXEMPT

    def test_the_collection_answer_replaces_the_bucket_entirely(self) -> None:
        """Planetary Computer serves the whole archive unharmonised, from a container nobody has
        classified and never will. Consulting the bucket there would refuse every date it has.
        """
        for bucket in (_UNLISTED_BUCKET, None, _HARMONISED_BUCKET):
            assert source_decision(bucket, 510, _T, Harmonisation.RAW) is OffsetDecision.OWED, bucket
            assert source_decision(bucket, 206, _T, Harmonisation.RAW) is OffsetDecision.EXEMPT, bucket
            assert source_decision(bucket, None, _T, Harmonisation.RAW) is OffsetDecision.UNDECIDABLE, bucket

    def test_a_collection_known_harmonised_owes_nothing_anywhere(self) -> None:
        assert source_decision(_RAW_BUCKET, 510, _T, Harmonisation.HARMONISED) is OffsetDecision.EXEMPT

    def test_an_answer_naming_no_producer_cannot_be_passed_at_all(self) -> None:
        """MIXED and UNKNOWN used to reach here and be refused. Now they cannot arrive.

        ``known_harmonisation`` is typed :data:`SettledProducer`, which is the two states that name
        a producer — so an answer that names none is rejected by the type checker at the call site
        rather than detected here and turned into a refusal. That is why this function has no case
        for them: correcting on an unresolved producer is wrong by exactly the offset and silent,
        and the cheapest way to prevent it is to make it unsayable.

        Asserted on the TYPE rather than on behaviour, because there is no behaviour left to
        assert. This fails if someone widens the parameter back.
        """
        import typing

        settled = typing.get_args(SettledProducer)
        assert set(settled) == {Harmonisation.HARMONISED, Harmonisation.RAW}
        assert Harmonisation.MIXED not in settled
        assert Harmonisation.UNKNOWN not in settled


class TestNothingCanSupplyAnUnsettledProducer:
    """The refusal above is a dormant guard. These pin that, rather than restating it.

    ``known_harmonisation`` reaches :func:`source_decision` from exactly one place: the parser
    :func:`load_stac_items` builds, which takes it from :func:`collection_harmonisation`. That
    function can only answer ``RAW`` or ``None``, so no configured collection drives the
    undecidable branch today. Should one ever be able to, these fail — which turns the guard into
    a behaviour change somebody has to think about rather than one that ships silently.
    """

    def test_no_configured_collection_yields_an_unsettled_producer(self) -> None:
        answers = {
            (provider_name, alias): collection_harmonisation(config)
            for provider_name, provider in PROVIDERS.items()
            for alias, config in provider.collections.items()
        }
        assert answers, "no collections configured, so this census would pass vacuously"
        unsettled = {key: value for key, value in answers.items() if value not in (None, Harmonisation.RAW)}
        assert not unsettled, f"these collections would reach the undecidable branch: {unsettled}"

    def test_the_collection_answer_is_settled_for_every_configuration(self) -> None:
        """Exhaustive over the two fields :func:`collection_harmonisation` reads.

        The census above covers the collections that exist; this covers the ones that could be
        added, so a new entry cannot introduce an unsettled answer without failing here.
        """
        for varies in (False, True):
            for threshold in (None, S2_BASELINE_THRESHOLD):
                config = CollectionConfig(
                    collection_id="probe",
                    bands=S2_L2A_BANDS,
                    resolution=10,
                    baseline_threshold=threshold,
                    harmonisation_varies_by_item=varies,
                )
                answer = collection_harmonisation(config)
                assert answer in (None, Harmonisation.RAW), (varies, threshold, answer)


def _item(
    iid: str,
    *,
    hrefs: dict[str, str],
    baseline: str | None = "05.00",
    tile: str = "MGRS-33TWM",
    eo_names: dict[str, str] | None = None,
) -> pystac.Item:
    """A minimal S2-shaped item. ``eo_names`` gives an asset a common name, as Planetary Computer does."""
    props: dict[str, Any] = {"grid:code": tile, "s2:sequence": "0", "datetime": "2017-11-16T12:00:00Z"}
    if baseline is not None:
        props["s2:processing_baseline"] = baseline
    item = pystac.Item(
        id=iid,
        geometry=None,
        bbox=None,
        datetime=dt.datetime(2017, 11, 16, 12, tzinfo=dt.UTC),
        properties=props,
        collection="sentinel-2-l2a",
        stac_extensions=["https://stac-extensions.github.io/eo/v1.1.0/schema.json"],
    )
    for key, href in hrefs.items():
        extra: dict[str, Any] = {"raster:bands": [{"nodata": 0, "data_type": "uint16"}]}
        if eo_names and key in eo_names:
            extra["eo:bands"] = [{"name": key, "common_name": eo_names[key]}]
        item.add_asset(key, pystac.Asset(href=href, media_type=pystac.MediaType.COG, extra_fields=extra))
    return item


def _s2_hrefs(bucket: str) -> dict[str, str]:
    return {band: f"s3://{bucket}/33/T/WM/{band}.tif" for band in (*S2_L2A_BANDS, "scl")}


class TestTheParserStampsEachSource:
    """What reaches a source, and what refuses before any graph is built."""

    def _parse(self, item: pystac.Item, keys: frozenset[str] | None = None) -> BoaOffsetParser:
        parser = BoaOffsetParser({}, keys or frozenset(S2_L2A_BANDS), _T, S2_BASELINE_OFFSET, None)
        for band in (*S2_L2A_BANDS, "scl"):
            parser.driver_data(item, (band, 1))
        return parser

    def test_a_raw_item_over_the_threshold_stamps_the_offset(self) -> None:
        parser = self._parse(_item("raw", hrefs=_s2_hrefs(_RAW_BUCKET), baseline="05.00"))
        assert parser.owed == len(S2_L2A_BANDS)
        assert parser.stamped == len(S2_L2A_BANDS)

    def test_scl_is_never_stamped(self) -> None:
        """Structurally, not by a list of bands the corrector was told to skip.

        The scene classification layer is read, and it counts for locality, but it is categorical:
        subtracting the offset from a class label is meaningless. It carries no decision at all,
        which is a distinct state from a decision of zero.
        """
        item = _item("raw", hrefs=_s2_hrefs(_RAW_BUCKET), baseline="05.00")
        parser = self._parse(item)
        assert parser.driver_data(item, ("scl", 1)) is None
        assert parser.stamped == len(S2_L2A_BANDS), "scl did not add to the count"

    def test_a_harmonised_item_stamps_a_zero_rather_than_nothing(self) -> None:
        """A decided-owes-nothing source must be distinguishable from one never asked about.

        A reader that could not tell them apart would treat a wiring failure as a correct exemption.
        """
        item = _item("harm", hrefs=_s2_hrefs(_HARMONISED_BUCKET), baseline="05.00")
        fresh = BoaOffsetParser({}, frozenset(S2_L2A_BANDS), _T, S2_BASELINE_OFFSET, None)
        assert fresh.driver_data(item, ("blue", 1)) == {"boa_offset": 0}
        parser = self._parse(item)
        assert parser.owed == 0
        assert parser.stamped == len(S2_L2A_BANDS)

    def test_an_item_whose_bands_span_two_producers_is_corrected_band_by_band(self) -> None:
        """This used to refuse its entire solar day. Per source, it is ordinary.

        The decision is a property of the asset, so a harmonised band and a raw band on one item
        simply get different answers.
        """
        hrefs = _s2_hrefs(_HARMONISED_BUCKET)
        hrefs["red"] = f"s3://{_RAW_BUCKET}/33/T/WM/red.tif"
        item = _item("mixed", hrefs=hrefs, baseline="05.00")
        parser = self._parse(item)
        assert parser.driver_data(item, ("red", 1)) == {"boa_offset": S2_BASELINE_OFFSET}
        assert parser.driver_data(item, ("blue", 1)) == {"boa_offset": 0}

    def test_an_unclassified_bucket_over_the_threshold_refuses(self) -> None:
        item = _item("unlisted", hrefs=_s2_hrefs(_UNLISTED_BUCKET), baseline="05.00")
        with pytest.raises(HeterogeneousProducerError, match="producer cannot be determined"):
            self._parse(item)

    def test_an_unreadable_baseline_on_a_raw_item_refuses(self) -> None:
        item = _item("no-baseline", hrefs=_s2_hrefs(_RAW_BUCKET), baseline=None)
        with pytest.raises(HeterogeneousProducerError, match="producer cannot be determined"):
            self._parse(item)

    def test_a_pre_threshold_item_never_refuses_however_unclassified(self) -> None:
        """Gated on the threshold, because below it no producer changes a pixel."""
        item = _item("old", hrefs=_s2_hrefs(_UNLISTED_BUCKET), baseline="02.06")
        assert self._parse(item).owed == 0

    def test_the_parser_is_tiny_and_does_not_grow_with_the_item_count(self) -> None:
        """The reason the decision rides on the SOURCE rather than in a table on the driver.

        The driver is embedded in every dask task and `distributed` serialises tasks individually,
        so a `(uri, band) -> offset` table is duplicated per task. Measured at ROI scale, a
        3,081-entry table cost 14.6 MB of graph across 64 chunk tasks.
        """
        parser = BoaOffsetParser({}, frozenset(S2_L2A_BANDS), _T, S2_BASELINE_OFFSET, None)
        assert len(pickle.dumps(_BoaCorrectingDriver(parser))) < 2048


class TestReflectanceAssetKeysComeFromOdc:
    """Which assets carry the reflectance bands, resolved the way the loader will resolve them."""

    def test_earth_search_names_its_assets_after_the_bands(self) -> None:
        item = _item("es", hrefs=_s2_hrefs(_HARMONISED_BUCKET))
        from tessera_embeddings.config import PROVIDERS

        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        assert _reflectance_asset_keys([item], config) == frozenset(S2_L2A_BANDS)

    def test_a_native_keyed_item_resolves_through_the_alias_table(self) -> None:
        """Planetary Computer serves ``blue`` as an asset called ``B02``.

        Deciding the offset against the CONFIGURED names would match none of its assets and correct
        nothing, silently — so the resolution has to be odc's own rather than a mapping kept here.
        """
        from tessera_embeddings.config import PROVIDERS

        config = PROVIDERS["planetary-computer"].collections["sentinel-2-l2a"]
        native = {"B02": "blue", "B03": "green"}
        item = _item(
            "pc",
            hrefs={k: f"https://x.blob.core.windows.net/{k}.tif" for k in (*native, "SCL")},
            eo_names=native,
        )
        resolved = _reflectance_asset_keys([item], config)
        assert "B02" in resolved and "B03" in resolved, resolved
        assert "SCL" not in resolved, "scl is read but never corrected"

    def test_an_unreadable_item_falls_through_to_the_next_one(self) -> None:
        from tessera_embeddings.config import PROVIDERS

        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        good = _item("good", hrefs=_s2_hrefs(_HARMONISED_BUCKET))
        assert _reflectance_asset_keys([object(), good], config) == frozenset(S2_L2A_BANDS)

    def test_nothing_resolvable_yields_an_empty_set(self) -> None:
        """Which the caller turns into a refusal or a fallback — it never means "nothing to do"."""
        from tessera_embeddings.config import PROVIDERS

        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        assert _reflectance_asset_keys([object()], config) == frozenset()


_UTM = "EPSG:32633"
_ORIGIN_N = 5_000_000.0
_ORIGIN_E = 400_000.0
_RES = 20
_SIZE = 32


def _write_cog(path: str, value: int, x0: float) -> None:
    pixels = np.full((_SIZE, _SIZE), value, dtype="uint16")
    pixels[0, 0] = 0  # nodata, the one code carrying no offset
    _write_values(path, pixels, x0)


def _write_values(path: str, pixels: np.ndarray, x0: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_SIZE,
        width=_SIZE,
        count=1,
        dtype="uint16",
        crs=_UTM,
        nodata=0,
        transform=from_origin(x0, _ORIGIN_N, _RES, _RES),
    ) as dataset:
        dataset.write(pixels, 1)


def _tile_item(iid: str, tmp_path: Any, *, value: int, x0: float, baseline: str, tile: str) -> pystac.Item:
    path = str(tmp_path / f"{iid}.tif")
    _write_cog(path, value, x0)
    transformer = Transformer.from_crs(_UTM, "EPSG:4326", always_xy=True)
    corners = [
        transformer.transform(x, y) for x in (x0, x0 + _RES * _SIZE) for y in (_ORIGIN_N - _RES * _SIZE, _ORIGIN_N)
    ]
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    item = _item(iid, hrefs=dict.fromkeys((*S2_L2A_BANDS, "scl"), path), baseline=baseline, tile=tile)
    item.bbox = bbox
    item.geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]
        ],
    }
    item.properties["proj:epsg"] = 32633
    item.properties["proj:shape"] = [_SIZE, _SIZE]
    item.properties["proj:transform"] = [_RES, 0, x0, 0, -_RES, _ORIGIN_N, 0, 0, 1]
    return item


class TestTheOffsetIsRemovedPerImage:
    """The pixels, through a real ``odc.stac.load``.

    Nothing else in the suite can catch a correction that never fires: every other test either
    replaces the loader or compares two arms of the same code. This drives the whole path — parser,
    driver, reader, fuser — over real GeoTIFFs.
    """

    def _load(self, items: list[pystac.Item], monkeypatch: pytest.MonkeyPatch, tiles: int) -> Any:
        # Only the href-to-bucket step is stubbed, because a local file addresses no bucket. That
        # step is unit-tested on its own in `test_asset_locations`; everything downstream is real.
        buckets = {item.assets["blue"].href: item.properties["_bucket"] for item in items}
        monkeypatch.setattr("tessera_embeddings.ingest.stac.asset_bucket", buckets.get)
        geobox = GeoBox.from_bbox(
            (_ORIGIN_E, _ORIGIN_N - _RES * _SIZE, _ORIGIN_E + tiles * _RES * _SIZE, _ORIGIN_N),
            _UTM,
            resolution=10,
        )
        return load_stac_items(
            items,
            provider="earth-search",
            collection="sentinel-2-l2a",
            extra_bands=["scl"],
            geobox=geobox,
            chunks={"time": 1, "northing": 320, "easting": 320},
        ).compute()

    def test_the_day_that_used_to_be_refused_now_loads_correctly(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zone 01N, 2017-11-16 in miniature — the day ADR 020 records as a total loss.

        Three tiles on one solar day: an ESA original at 05.00 that carries the offset, an ESA
        original at 00.01 that never had it, and an Element 84 COG that already had it removed. No
        single date-wide answer fits, so the whole day was refused and its 27 sound images lost
        with it. Per image, each of the three is simply right.
        """
        width = _RES * _SIZE
        raw_new = _tile_item("raw_v0500", tmp_path, value=1500, x0=_ORIGIN_E, baseline="05.00", tile="MGRS-33TWM")
        raw_old = _tile_item(
            "raw_v0001", tmp_path, value=1500, x0=_ORIGIN_E + width, baseline="00.01", tile="MGRS-33TWN"
        )
        harmonised = _tile_item(
            "e84_cog", tmp_path, value=400, x0=_ORIGIN_E + 2 * width, baseline="05.00", tile="MGRS-33TWP"
        )
        raw_new.properties["_bucket"] = _RAW_BUCKET
        raw_old.properties["_bucket"] = _RAW_BUCKET
        harmonised.properties["_bucket"] = _HARMONISED_BUCKET

        data = self._load([raw_new, raw_old, harmonised], monkeypatch, tiles=3)

        assert data.time.size == 1, "one solar day, one mosaic"
        blue = data.blue.isel(time=0).values
        stride = width // 10
        assert set(np.unique(blue[:, :stride])) == {0, 500}, "the ESA original at 05.00 lost the offset"
        assert set(np.unique(blue[:, stride : 2 * stride])) == {0, 1500}, "the pre-04.00 original was owed nothing"
        assert set(np.unique(blue[:, 2 * stride :])) == {0, 400}, "the harmonised COG was left alone"

    def test_the_stored_dtype_is_the_one_the_store_expects(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A signed result would make the store's dtype depend on which date landed first."""
        raw = _tile_item("raw", tmp_path, value=1500, x0=_ORIGIN_E, baseline="05.00", tile="MGRS-33TWM")
        raw.properties["_bucket"] = _RAW_BUCKET
        data = self._load([raw], monkeypatch, tiles=1)
        assert data.blue.dtype == np.uint16

    def test_nodata_survives_the_correction(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """DN 0 is the one code carrying no offset, and it must not become a dark observation."""
        raw = _tile_item("raw", tmp_path, value=1500, x0=_ORIGIN_E, baseline="05.00", tile="MGRS-33TWM")
        raw.properties["_bucket"] = _RAW_BUCKET
        data = self._load([raw], monkeypatch, tiles=1)
        assert 0 in np.unique(data.blue.isel(time=0).values)

    def test_scl_is_untouched_by_a_load_that_corrects_everything_else(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fixture serves scl from the same object as the bands, so an unchanged value proves
        the exclusion rather than merely being consistent with it.
        """
        raw = _tile_item("raw", tmp_path, value=1500, x0=_ORIGIN_E, baseline="05.00", tile="MGRS-33TWM")
        raw.properties["_bucket"] = _RAW_BUCKET
        data = self._load([raw], monkeypatch, tiles=1)
        assert set(np.unique(data.scl.isel(time=0).values)) == {0, 1500}
        assert set(np.unique(data.blue.isel(time=0).values)) == {0, 500}

    def test_a_refusal_escapes_before_any_pixel_is_read(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Items are parsed synchronously, so an undecidable source fails the call rather than a task.

        A task failure hours into a leg is attributed as a read error and sent to the fallback
        ladder, which cannot recover from a refusal.
        """
        unlisted = _tile_item("unlisted", tmp_path, value=1500, x0=_ORIGIN_E, baseline="05.00", tile="MGRS-33TWM")
        unlisted.properties["_bucket"] = _UNLISTED_BUCKET
        with pytest.raises(HeterogeneousProducerError):
            self._load([unlisted], monkeypatch, tiles=1)


class TestNoExistingPixelChanges:
    """The additivity claim, which is what makes this mergeable without re-ingesting anything.

    The amount subtracted is a CONSTANT: the processing baseline decides only *whether* the offset
    is removed, never how much. So on every day the pipeline already loaded, moving the decision
    from per-date to per-source is arithmetically the same operation, and the pixels must be
    bit-identical. Only days that previously REFUSED change, and those had no output to preserve.

    Asserted against a reference implementation of the deleted date-wide corrector rather than
    argued, because "no store needs re-ingesting" is the claim a reviewer has to trust.
    """

    @staticmethod
    def _date_wide_reference(pixels: np.ndarray, owed: bool) -> np.ndarray:
        """What the deleted date-wide corrector did to one band of one time slice."""
        if not owed:
            return pixels
        shifted = np.clip(pixels.astype(np.int32) + S2_BASELINE_OFFSET, 1, None)
        return np.where(pixels > 0, shifted, pixels.astype(np.int32)).astype(pixels.dtype)

    @pytest.mark.parametrize(
        ("bucket", "baseline", "owed"),
        [
            (_HARMONISED_BUCKET, "05.09", False),
            (_RAW_BUCKET, "05.09", True),
            (_RAW_BUCKET, "04.00", True),
            (_RAW_BUCKET, "02.06", False),
        ],
    )
    def test_a_day_that_used_to_load_is_unchanged(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, bucket: str, baseline: str, owed: bool
    ) -> None:
        rng = np.random.default_rng(7)
        values = rng.integers(0, 4000, size=(_SIZE, _SIZE)).astype("uint16")
        # nodata, the two ends of the dark band, and the floor boundary, all present on purpose
        values[0, :4] = [0, 1, 999, 1000]

        def _load(host: str) -> np.ndarray:
            item = _tile_item(
                f"{host}-{baseline}", tmp_path, value=0, x0=_ORIGIN_E, baseline=baseline, tile="MGRS-33TWM"
            )
            _write_values(str(tmp_path / f"{item.id}.tif"), values, _ORIGIN_E)
            item.properties["_bucket"] = host
            monkeypatch.setattr(
                "tessera_embeddings.ingest.stac.asset_bucket",
                {item.assets["blue"].href: host}.get,
            )
            geobox = GeoBox.from_bbox(
                (_ORIGIN_E, _ORIGIN_N - _RES * _SIZE, _ORIGIN_E + _RES * _SIZE, _ORIGIN_N), _UTM, resolution=10
            )
            loaded = load_stac_items(
                [item],
                provider="earth-search",
                collection="sentinel-2-l2a",
                extra_bands=["scl"],
                geobox=geobox,
                chunks={"time": 1, "northing": 320, "easting": 320},
            ).compute()
            return loaded.blue.isel(time=0).values

        # The same load with the producer classified harmonised gives the UNCORRECTED pixels, which
        # is the input the old date-wide corrector would have been handed.
        uncorrected = _load(_HARMONISED_BUCKET)
        actual = _load(bucket)
        np.testing.assert_array_equal(actual, self._date_wide_reference(uncorrected, owed))


class TestTheDriverDoesNotDisturbTheRead:
    """What the wrapper must NOT change, because other machinery depends on it."""

    def test_it_leaves_the_credential_session_hook_alone(self) -> None:
        """`ingest.auth` patches `odc.loader._rio.ThreadSession` to refresh credentials on
        long-lived workers, and the patch is reached through `restore_env`, not through `open`.
        Overriding either environment method would silently take workers off credential refresh.
        """
        from odc.loader import RioDriver

        for method in ("restore_env", "capture_env", "new_load", "finalise_load"):
            assert getattr(_BoaCorrectingDriver, method, None) is getattr(RioDriver, method, None), method

    def test_it_reads_through_odcs_own_reader(self) -> None:
        """So the GDAL environment, the retry behaviour and the log records
        `ingest.loader_failures` attributes back to an object are all the ones odc produces.
        """
        from odc.loader import RioReader

        from tessera_embeddings.ingest.stac import _BoaCorrectingReader

        assert issubclass(_BoaCorrectingReader, RioReader)
        assert _BoaCorrectingReader.read is not RioReader.read, "and it does add the correction"
