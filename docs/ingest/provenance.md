# What the ingest records about its granules

When the `provenance` switch is on, the ingest writes two records into every mosaic it builds:
the list of granules fused into each day's slot, and, for every pixel of the slot, which of
those granules supplied it. Assembly later publishes them beside the registry, so a reader of the
store can answer "what imagery is behind this embedding" after the mosaics are deleted. This
document covers the ingest's half. The whole record, and why it is shaped as it is, is in
[`context_docs/storage/registry-provenance-plan.md`](../../context_docs/storage/registry-provenance-plan.md).

## The switch

The record is for the public global store. Dev runs and Arbol's own internal stores do not need
it, so it is off by default and nothing about a mosaic changes while it is off.

| where | how it is set |
|---|---|
| the campaign flows (`ingest-zone-year`, `run-global-campaign`, `fill-zones-sequential`) | `ingest_settings.provenance` |
| the base ingest flows and their tasks | a flat `provenance` parameter |
| the domain functions | `ingest_s2_roi_reflectance(provenance=…)`, `ingest_s1_roi_sar(provenance=…)` |

A mosaic records the choice in its ingest manifest as `provenance_version`, present only when the
record is on. Like the manifest's other terms it is checked on every write, and its absence means
off:

- a run with the switch on refuses a store seeded with it off, or seeded before the field existed;
- a run with the switch off refuses a store seeded with it on, so a granule list can never fall
  short of its time axis;
- a store from before the field resumes with the switch off, so existing dev stores stay usable.

The zone-year completion marker carries `provenance: true` when the switch is on, so a finished
mosaic marked without the record is refused by a run that wants it rather than embedded silently.

## The granule list

The mosaic gains a one-dimensional `granules` array along `time`, holding one JSON string per
slot. Each string is the day's granules in fusion order, one object per granule:

| field | meaning |
|---|---|
| `granule_id` | the catalogue id of the copy read |
| `image_key` | which image it is a copy of: Sentinel-2 `<datatake>/<datastrip sensing start>`, Sentinel-1 the burst id |
| `platform` | `S2A`, `S1C` and so on, from the id |
| `acquired_at` | the acquisition instant, which solar-day normalisation never overwrites |
| `cloud_cover`, `processing_baseline`, `product_uri` | Sentinel-2 only; null for radar |
| `provider`, `collection`, `stac_item_url` | where the item came from; the URL is rebuilt from the catalogue, collection and id |
| `assets` | asset key to the exact URL the loader read, after any bucket rewrite |
| `stepped_down` | this copy replaced one that would not read ([the fallback ladder](duplicate-copies.md#the-fallback-ladder)) |
| `geometry`, `bbox` | the footprint as the catalogue published it |

The list for a date is appended in the same Icechunk commit as the date's pixels, under both the
per-date and the batched write. A date that did not commit has no slot and no list, and a resumed
ingest skips dates already on the axis, so a list is written once and never touched again.

## The source index

The mosaic also gains a `source_index` band, `uint16`, one value per pixel per slot: the
granule's position in the day's fusion order plus one, or zero where no granule had data. It is
16 bits because a day's fusion order is zone-wide, and a long land zone can hold more than 255
scenes or bursts in one day. The model never reads it as a band; the inference actor reads it
beside each sensor's mask to stage its shard's share of the record
([inference README](../../src/tessera_embeddings/inference/README.md)).

odc's fuser keeps the first granule with data at each pixel and reports nothing about which one
that was. So one band per sensor — SCL for optical, VV for radar, the band each sensor's
validity is judged on — is read through a reader that packs the granule's position into the high
bits of every pixel it returns:

```text
  granule k's pixel  ──▶  (k + 1) << native bits | native bits      or 0 where the pixel is nodata
                                    │
  odc's fuser, unchanged: fills only where the destination is still 0, so the first granule
  with data wins, exactly as before
                                    │
  split after the load  ──▶  the band, identical to an unpacked load,  and  source_index
```

The packed value is zero exactly where the fuser's own nodata test says the granule has no data,
so the packed fuse keeps the same granule the plain fuse would, and the band comes back
bit-identical. The band is still read once, in its native dtype and resampling. SCL packs into
`uint32` and VV into `uint64` while it is in flight; nothing extra is added to the task graph.

The parser stamps each mask-band source with its granule's position and native metadata
(`stac.SourceParser`), and the reader packs as it reads (`stac._SourceReader`) — the same
per-source hook that removes the Sentinel-2 reflectance offset. Positions are stamped on the
items just before the load (`provenance.stamp_fusion_positions`), so the granule list built from
them afterwards (`provenance.day_granules`) numbers the granules exactly as the index does.

On the optical path the index is masked to the ROI only, like SCL: it names the granule behind
every pixel, cloudy or clear, and the model's own validity test decides later which of those it
saw. On the radar path the dB transform runs over the bands before the index is added, so the
index is never rescaled.

## What it costs

The granule list is under 100 MB a zone-year. The index band adds two bytes a pixel per slot
before compression, for as long as the mosaic exists. The dev test compares the ingest's time
per day with the record on against #250's, which should be level.
