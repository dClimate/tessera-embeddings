# 032 — v2 stores name their spatial axes `y` and `x`

**Status:** Proposed 2026-10-22. Built on [PR #247](https://github.com/dClimate/tessera-embeddings/pull/247)'s
one record per model, and on [PR #218](https://github.com/dClimate/tessera-embeddings/pull/218)'s
convention work.

Every output store so far names its spatial dimensions `northing` and `easting`. No standard asks
for that, and the tools downstream users reach for look for `y` and `x`. This makes the names a
property of the model: v2 stores use `y`/`x`, and v1.1 keeps `northing`/`easting` because its
published store and every v1.1 store a run appends to already use them.

---

## 1. Where `northing`/`easting` came from

A local choice, not a convention. The code arrived from the yield-modeling repository in this
repository's first commit (`9e182507`, 2026-05-15). There, the first version of the convention
attributes wrote `spatial:dimensions: ["y", "x"]` (`e4a5016`, 2026-03-18). A week later one commit
renamed every `y`/`x` to `northing`/`easting` across 33 files, with no reason given in the message
or any document (`476d185`, "easting and northing + linting").

"Easting" and "Northing" are the EPSG and ISO 19111 names for a projected CRS's *axes*, and they
appear inside CRS definitions. None of the conventions the stores follow uses them as dimension
names:

- **CF** identifies projected axes by `standard_name` and `axis`; its own examples name the dimensions `x` and `y`.
- **The `spatial` convention**'s examples use `["y", "x"]`, and so does the geoembeddings convention's TESSERA example.
- **rioxarray** recognises `x`/`y` (or `longitude`/`latitude`) by name, and otherwise only through CF `axis` attributes.
- **odc-geo** recognises `y`/`x`, `latitude`/`longitude` and `lat`/`lon`, and otherwise guesses.

## 2. Decision

**The model names a new store's axes.** `ModelFamily.spatial_dims` (`config/inference.py`) is
`("northing", "easting")` for v1.1 and `("y", "x")` for v2 Large, in `spatial:dimensions` order.
Seeding and single-area assembly apply it with `StoreLayout.with_spatial_dims`. The presets in
`config/store_layout.py` keep their v1.1 spelling.

**A store is named once.** A new store takes its model's names. An existing store keeps its own,
whatever the model writing to it, so appending to it or seeding more zones into it never mixes two
vocabularies. `spatial_dims_of` reads a store's names from its `spatial:dimensions`, or from its
required `scales` array, `(time, Y, X)`, when a store with pixel-index coordinates declares none.

**Code that reads an output store asks the store.** The append grid check, the zone-fill grid
check, the seed flow's completeness check, the published-store audit and `resolve_region` all take
the names from `spatial_dims_of`, never from a constant; `resolve_region` keeps its `northing` and
`easting` range arguments and returns slices under the store's names. The CF attributes (PR #218)
are keyed by axis, not by name.

**A store without projected coordinates names no CRS.** Without a mosaic, single-area assembly
writes pixel indices. rioxarray finds `y`/`x` by name, so a CRS beside those indices would place
the raster at that CRS's origin; such a store gets no `proj:` attributes.

**Output stores only.** Mosaics, staged tiles, `ZoneSpec`'s extents and `SpatialCoords` keep
`northing`/`easting`. Both models share them, nothing outside the pipeline reads them, and
renaming them would mean reading every existing mosaic under either name.

## 3. Rejected alternatives

- **Rename everything, v1.1 included.** It breaks every reader of the published v1.1 store
  (`ds.isel(northing=...)`, including this repository's own examples), and a v1.1 append would
  have to rename an existing store.
- **Rename everything internally and map v1.1 back at the writer.** One vocabulary in the code,
  but every existing mosaic would have to be read under both names. The ingest fingerprint would
  also move, refusing appends to partly ingested mosaics, for no change a user can see.
- **Name stores by the store rather than the model** (for example, a seeding parameter). The model
  is already the one thing a store is created for, and #247 put every other per-model difference
  on its record.

## 4. Consequences

- **Two vocabularies inside the code.** Inputs and internals say `northing`/`easting`; output stores
  say whatever they were created with. The boundary is seeding and assembly.
- **v2 stores created before this change keep `northing`/`easting`**, under the "named once" rule.
  These are the dev stores used for v2 testing.
- **The band-first v2 layout of [ADR 031](031-the-v2-store-chunks-the-band-axis-at-16.md) is
  `(time, band, y, x)`.** v1.1's band-last order keeps its names, `(time, northing, easting, band)`.
- **Readers find both.** rioxarray finds a v2 store's axes by name and a v1.1 store's through the CF
  `axis` attributes. Both pass the strict schema audit in
  [conforming to the Zarr conventions](../storage/conforming-to-the-zarr-conventions.md).
- **The ingest fingerprint moves.** `config/inference.py` and `config/store_layout.py` sit inside
  it, but #209 and #247 already move it, so this ships in the same deploy, between campaigns.
