# 034 — v2 stores use the geoembeddings `utm_zones` layout

**Status:** Proposed 2026-10-27. Follows the pattern of
[ADR 032](032-v2-stores-name-their-axes-y-and-x.md): one more per-model field, and a store keeps
what it was created with. The campaign fills each column as two halves:
[ADR 036](036-the-campaign-fills-a-utm-column-in-two-halves.md).

A global store has so far been 120 zone groups, `01N` to `60S`, each in its hemisphere's own EPSG
code. Other global embedding datasets use a different layout: the Source Coop copy of our v1.1
store, AlphaEarth and OlmoEarth. They have 60 groups, `utm01` to `utm60`. Each group covers a
whole six-degree column in its northern EPSG code, with the southern hemisphere as negative
northings. That is the geoembeddings convention's `utm_zones` layout. From v2 on, our stores use
it too.

---

## 1. Where the hemisphere groups came from

[ADR 011](011-campaign-zone-ingestion.md) named zones by their UTM common name because a `utm33`
group name "cannot express the hemisphere". That holds only if every group uses its hemisphere's
own CRS. The convention instead puts both hemispheres in one CRS. EPSG:326xx is defined south of
the equator, where northings are negative, and the convention's own TESSERA example does exactly
this. So a stated limit of the convention was in fact a choice of CRS. Stores in our layout also
could not declare `geoemb:spatial_layout` (the convention's value for how a store is split into
groups), and a reader moving between our store and the others had to translate group names.

## 2. Decision

**The model chooses a new store's zone layout.** `ModelFamily.zone_layout`
(`config/inference.py`) is `"hemispheres"` for v1.1 and `"utm_zones"` for v2 Large. The seed flow
applies it to a new store. **An existing store keeps its layout**: the layout is read from the
names of the groups it already holds (`zone_grid.zone_layout_of`), so a v2 seed into an existing
hemisphere store completes that store's 120 groups. `seed_zone_groups` refuses to put both
layouts in one store, because the two would cover the world twice.

**A zone's name says which layout it belongs to.** `utm33` and `33N` never collide, so
`zone_grid.zone(name)` resolves either one, and mosaic paths, tags, staging run ids, mask groups
and flow parameters all work unchanged. Only questions about every zone take a layout:
`zone_grid.zones(layout)` for the groups seeding creates, and `zone_grid.fill_zones(layout)` for
what the land mask, the ROI export and the campaign's default scope cover (ADR 036). That scope is
every fill zone of the layout the store's groups use. `canonicalize_zone` accepts `utm33`, `UTM7` and the old forms alike.

**The grid of a `utm` column** comes from the same pyproj derivation as before, taking the union
of both hemispheres' areas of use projected into the northern CRS and snapping outward to the
2048-pixel shard pitch. Every column has eastings from 163,840 to 839,680 m and northings from
−8,888,320 to 9,338,880 m, which is 890 by 33 shards. The northern half is the `N` zone's grid
exactly. The southern half shares every pixel with the `S` zone, because 10,000 km of false
northing is a whole number of pixels, but the two disagree on shard boundaries, because it is not
a whole number of shards.

**A `utm_zones` store declares itself.** The seed writes `geoemb:spatial_layout: "utm_zones"` on
its root, which is now true. Hemisphere stores still omit it.

## 3. Rejected alternatives

- **Keep 120 groups and rename them** (`utm33n`, `utm33s`). This is a cheap rename that matches
  nothing: readers following the convention would still not find `utm33`.
- **Keep v1.1's hemisphere zones as the work unit and map them into columns when writing.** This
  would reuse v1.1's southern mosaics and mask, but the `S` zone's shards do not line up with the
  column's. Every southern shard would straddle two column shards, and two writers would contend
  for one shard. (ADR 036's halves are cut from the column's own grid, so they do not.)
- **Copy Source Coop's extents exactly.** That copy pads each column to a 4,096-pixel pitch for
  its own chunking. Our grid snaps to our own shard pitch, as it always has. Only the group names
  and the CRS are what make the layout match.

## 4. Consequences

- **A `utm_zones` store needs its own mask and its own mosaics.** Run `build_land_mask` with
  `zone_layout="utm_zones"` into a new mask name, then `export_zone_rois` with the same layout.
  The build is pure geometry over the registry and takes under a minute. Mosaics are keyed by
  fill zone, so a v2 ingest writes `utm33n` and `utm33s` mosaics and never touches v1.1's `33N`
  and `33S`.
- **A zone-year of the store is a whole column**, roughly the north and south zone-years combined.
  The campaign fills it as two halves (ADR 036), so it still has 120 units per year.
- **Southern imagery is warped into the northern CRS.** Sentinel-2 tiles south of the equator
  arrive in EPSG:327xx. Their pixel grid coincides with the column's, so the warp is a pure shift,
  but it is a warp rather than a copy. Neither its output nor its cost has been measured yet; a
  v2 development run should check both.
- **Callers outside this repository must use the new names for a v2 store**: deployment
  parameters and any zone lists the orchestration repository passes.
- **v1.1 is unchanged.** The published store, its updates, its registry, and the scripts that
  census it all stay on `01N` to `60S`.
