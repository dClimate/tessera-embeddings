# 036 — The campaign fills a `utm` column in two halves

**Status:** Proposed 2026-10-28. Builds on [ADR 034](034-v2-stores-use-the-utm-zones-layout.md),
which made a v2 store 60 full-column groups, `utm01` to `utm60`.

ADR 034 changed the store's groups, and with them the campaign's unit of work: one column-year is
roughly a v1.1 northern zone-year and a southern one together. The campaign hands zones to
clusters densest-first, so 60 coarser units per year balance the clusters less evenly than v1.1's
120. This decision keeps the store's layout and splits the *work* back into halves: the campaign
fills `utm33` as `utm33n` and `utm33s`, which are about the size of v1.1's `33N` and `33S`.

---

## 1. The equator is a shard boundary

A column's northings run from −8,888,320 m to +9,338,880 m. The top is 456 shards of 20,480 m
above the equator, and the bottom is 434 shards below it, so northing 0 falls exactly between
shard rows 455 and 456:

```
utm33 (EPSG:32633), shard rows 0..889
row   0  ┬ northing +9,338,880
         │  utm33n owns rows 0..455
row 455  ┴
──────────── northing 0 (equator) ── a shard boundary
row 456  ┬
         │  utm33s owns rows 456..889
row 889  ┴ northing −8,888,320
```

Each half therefore writes a set of shards the other never touches. This is not the alternative
ADR 034 rejected: that one kept v1.1's southern grid, which is 10,000 km of false northing away,
a whole number of pixels but not of shards, so its shards straddled the column's. Both halves here
use the column's own grid.

## 2. Decision

**Fill zones are named, and one function maps them to the store.** `zone_grid.zone("utm33s")` is
a spec with the column's full grid, `name` `utm33s`, `group_name` `utm33`, and `tile_rows`
(456, 890). `zones(layout)` still lists the store's groups (what seeding creates);
`fill_zones(layout)` lists what the campaign fills: each column's two halves, or in the
`hemispheres` layout the groups themselves. `group_and_half(name)` is the single mapping from a
fill zone to what it addresses in the store, `("utm33", "s")`, and leaves every other name whole.
`canonicalize_zone` accepts `UTM33S`.

**Everything keyed by the work uses the half's name.** That is the land mask's groups, the ingest
ROIs and mosaic folders, staging run ids, flow parameters and the zone-year tags
(`zone-utm33s-2024`). **Everything that addresses the store uses the group**: the fill runner, the
update path and the campaign's seeded checks all go through `group_and_half`.

**The mask has one group per half, built from the whole column.** `build_land_mask` with
`zone_layout="utm_zones"` builds each half from all of its column's registry cells, so a cell's
footprint that crosses the equator still sets its bits, then keeps only the bits on its own rows.
The two halves are exactly the column's coverage, cut at a shard row, and share no shard. Each
counts only the cells centred on its side, so a half with no land of its own validates cleanly.

**The group's attributes record each half, and the year only when both are in.** `commit_year_attrs`
is still the one writer, and each half still inserts only its own keys:

```
utm33.attrs
  halves_complete  {"2024": ["n", "s"], "2025": ["n"]}     which halves have landed each year
  years_complete   [2024]                                  a year enters with its second half
  runs             {"2024": {"n": {run_id, commit, ...},   one record per half, each with
                             "s": {run_id, empty, ...}}}   its own `previous` history
  updates          {"2024": [{update, half: "s", ...}]}    a half's update names the half
```

`years_complete` keeps its meaning, "the whole group has this year", so every reader of it is
unchanged: the published-store checks, the year-complete tag, and `cell_optical_min_obs`.
`shard_writer.cell_complete(node, year, half)` answers for one half, and `year_runs` reads
either shape of `runs[year]`. The shard commit's `shard_write` metadata adds `half`.

**The campaign's status reads halves.** `CampaignStatus.campaign_zones()` is every fill zone of
the store's layout; `has("utm33s", 2024)` reads the half's mark. The year-complete tag still
requires every unit, which for a column store means all 120 halves.

**The session catch-up is narrowed to the shards being written.** A fill keeps its session current
while its workers write, and refuses to skip past any commit that touched its own group (see
[writing-to-the-global-store.md §3](../storage/writing-to-the-global-store.md)). With two halves
filling one group at once, the other half's shard commit, or its attrs-only "empty" mark, would
block the catch-up for the rest of the fill. That brings back the deep rebase walk the catch-up
exists to prevent. The shard writer now passes the `(time, row, col)` shards it writes. A commit
inside the group is then skipped only if it touched none of those shards and changed nothing in the
group but its attributes. Those are exactly the commits Icechunk's own conflict check accepts.
Structural changes and coordinate-array chunks still block, and the commit's rebase is untouched.

## 3. Rejected alternatives

- **Fill whole columns.** This is what ADR 034 shipped. It gives 60 units of about twice the size,
  and the campaign's cluster balance is set over those units.
- **Separate groups per half.** This would bring back 120 groups, which the `utm_zones` layout
  exists to avoid; readers following the convention look for `utm33`.
- **One mask group per column, filtered to a half's rows when read.** This has the same effect,
  but every mask reader (the ingest, the ROI export, the fill, the campaign's sizing) opens a
  group by name. With a group per half they all stay unchanged.

## 4. Consequences

- **120 fill units per year, as in v1.1**, each on the column's grid. The two halves of a column
  routinely run at once on different clusters.
- **Fills sharing a group are safe together**, whether two years of a zone or two halves of a
  column. Their shards are disjoint, and their attrs commits retry. Only the same cell twice at once
  is unsafe, and the work list's deduplication already prevents that.
- **The registry stays keyed by group.** Parts are `zone=utm33`, and each run's file is named by
  its own run id, so the halves never overwrite each other. The master is merged per group-year by
  whichever half lands. If one half merges while the other's commit has landed but its part has not
  yet been published, the merge logs a disagreement and keeps the previous master. The second
  half's own merge then brings the master current.
- **A mosaic still spans the column's grid**, because the fill validates it against the group. Only
  the half's live windows are ingested, as with any mask.
- **v1.1 is unchanged.** In the `hemispheres` layout a fill zone is its group, `group_and_half`
  returns the name whole, and no group carries `halves_complete`.
- **Not yet run anywhere.** The unit tests fill both halves of a miniature column end to end and
  exercise the catch-up against a real repository. A v2 dev run straddling the equator should
  confirm the split, the master merge and the catch-up at scale, alongside the 327xx → 326xx warp
  that ADR 034 left to measure.
