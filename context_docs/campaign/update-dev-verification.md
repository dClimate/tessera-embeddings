# Targeted updates: dev verification (2026-10-12)

The end-to-end check of [ADR 027](../decisions/027-targeted-updates.md) on global-tessera-dev,
before the update code reached a store anyone reads. **Both modes did exactly what the ADR says, on
both drivers, and found one defect**, in the registry rows of merged shards, fixed in the same PR.

## Setup

Two cloudy equatorial blocks, picked from the published registry for a mix of embedded pixels and
thin refusals just below the line of 15 (`plan_update.py`-style query, read-only):

| cell | where | tiles | thin-refused share | median depth of the thin pixels |
|---|---|---|---|---|
| 18S/2021 | Peruvian Amazon, -77.7°, -5.8° | `chunk_30_2`, `_30_3`, `_31_2`, `_31_3` | 60% | 10 |
| 21N/2021 | Guyana, -58.6°, 3.3° | `chunk_436_8`, `_436_9`, `_437_8`, `_437_9` | 60% | 12 |

- **Masks**, built by `build-land-mask` from `registry.txt` subsets of the 0.1° cells lying wholly
  inside the blocks: `update-dev-base` (the eight tiles) and `update-dev-alt` (18S's four, plus
  `chunk_30_4` and `chunk_31_4`, which the base never wrote).
- **Store:** a fresh `update-dev`, seeded at `optical_min_obs=15`.
- **Baseline:** both cells filled through `run-global-campaign` (two clusters) at the campaign's
  `min_valid_coverage` of 0.1.
- **Code:** this PR at `911bcbbb`, deployed from the YE branch `dev/global-tessera-update` with its
  own baked AMI.
- **Isolation:** each run used its own Prefect ingest limit and no pause gate, so no shared Prefect
  control was touched.

## Results

**Holes update, multi-cluster** (`run-global-campaign`, plan from the dev registry's `master/` with
`where="refused_thin_px > 0"`). Checked with an Icechunk diff from `update-update-dev-holes-base` to
the tip:

| check | result |
|---|---|
| changed shards | the 8 targets, all 8 arrays, year 2021 only; 0 outside |
| `runs`, `years_complete`, zone-year tags | unchanged |
| update record | one per cell, `written` = the 4 tiles |
| pixels embedded at baseline | byte-identical in all 8 arrays, every tile |
| holes filled | 4 to 1,716 per tile; no hole pixel's observation count fell |
| registry `master/` | every row names `update-dev-holes`; `verify` passes |
| a second dispatch | 0 cells of work, branch tip unchanged |
| `update-update-dev-holes-complete` | pinned |

After the update the dev store's `refused_thin_px` and `refused_no_optical_px` equal the published
registry's on all eight tiles. The baseline was thinner than the published store by exactly the
pixels the update filled: on a four-tile ROI the 0.1% date threshold dropped cloudy dates that the
zone-wide published run kept, and the update's `1e-9` floor restored them.

**Shards update, one cluster, alternative mask** (`fill-zones-sequential`, plan from a CSV of two
baseline tiles and the two never-written ones):

| check | result |
|---|---|
| changed shards | the 4 targets only |
| never-written tiles | `chunk_30_4` 0 → 1,718,992 embedded pixels, `chunk_31_4` 0 → 2,471,820 |
| rewritten tiles | the same embedded pixel sets as before |
| `runs`, `years_complete` | unchanged |

**Holes inference equals full inference, per pixel.** On the pixels the holes update filled, the
shards update's whole-tile run agrees within the
[ADR 012](../decisions/012-validated-equivalence-for-inference-outputs.md) cross-config envelope:
99.97–99.99% within one level, at most 2 levels, scale drift ≤ 1.2%, worst cosine 0.99990. Over
every pixel whose inputs were identical in both runs it is 100.000% within one level, worst cosine
0.99990. The pixels that differ by more (976 in `chunk_30_2`, 9,937 in `chunk_31_3`) are exactly
those whose observation counts changed, because the shards run kept dates the baseline's threshold
had dropped.

## Cost and capacity

| per tile, warm actor | baseline | holes | shards |
|---|---|---|---|
| total | 52–169 s | 28–41 s | 113–298 s |
| of which imagery reads (prologue) | 19–28 s | 18–26 s | 20–26 s |
| GPU inference | 32–141 s | 0–3 s | 89–273 s |

These holes filled few pixels, so the holes column is the floor: reading and writing a tile, about
30–40 s, whatever the tile's depth. The first tile on each actor carries about 110 s of one-off
warm-up in every mode. The shards column swings with the card it ran on (the paired-arms card
confound); the baseline and shards runs embedded the same pixels.

**Assembly.** A merged shard costs each worker about 3 s more, to read the shard back (12.3–12.5 s
of `pixel_merge_s` over 4 shards), and assembly took 15.7–17.8 s against 14.7–15.3 s for the
baseline. The runner peaked at 5.1–5.7 GiB against 1.2–2.2 GiB, about 0.9 GiB per merging worker,
on a 32-vCPU, 244 GiB task; at 32 workers that is roughly 28 GiB on top of a fill's measured ~94
GiB. CPU stayed under 2 vCPU.

## The defect, and the fix

The first holes run published registry rows with every refusal count zero and the optical columns
null. A merged shard's measurements are re-derived from the store, which returns registry columns,
but the part writer reads the actors' coverage-record shape; nothing raised, and the rows were
wrong. `registry_rebuild.as_coverage_record` now converts them, a unit test pins the round trip,
and the two cells' parts were re-published through the fixed code path before the `verify` above.

## Teardown

Removed in-region: the store, its registry and staging under
`s3://global-tessera-embeddings-dev/global-tessera-update/`, `updates/update-dev-*`, the two masks
and their registry subsets. The shared `rois/zarrs/zone_21N.zarr`, which the baseline rewrote, was
restored from its backup; `zone_18S.zarr` had since been rewritten by another session's run and was
left as that session left it. The YE branch and its deployments were removed.
