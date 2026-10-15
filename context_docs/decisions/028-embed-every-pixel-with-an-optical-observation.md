# 028 — Embed every pixel with an optical observation

**Status:** Accepted 2026-10-12 (Robert). Supersedes [ADR 018](018-refuse-pixels-below-minimum-optical-depth.md).
Built on `rule/optical-line-one`, stacked on the targeted updates of [ADR 027](027-targeted-updates.md).

ADR 018 refused any pixel with fewer than 15 valid Sentinel-2 observations in its year, putting the
dataset's reputation ahead of coverage. Downstream users need wall-to-wall coverage: a hole changes
the geometry of a study area and the length of a time series, which is the objection ADR 018
recorded and overruled. This reverses that ruling, and says how a store already filled at 15 — the
published v1.1 store among them — gets there without re-embedding anything it already holds, and
without its attrs claiming a line its other cells were never judged against.

---

## 1. Decision

**The line is one: any valid observation embeds.** `OPTICAL_MIN_OBS = 1` (`config/inference.py`).
The dataset refuses a pixel with no valid observation as `no_optical` whatever the rule
(`has_optical = s2_nonzero & (s2_valid_count > 0)` in `inference/dataset.py`), so at one the thin
refusal is empty by construction: `refused_thin` per strip, `s2_thin_px` per chunk and the
registry's `refused_thin_px` are invariants at 0, and a non-zero one means the gate leaked.

**Stated as 1, never as `None`.** `None` selects the same pixels, but it is what a store that never
had a rule carries — every store seeded before 2026-08-13 — and it gives a reader nothing to check:
the thin invariant has no line to be measured from, and a fill's assertion against the store
compares nothing. `1` is stamped on the root, asserted by every fill, and written on every registry
row and update record. Zero stays refused, as a rule that refuses nothing. This is the global store's
statement: a single-area run keeps `None`, which embeds the same pixels, because its manifest treats
an absent rule as none and would refuse the next append to every existing single-area store.

**A filled store's rule never moves; an update carries its own, looser line.** The root's
`optical_min_obs` stays write-once. An update plan may set `optical_min_obs`
(`plan_update.py --optical-min-obs 1`), and that line holds for the plan's cells only: the fill
applies it in place of the root's (`storage/update.applied_optical_min_obs`), the runner asserts
it, and every cell of the run takes the same line through the run's config. A line stricter than
the store's is refused, as is any line on a store that declares no rule.

**The store's attrs then understate an updated cell, and that is accepted** (Robert, 2026-10-15).
The root says 15 while an updated 2024 cell holds pixels with 1–14 observations. Writing the new
line onto a zone group instead would misdescribe that zone's other years, and onto the root every
zone. The truth is kept per cell and per tile: each `updates[year]` record states the line its
update applied, each registry row its own, and `storage/update.cell_optical_min_obs` returns the
loosest line any write to a cell applied, the floor every pixel embedded there meets. It is not a
promise of coverage: an update may reach only some of a cell's tiles. A later update may not apply
a stricter line to a cell than an earlier one did; the runner refuses it, since a `shards` rewrite
would drop the pixels the earlier update embedded and a `holes` merge would count them as thin.

**Looser only**, because a looser line leaves every embedded pixel compliant and makes every newly
eligible pixel identifiable in the store as `isnan(scales) & (s2_obs_count >= line)`. A stricter one
would leave embedded pixels below the line their cell's records state, with nothing to tell a reader
which.

**The update is ADR 027's `holes` mode over `refused_thin_px > 0`.** Under a looser line every
thin-refused pixel is a hole, so `holes` infers exactly the newly eligible pixels and merges them;
an embedded pixel is never touched. The steps are in [`docs/updates.md`](../../docs/updates.md):
plan with `plan_update.py from-registry --mode holes --refused thin --optical-min-obs 1`, run, and
check that the updated rows say `optical_min_obs = 1` and `refused_thin_px = 0` and that the same
plan now selects nothing.

## 2. Rejected alternatives

- **`None` instead of 1.** The same pixels, and nothing to verify them by (above).
- **A new store at one.** Re-ingests everything and re-embeds the 94% of pixels ADR 018's line
  already kept (pixel-weighted over its 40 measured cells) to add the rest. `holes` pays only for
  the pixels it adds and the imagery reads of their shards.
- **Re-stamping through the seeder.** The seed's identity check exists so that a rerun cannot
  silently change what a store advertises.
- **Loosening the rule in place, for the store or per zone.** Built and verified on dev (§4) as
  `loosen_optical_min_obs`, which wrote the new line and a history of old ones onto the root or
  named zone groups, then withdrawn before merging (Robert, 2026-10-15). A zone attr spans every
  year of its zone, so loosening it for a 2024–2025 update misdescribes the zone's 2017–2023, and a
  root loosen misdescribes every zone. Leaving the attrs alone misdescribes only the updated cells,
  whose own records are exact.
- **A stricter update line.** It would leave embedded pixels below the line their records state.

## 3. Consequences

- **Both code identities move.** The constant lives in `config/inference.py`, which is in both
  closures. `inference_code_identity` moves, so staging under the old identity is not resumed (it
  moves with ADR 027 as well). `ingest_code_identity` is checked only when a run appends to a mosaic
  (`config/ingest.py`), never on a finished one, and every update ingests fresh mosaics under its
  own prefix; the only exposure is a campaign mosaic part-way through ingest at deploy time, whose
  next append would be refused. **Deploy at a boundary with no ingest in flight.** There is no way
  to move the line without this: the constant lives in that file.
- **Thin embeddings are less reproducible, knowingly.** Below 25 observations, two independent
  embeddings of the same ground disagree at two to four times the pipeline's own noise floor
  ([`minimum-optical-depth.md`](../inference/minimum-optical-depth.md) §4). `s2_obs_count` is
  published per pixel, so a consumer who needs a stricter line applies it when reading.
- **ADR 018's deferral premium is now paid.** It recorded refusal as a deferred spend that costs
  more to recover than to embed once. `plan_update.py` prints each cell's shard counts; price them
  against [`campaign-cost-model.md`](../campaign/campaign-cost-model.md) and the holes-mode
  measurement (ADR 027 §3) before a large update, rather than estimating.
- **A tile no update has reached reflects the earlier line.** Its registry row still says
  `optical_min_obs = 15` with its `refused_thin_px`, which is exactly the update's work list.
- **The attrs understate updated cells.** A reader judging pixels by the root's `optical_min_obs`
  finds pixels below it in an updated cell; `cell_optical_min_obs`, or the cell's update records,
  give the right line ([`reading-the-published-store.md`](../storage/reading-the-published-store.md)).
- **yield-embeddings follows** (Arbol-Project/yield-embeddings#96): `CAMPAIGN_OPTICAL_MIN_OBS = 1`
  with the pin bump (its test asserts equality with this constant); its validator grades each cell,
  and counts its thin pixels, against `cell_optical_min_obs`; its runbook plans updates with the line.
- **Unchanged:** the ingest date filter `min_valid_coverage` (campaign 0.1%, updates 1e-9), and
  ADR 027's drivers and modes.

## 4. Verified on global-tessera-dev

**2026-10-15, as built** ([`stack-dev-verification.md`](../campaign/stack-dev-verification.md)): a
`holes` update planned with `--optical-min-obs 1`, against a store seeded at 15, on a 17S/2024 tile
the store never wrote because the line of 15 refused all of it. It embedded every one of its
4,134,860 pixels with an optical observation, left its 59,444 with none empty, did not touch the
neighbouring tile, and left the root at 15 with the update record stating 1.

**2026-10-12**, the run below, used the in-place loosen that §2 records as withdrawn: 18N's zone
attr was loosened to 1 and 18S kept 15, so one run applied two lines. What carries over is the
`holes` update at a line of one: it filled every eligible hole, changed no embedded pixel, and its
embeddings look like the rest.

Two cloudy tiles the published store refuses about 60% of under 15, both fully land with radar,
year 2023: 18N `chunk_448_1` (Colombia–Ecuador, 1.3°N) and 18S `chunk_28_1` (northern Peru, 5.2°S).
A fresh dev store seeded at 15 was filled through `fill-zones-sequential` on the campaign path; its
registry rows reproduced the published `refused_thin_px` exactly (2,505,700 and 2,514,335). Only 18N
was then loosened to 1, and one `holes` update over both tiles ran on one cluster. Checked against
the update's base tag:

| zone | rule applied | embedded before | after | newly embedded below the zone's rule | baseline pixels changed |
|---|---:|---:|---:|---:|---|
| 18N | 1 | 1,684,989 | 4,190,689 | 0 | none, in all eight arrays |
| 18S | 15 | 1,679,709 | 1,679,709 | 0 | none (tile kept) |

- **Every 18N hole with an optical observation was filled**; the 3,615 left empty have none.
- **One chained session applied two rules**: the run logged 1 for 18N and 15 for 18S, the update
  records state the same, and the registry rows read `optical_min_obs` 1 with `refused_thin_px` 0
  for 18N and 15 with 2,514,335 for 18S.
- **The new embeddings look like the rest**: in a sampled 256-px window, pixels with 3–14
  observations have a median L2 norm of 9.63 against 9.96 for those with 15–25, and no zero vectors.
- Baseline and update each took about 30 minutes end to end. Everything the test wrote was deleted
  afterwards.
