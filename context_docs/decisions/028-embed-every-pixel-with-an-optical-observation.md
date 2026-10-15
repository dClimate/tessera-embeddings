# 028 — Embed every pixel with an optical observation, and loosen a filled store's line in place

**Status:** Accepted 2026-10-12 (Robert). Supersedes [ADR 018](018-refuse-pixels-below-minimum-optical-depth.md).
Built on `rule/optical-line-one`, stacked on the targeted updates of [ADR 027](027-targeted-updates.md).

ADR 018 refused any pixel with fewer than 15 valid Sentinel-2 observations in its year, putting the
dataset's reputation ahead of coverage. Downstream users need wall-to-wall coverage: a hole changes
the geometry of a study area and the length of a time series, which is the objection ADR 018
recorded and overruled. This reverses that ruling, and says how a store already filled at 15 — the
published v1.1 store among them — gets there without re-embedding anything it already holds.

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

**A stamped rule can be loosened, never tightened, for the whole store or one zone at a time.**
`storage/global_store.loosen_optical_min_obs`, one commit:

| call | writes | also |
|---|---|---|
| `zones=None` | the root's `optical_min_obs` | removes any zone's own rule at or above the new line, which would otherwise read as a tightening |
| `zones=[...]` | each named zone group's own `optical_min_obs` | the root is untouched; an unseeded zone is refused |

Each line given up is appended to that root's or zone's `optical_min_obs_history` as
`{"line", "until", "snapshot"}`, the snapshot being the store's tip when the line was given up. A fill
already running under the old line can still land after it, which leaves the same holes a loosen always
leaves, and its registry rows state the line it applied, so the update that follows finds them. Loosening
to the current line commits nothing; a tightening, a store with no rule and an unstamped store are refused,
each saying why. Operators use `scripts/update/loosen_optical_min_obs.py`, a dry run unless `--write`.

**The rule in force for a zone is its own attr when present, else the root's**
(`effective_optical_min_obs`), and everything that reads the rule goes through it: the per-cell and
chained fills, the campaign driver's staging fingerprint and the runner's assertion. A chained
session spans zones, so the rule rides each cell's work items to the actors beside its window and
orbit (`ZoneContext.optical_min_obs`); an actor built from another zone's config would otherwise
judge the cell against the wrong line.

**Down only**, because loosening leaves every embedded pixel compliant with the new line and makes
every newly eligible pixel identifiable in the store as `isnan(scales) & (s2_obs_count >= line)`.
Tightening would leave embedded pixels below the advertised line, with nothing to tell a reader
which. **Per zone**, because updating the whole published store is a large spend run zone by zone,
and a zone that has been updated should advertise the line its pixels were judged against while the
rest of the store keeps its own.

**The update is ADR 027's `holes` mode over `refused_thin_px > 0`.** Under a looser line every
thin-refused pixel is a hole, so `holes` infers exactly the newly eligible pixels and merges them;
an embedded pixel is never touched. Each cell's `updates[year]` record carries the
`optical_min_obs` the update applied, and each registry row its own. The steps are in
[`docs/updates.md`](../../docs/updates.md) ("Before step 1, only when lowering the depth rule"): loosen, plan with `plan_update.py from-registry --mode holes --refused thin`
over the same zones, run, and check that the updated rows say `optical_min_obs = 1` and
`refused_thin_px = 0` and that the same plan now selects nothing.

## 2. Rejected alternatives

- **`None` instead of 1.** The same pixels, and nothing to verify them by (above).
- **A new store at one.** Re-ingests everything and re-embeds the 94% of pixels ADR 018's line
  already kept (pixel-weighted over its 40 measured cells) to add the rest. `holes` pays only for
  the pixels it adds and the imagery reads of their shards.
- **Re-stamping through the seeder.** The seed's identity check exists so that a rerun cannot
  silently change what a store advertises. A separate, explicit and recorded loosen keeps that
  check strict, and `check_root_identity` now names it when it refuses.
- **Allowing either direction.** A tightened store would advertise a line some of its embedded
  pixels do not meet.

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
- **yield-embeddings follows** (Arbol-Project/yield-embeddings#96): `CAMPAIGN_OPTICAL_MIN_OBS = 1`
  with the pin bump (its test asserts equality with this constant); its validator resolves the rule
  per zone and counts thin pixels against that same rule; its runbook gains the loosen step.
- **Unchanged:** the ingest date filter `min_valid_coverage` (campaign 0.1%, updates 1e-9), and
  ADR 027's drivers and modes.

## 4. Verified on global-tessera-dev, 2026-10-12

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
