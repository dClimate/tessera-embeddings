# Minimum optical depth: refuse pixels below 30 observations — implementation plan

**Status:** approved, not yet built. Handoff document for the implementing model.
**Decision date:** 2026-08-13. **Evidence:** [`optical_depth_census_2026_08.md`](optical_depth_census_2026_08.md).

---

## 1. The rule

**A pixel is not embedded if it has fewer than 30 valid Sentinel-2 L2A observations in the
calendar year being filled.** The refused pixel is written as fill, exactly as an
out-of-ROI pixel is, and the reason is recorded.

Three decisions were taken with the rule and are **not open** for the implementer to
revisit:

| decision | ruling |
|---|---|
| **2017**, where 65.7% of shard-years fall below 30 | **Same rule, populate it anyway, warn about it.** No per-year threshold — one rule keeps the product consistent. 2017 publishes roughly a third populated, and that must be stated where a user meets it (§7.1), not left for them to discover. |
| **Temporal consistency** — a pixel can clear 30 in 2020 and fail in 2021 | **Per pixel, per year.** Holes move between years. No all-years gate, no stability mask in this iteration. |
| **Scope** | **Global campaign only.** The single-ROI path (`tessera_embeddings` flow, `plain` runner) keeps today's behaviour. |

### Why 30, and why it is safe to apply per pixel

The repo's own quality measurement is *crisp at 30+ observations, noisy below about 20*,
across 15 cells. 30 is the conservative end of that band.

> **That measurement is being re-taken, and the threshold should not ship before it lands
> (raised and accepted 2026-08-13).** Two defects were found in the instrument that produced it,
> both on the same day, and this rule is the one decision in the campaign that cannot be revisited
> later — a refused pixel does not exist, so recovery is a re-run rather than a filter change.
>
> 1. **The figures were drawn with a per-window contrast stretch.** Each window was stretched to its
>    own 2nd–98th percentile, so a window whose ground varies little had its noise floor expanded to
>    full contrast *whatever its depth*. Some of the "noise below 20" was the renderer. The stretch
>    is now shared across a cell's windows.
> 2. **Depth was read per CELL, not per window.** A zone spans most of a hemisphere and depth falls
>    with latitude and with cloud, so a cell mean describes a mixed population. Measured: a cell of
>    mean 46.9 holds windows at 34.5 and 61.0, and its *thinnest* window is the one with the most
>    structure. Each window now records its own mean and tenth percentile.
>
> The re-measurement is a blind one — 73 windows from 9 cells spanning **13 to 98** observations per
> pixel, with 18 of them below the 30 line, reviewed for whether ground features are nameable, with
> depth withheld from the reviewer and joined afterwards. It also carries a confound it cannot
> remove and must therefore report: **a uniform biome is illegible at any depth**, so desert,
> unbroken canopy and open water read like thin data. One pair already visible in the corpus makes
> the point — at 13 observations, one window is the most structured in its cell while three of its
> neighbours at the same depth have almost no structure at all.
>
> **What the re-measurement can change:** the value, or the decision to refuse at all. What it
> cannot change is the shape of everything else here, which is why the rest of this plan is worth
> building either way — the registry, the per-shard record and the cycle model are all wanted even
> if the line moves or becomes a label again.

The census that produced the campaign figures measured **shard means**, and this rule
refuses **individual pixels** — a different question, and the class of error the
corrections register calls *presence counted where coverage was meant*. It was therefore
measured directly against `s2_obs_count` as written, over **844 chunks and 55.3M populated
pixels in 15 zones**:

| line | % of PIXELS below | % of CHUNKS whose mean is below | ratio |
|---|---:|---:|---:|
| 15 | 2.27% | 2.25% | 1.01 |
| 20 | 3.73% | 3.79% | 0.98 |
| **30** | **10.90%** | **11.37%** | **0.96** |
| 40 | 21.66% | 21.45% | 1.01 |

The two views agree to within 4%, so the census's shard-level figures carry over. (The
absolute level in that sample is below the census's global 18.4% because the sampled zones
are dominated by dry 37N/38N; the **ratio** is the transferable result, not the level.)

The loss is also concentrated rather than smeared — at the 30 line, **90.4% of refused
pixels sit in chunks whose own mean is already below 30**, and chunks averaging 45+
contribute 0.1%. This is what makes a per-shard registry an honest summary: refusals
cluster, so a shard-level number is not hiding a diffuse scatter.

---

## 2. Non-goals — do not build these

**No ingest-time skipping.** The only signal available before reading a byte is the
catalogue's **acquisition-date count**, which upper-bounds the observation count and is
therefore a sound, zero-false-positive skip test. It was measured and it does not pay:

| year | provably skippable pre-ingest | actually refused after masking |
|---|---:|---:|
| **2017** | **37.15%** | 65.69% |
| 2018 | 0.34% | 14.06% |
| 2019–2024 | 0.07–0.23% | 11.1–16.8% |
| 2025 | 0.07% | 7.58% |

Overall the bound catches **23% of refusals, and almost all of it is 2017**. Three further
reasons it is the wrong trade: the ingest unit is a **4096-px chunk**, so an entire 40 km
block would have to fail rather than a location; skipping ingest destroys the audit record,
because `s2_obs_count` is derived from the mask that was not built; and the mosaic is shared
with the radar legs. Changing the most fragile path in the system for a fifth of a percent
outside 2017 is not worth it.

**Also out of scope:** any change to the land mask; any change to the single-ROI path;
re-running dev cells already filled under the old rule; a per-pixel "years cleared" stability
mask (considered, deferred).

---

## 3. What already exists and must be built on, not reinvented

| thing | where | note |
|---|---|---|
| the per-pixel gate | `inference/dataset.py` ~L123, `valid_mask` | the enforcement point is one conjunct |
| per-pixel S2 depth | `s2_obs_count` array in every zone group | **written for every pixel from the mask bundle, independent of the gate** (`actors.py` ~L1229) — so a refused pixel's depth is still recorded |
| the embedded mask | `scales`, NaN except where embedded | `~isnan(scales)` IS the per-pixel embedded mask |
| per-chunk counters | `actors.py` ~L1328–1341, emitted on `CHUNK_SUMMARY` | `s1_free_px` / `s1_thin_px` / `s2_thin_px` — the pattern to copy |
| per-year roll-ups | `assembly.summarise_radar_coverage`, `assembly.summarise_optical_skips` | already answer "which live tiles published as fill" |
| the year record | `storage/shard_writer.run_provenance` | the schema's one owner |
| the resume trap | documented in `summarise_optical_skips` | a resumed leg reports synthetic successes **with no counters** — see §6 |
| skip markers | `assembly.ZarrWriter.write_skip_marker` | currently **zero-byte**; resume reads only the object NAME, so a payload can be added safely |

**One inference chunk is one shard** (`SHARD_PX = 2048` for both), so per-chunk accounting
in the actor *is* per-shard accounting. Nothing needs to be re-read to build the registry.

---

## 4. Work item 1 — the threshold, recorded once and enforced

**One constant, carrying one value — and RENAME it. `OPTICAL_THIN_MAX_OBS` becomes
`OPTICAL_MIN_OBS`, holding 30.** No second constant: two names holding the same number would be
two things to keep in step for no gain (settled 2026-08-13, against an earlier suggestion of a
separate refusal constant). The comparison itself barely changes; what changes is that failing it
**refuses** the pixel instead of labelling it.

**The rename is the safety mechanism, and it is the whole reason not to keep the old name.** The
constant has about twenty consumers across two repositories, and the plan's own §9 has to reconcile
four documents plus two now-false statements in the validator. Keeping the name means every one of
those consumers silently changes meaning, and the ones that are prose rather than logic change
meaning without changing behaviour — the worst case, because nothing fails. Renaming makes every
consumer an import error, so the compiler produces the reconciliation list instead of a grep.
Two further reasons it earns its keep:

- **`RADAR_THIN_MAX_OBS` stays a reporting line.** After this change, two identically-shaped
  `*_THIN_MAX_OBS` constants would mean opposite things side by side in one config module.
- **A stored verdict is a permanent record.** Verdicts already written carry `s2_thin_*` fields
  computed under the labelling meaning. A new name upstream is what stops a later reader comparing
  two verdicts that used one key for two quantities.

> **Rewrite the docstring in the same edit.** It currently says, in terms, *"Only a REPORTING line:
> nothing refuses a cell for being under it, and the exact per-pixel counts live in the store's
> `s2_obs_count` array either way."* That sentence becomes false the moment this ships. The new one
> must state that the line is a refusal, that the value has been 40, then 15, then 30, and that a
> reader of older commits or documents will find the opposite claim under the old name.

### Where it lives

**Root group attrs of the global store, campaign-wide** — one value for the whole product,
visible to anyone who opens the store. `storage/global_store.py::_root_attrs`.

That location is not merely convenient: the root already has a **write-once identity check**
(`global_store.py` ~L297) whose comment explains the exact hazard — *"an incremental seed must
NOT silently re-stamp it … that would let already-seeded/filled zones (encoder A) be mixed
with a new one (encoder B) under a root now advertising B"*. Adding the threshold to that
`identity` tuple gets the guarantee for free:

- first seed stamps it;
- a matching reseed is a no-op;
- **a reseed with a different value is rejected**, with the existing error.

Fills already read the root attrs for the model gate, so `fill_zone_year` should read the
threshold the same way and refuse a cell whose config disagrees with the store it is writing
into. That closes the mixing hazard with machinery that already exists rather than a new one.

**Two consequences to rule on explicitly, because both are silent otherwise.**

**A store whose root carries no threshold at all.** That is the state of every store seeded so far,
including the dev store the campaign is tested against — the attribute did not exist when they were
seeded. The ruling: **a fill against a root with no threshold refuses to run, and names the
remediation.** The alternative, running unrefused, would publish data under a rule the store never
declared and could not later be told apart from data that legitimately predates the rule. The
remediation is a one-line re-stamp of the root, which is safe precisely because the identity check
makes a *conflicting* re-stamp impossible; add a `--stamp-threshold` path to the seeder for it. Dev
stores get re-stamped once, deliberately, rather than being silently grandfathered.

**The line can never change for a store that already has one.** Putting the threshold in the
write-once identity tuple is what buys the mixing guarantee, and it forecloses ever re-stamping a
different value — so a top-up cycle inherits the original line, permanently. That is the intended
behaviour and it is consistent with §12's "they clear the unchanged 30 on their own", but it must be
stated rather than discovered: **a decision to move the line is a decision to build a new store.**
The §1 re-measurement is therefore the last moment the value is free.

### The second place, and only the second

Add it to `_staging_run_id`'s key tuple (`run_global_campaign.py` L256 and its twin in
`fill_zones_sequential.py`). The root attr is the *published* guarantee; the run id is
*staging hygiene* — it stops a resume reusing tiles staged under a different line, including
in dev where the root may not be stamped yet. Two lines, and it sits beside
`min_valid_coverage` and `allow_s2_only`, which exist for exactly this reason.

**No `EmbeddingManifest` field is needed.** An earlier draft proposed one in
`REQUIRED_TO_APPEND`; the root identity check supersedes it and is less machinery.

## 5. Work item 2 — the gate

`InferenceConfig` carries the run's line as **`optical_min_obs: int | None = None`**, mirroring the
renamed constant. **`None` means "no refusal", and it is not the same value as zero** — that
distinction is the point, and an earlier draft of this section had it wrong twice over: it claimed
zero disabled refusal "naturally, because `obs < 0` is vacuously false" while showing code that
guarded on `> 0` and compared with `>=`, and it left zero meaning both "refuse nothing" and "nobody
configured this". Two failures already in the register have that exact shape — a throttle whose
off-state was not a stop, and a default nobody set that had therefore never run. So:

- `None` — no refusal. Every existing caller and the whole single-ROI path stay bit-identical.
- a positive integer — the line.
- **zero is refused at construction**, because it can only be a mistake: it would refuse nothing
  while looking like a configured rule.

**The campaign flows must not default it.** They resolve it from the store's root attr (§4) and
**refuse to dispatch if that is absent** — a campaign that silently ran unrefused because a
parameter defaulted is the failure this whole section exists to prevent. `tessera_embeddings` and
`plain` pass nothing and keep `None`.

Thread it through `build_inference_config` -> `fill_zone_year`, `run_global_campaign`,
`fill_zones_sequential`.

Then in `inference/dataset.py`, `MosaicChunkInferenceDataset.__init__`:

```python
valid_mask = s2_nonzero & (s2_valid_count > 0)
if self.optical_min_obs is not None:
    valid_mask &= s2_valid_count >= self.optical_min_obs
if not self.allow_s2_only:
    valid_mask &= s1_total_valid > 0
```

Pass `config.optical_min_obs` where the dataset is constructed (`actors.py`, `_load_strip`).

**Keep the three refusal reasons separable** — a pixel can fail for no S2 at all, for thin S2,
or for absent S1 — because §7's records depend on telling them apart. Compute the component
masks once and reuse them; do not collapse to a single boolean.

## 6. Work item 3 — keep fully-refused shards auditable

Under this rule the dominant skip reason becomes "every pixel was too thin", and a
fully-skipped shard currently writes a **zero-byte** marker and no `s2_obs_count` at all. Its
evidence would vanish.

Two changes, both small:

1. **Give the skip marker a payload.** `write_skip_marker` writes the same per-shard record
   defined in §7 as JSON instead of zero bytes. Assembly's resume scan reads only object
   names (`_list_skip_marker_labels`, and the validation at ~L1305), so content is free and
   this is backward compatible with markers already on disk — treat an empty marker as
   "record unavailable", never as zeros.
2. **Write a stats sidecar for every live shard**, embedded or skipped:
   `<staging>/<run_id>/<label>.stats.json`. This is what makes the registry survive a
   **resume**: a leg that resumes tiles staged by an earlier attempt reports
   `{"status": "success", "valid_pixels": 0, "resumed": True}` with no counters, so a registry
   built from one leg's results would record zeros for everything the previous leg did.
   `summarise_optical_skips` documents this exact trap; the registry inherits it. Sidecars
   persist alongside the staged artifacts and are read at assembly, so a resumed cell is
   complete.

## 7. Work item 4 — the per-shard record

Computed **in the actor**, where the obs array and the embedded mask are both already in
memory (`actors.py` ~L1328). No extra reads.

| field | type | meaning |
|---|---|---|
| `zone` | str | `"33N"` |
| `year` | int16 | |
| `tile_row`, `tile_col` | int32 | position in the zone's shard grid |
| `lon_min/lat_min/lon_max/lat_max` | float32 | WGS84 bbox — so a map needs no reprojection |
| `lon_c`, `lat_c` | float32 | centroid |
| `n_eligible_px` | int32 | ROI-live pixels in the shard — **the denominator** |
| `n_embedded_px` | int32 | |
| `n_refused_thin_px` | int32 | eligible, ≥1 S2 observation, below the line |
| `n_refused_no_optical_px` | int32 | eligible, zero S2 observations at all |
| `n_refused_no_radar_px` | int32 | eligible, refused by the S1 gate (0 when `allow_s2_only`) |
| `s2_obs_mean`, `s2_obs_median`, `s2_obs_p10` | float32 | over **eligible** pixels, not embedded ones |
| `chunks_skipped_mask` | uint64 | **bit _i_ set = inner 256-px chunk _i_ fully refused** |
| `n_chunks_eligible`, `n_chunks_skipped` | uint8 | of 64 |
| `status` | str | `"written"` \| `"skipped"` \| `"resumed_unknown"` |
| `optical_min_obs` | int16 | the rule this row was produced under |
| `refused_depth_hist` | 6 x uint32 | the THIN refusals binned by depth: 1-4, 5-9, 10-14, 15-19, 20-24, 25-29. Excludes zero-observation pixels, which `n_refused_no_optical_px` already counts, so the six bins **sum exactly to `n_refused_thin_px`** — a cheap invariant to assert |
| `mosaic_identity` | str | the ingest marker this cell was filled from — see below |

Four deliberate choices, each of which should survive review:

**Three categories, never two.** Live shards include the land mask's ~11 km sea buffer, so a
coastal shard can be majority ocean. "% skipped" against the full 2048² is meaningless;
`n_eligible_px` is the only honest denominator.

**No stored percentages.** Store counts; document the formulas. A stored percentage is a
second copy of a truth that can drift from its numerator — the exact shape of the corrections
register's *correction applied in one place and not the others*.

**Two fields exist for the planned top-up, not for reporting.** The intent is to revisit
refused pixels once Element 84 publishes more imagery, at which point their observation
counts rise and they clear 30 under the unchanged rule. That later pass needs to choose
*where* to spend without scanning a petabyte, and two cheap fields make the registry its work
list. `refused_depth_hist` answers **how close they were** — a shard whose refusals sit at
25-29 is rescued by a modest backfill, one whose refusals sit at 0-4 is not, and the
distinction is invisible in a single mean. `mosaic_identity` answers **what they were refused
against**, so a later pass can tell whether the input for that shard actually changed rather
than re-probing every mosaic. The staging run id already fingerprints the mosaic's ingest
marker; reuse that value.

**A `uint64` for chunk-level detail.** A shard holds exactly 64 inner chunks, so one integer
records precisely which were fully refused. Per-chunk *rows* would be 193M for the campaign;
this is chunk fidelity at 8 bytes, and the actor computes it by reshaping the refusal mask to
`(8, 256, 8, 256)` and testing `.all()` on the inner axes.

## 8. Work item 5 — the registry files

**The registry lives ALONGSIDE the Icechunk store, never inside it.** Four reasons, and the
first is decisive:

1. Icechunk is transactional and its tags are **write-once forever**. A parquet inside the
   store would make every registry update a commit on published data, and a registry
   correction could never re-pin the tag it belonged to.
2. Icechunk owns its internal layout — snapshots and chunk manifests. Arbitrary files are not
   addressable there by ordinary tooling.
3. The registry must be **rebuildable and replaceable without touching published data**. That
   is the property that makes it a cache rather than a second source of truth (§8.3).
4. Its consumers want `pd.read_parquet("s3://...")` and duckdb, not an icechunk dependency.

### Layout and filenames

**It goes in the AWS Open Data bucket, beside the published store** (ruled 2026-08-13). Production
no longer publishes to a bucket of ours: the store is `s3://tessera-embeddings/v1.1/dclimate.icechunk`,
and the registry is its sibling prefix.

```
s3://tessera-embeddings/v1.1/
├── dclimate.icechunk/                      # the store, untouched
└── dclimate.registry/
    ├── optical_depth_registry.parquet      # the master — compacted from parts/, never hand-written
    ├── index.json                          # per zone-year + campaign totals, human-readable
    └── parts/
        ├── zone=01N/year=2017/shards.parquet
        ├── zone=01N/year=2018/shards.parquet
        └── ...                             # one per landed cell, 1,008 in total
```

A sibling rather than a folder inside the store, for the reasons above **and** one that is specific
to Icechunk: it owns every key under its own prefix and its garbage collection enumerates that
prefix, so a parquet living in there is at best unrecognised and at worst collected.

Three consequences of the bucket it now sits in, none of them optional:

- **It needs Cambridge's write grant**, on `v1.1/dclimate.registry/*`, with the same actions as the
  store's prefix — including delete and multipart-abort, since compaction replaces the master. That
  prefix is already in the access request; what changes is that the request must say a parquet
  dataset lives there, because "1,008 part files plus a master" is a different thing to review than
  "an index file".
- **The schema becomes a published interface.** A registry in our own bucket could be rebuilt into
  a new shape whenever we liked; one in the public dataset cannot, so `schema_version` in
  `index.json` stops being decoration. Adding columns is safe; renaming or retyping one is a
  breaking change to somebody else's query.
- **The rebuildability property (§8.3) is what keeps it honest.** It is still a cache — every row is
  derivable from the store — and that is what makes it safe to replace in place after a correction.

Add `BucketPaths.optical_registry()` alongside `global_store()` and `land_mask_store()`, and **derive
it from the same override the store uses** rather than from `outputs`: production's store location is
a whole-URI override (`global_store_uri`), so a registry path built from `outputs` would land in our
bucket while the store sat in the public one, and every tool would still "work". The existing methods
carry the reason — *"a mask written to a path the ingest does not read would look like success"*.

### 8.1 Written incrementally, at assembly

One parquet per zone-year, in `assemble_zone_year`, at the point `run_provenance` is written
and the cell is tagged. **Never a global rewrite**: the campaign lands 1,008 cells over days
from up to eight clusters at once, so rewriting one object per cell is both a throughput
problem and a lost-update race.

**The master is never written by the campaign** — only by `build_optical_registry.py`, from
`parts/`, on demand. An earlier draft had it "written once at the end", which has two failure modes
and no upside: a campaign that crashes leaves no master at all, and a campaign that ends with
unfilled cells — which is now a **success with a warning**, not a failure — leaves a master that is
silently partial. Rebuilding from parts also means a registry bug is fixed by re-running one script.

**`index.json` carries its own completeness**, for the same reason: a `complete` boolean, the count
of cells expected against cells present, and the unfilled list. A summary file that cannot say
whether it is finished gets read as if it were.

### 8.2 `index.json` — work item 4 of the request

Per zone-year: `n_live_shards` (from the land mask, so the denominator exists even for an
empty cell), `n_shards_written`, `n_shards_fully_skipped`, the pixel counts summed,
`pct_eligible_refused`, `optical_min_obs`, the **cycle** the cell was last filled in and
its **generation** (§13), and the run id. Plus campaign totals, the current cycle, and a
`schema_version`.

Keyed by cycle because that is how users think about the product — `v1.0`, then `v1.1` after
a top-up — not by cell. **Everything here is derivable from the store's own attrs** (§13); the
registry exists so the answer takes one GET instead of 120 group reads.

Deliberately a small JSON a human can open, not a parquet needing tooling — it is the file
someone reads to answer "how did the campaign go" without installing anything.

### 8.3 Rebuildable from the store

`scripts/build_optical_registry.py` compacts `parts/` into the master **and** can rebuild any
row from the store: `s2_obs_count < threshold` combined with `isnan(scales)` reconstructs
every field except those of fully-skipped shards, which write no arrays at all — which is
exactly why §6 exists and why §13 states that exception plainly rather than glossing it.

**Protect this property with a test**, not just a docstring: it is what makes the registry a
cache rather than a second source of truth, and it is what lets a registry bug be fixed
without re-running inference.

Scale: 360,953 live shards x 9 years = **3.2M rows**. Comfortable for a single parquet.

## 9. Work item 6 — reconcile the records this contradicts

**Not optional tidying. Leaving any of these is how the repo ends up asserting two policies.**

1. **`OPTICAL_THIN_MAX_OBS` becomes `OPTICAL_MIN_OBS` at 30, and the meaning inverts** (§4).
   The rename is what produces this list: every use is an import error until it is revisited.
   `actors.py` L1339 becomes the refusal, and `assembly.py` L2248's `s2_thin_below_obs` now
   reports a refusal line and should be renamed with it.
2. `summarise_radar_coverage` reports `s2_thin_pct` over a denominator of **embedded**
   pixels. Once nothing below the line is embedded, that field is structurally `0.000`
   forever. Move the optical half to the **eligible** denominator and rename the fields for
   refusal (`s2_refused_px` / `s2_refused_pct`) — they are consumer-facing in the published
   provenance, and `run_provenance` is the schema's one owner.
3. **`final-data-validation-plan.md` §4 and `campaign-plan.md` state the standing policy as
   "a thin cell is not a broken cell — publish thin cells; label them". This decision reverses
   it.** Both change in the same commit. The AI figure review needs its instructions updated
   too: a sparse cell is now the *expected* output of a thin region, and a reviewer not told
   so will report every refusal as a defect.
4. A new ADR in `context_docs/decisions/` records the rule, the three rulings in §1, the
   evidence in §1-2, and the deferred-spend reasoning in §12.

### 9.2b What refusal does to validation — the interaction that costs the most to miss

Item 3 above says the window reviewer needs new instructions. That understates it: **this rule
removes the review's single most reliable signal, and it opens a hole in coverage checking that
nothing currently fills.** Both were found on 2026-08-13, the second while measuring the first.

**A partially refused shard is the new common case, and nothing has an expectation for it.** The rule
refuses individual pixels, so a shard is now one of three things rather than two: fully embedded, no
pixel below the line; **fully refused**, every eligible pixel below it, which writes a skip marker and
no arrays; or **partially refused** — written, but with NaN scales wherever a pixel failed. Because
refusals cluster (§1: 90.4% of them inside chunks already below the line), a partially refused shard
typically holds whole 256-px chunks of nothing next to healthy ground.

The consequence is a granularity mismatch. `optical_skips` counts whole TILES, and the validator's
identity `written + skipped == live` is a tile-level check, so it stays sound and a fully refused
shard remains auditable — that is what §6 protects. But a *partially* refused shard is "written" by
that identity while missing an arbitrary share of its pixels, and **no record says how much of it
should be missing.** The validator would see a low embedded fraction and have nothing to compare it
against. So:

- **the validator reads the per-shard record** (§7) and grades the embedded fraction against
  `n_eligible_px - n_refused_*` rather than against the whole footprint;
- **`chunks_skipped_mask` becomes load-bearing for validation**, not just for top-up planning — it is
  the only thing that says *which* holes are expected, at the granularity the holes have.

**The artifact signature collision, which is the sharper problem.** A fully refused inner chunk
renders as a flat grey 256-pixel square with chunk-aligned straight edges. That is pixel-for-pixel
the shape of the defect class the window review is best at catching: a blind calibration on 2026-08-13
caught 3 of 3 synthetic dead blocks, describing one as *"a 128x128 flat rectangle whose interior
variance is a fraction of the scene's"*, and 7 of 7 defects overall with no false alarms. After this
rule, that exact geometry becomes a **normal feature of the product**. Two things follow:

- The review must be handed the cell's per-shard record and the chunk bitmask, so it can ask whether
  a rectangular hole is one of the expected ones rather than whether it is rectangular. Without
  that, its sharpest discriminator becomes its highest false-alarm source, and a reviewer that cries
  wolf on every refused chunk is worse than no reviewer.
- The calibration set has to be re-scored **after** the rule ships, with refused-chunk holes present
  in the clean cases. Otherwise it certifies a reviewer against a product that no longer exists.

### 9.1 Tell users about 2017 where they meet it

2017 publishes roughly a third populated, and that is the expected output rather than a fault.
It must be stated in three places a user actually encounters:

- the root store attrs, beside the threshold — a one-line note that coverage varies by year
  and that 2017 is substantially sparser because Sentinel-2B was not yet in routine operation;
- `index.json`, where the per-year figures make it self-evident;
- the root `README` section on reading the global store.

## 10. Tests

- **The gate**: a pixel at 29 is refused and at 30 is kept; at `optical_min_obs=None` the mask is
  bit-identical to today (pin this — it is what protects the single-ROI path); and
  `optical_min_obs=0` is REFUSED at construction rather than quietly disabling the rule (§5).
- **Separability**: a pixel refused for thin optical, one for no optical, and one for absent
  radar land in the three different counters.
- **The bitmask**: a shard with exactly one fully-refused inner chunk sets exactly one bit,
  in the right position.
- **The denominator**: a half-ocean coastal shard reports `n_eligible_px` as its land half,
  not 2048².
- **Resume**: a cell resumed from a prior leg produces a complete registry row from sidecars,
  not a row of zeros. This is the one most likely to be got wrong.
- **Root identity**: reseeding a store with a different `optical_min_obs` is rejected by the
  existing write-once check; a fill whose config disagrees with the root is refused; and a fill
  against a root carrying NO threshold refuses to run rather than running unrefused (§4).
- **Cycles** (§13): `runs[year]` appends rather than replaces, so a second fill keeps the
  first's provenance; generation is the list length; and the work list treats a cell as done
  for the *current cycle* only, so a top-up cycle re-selects it while a resume does not.
- **Fingerprint**: the staging `run_id` changes when `optical_min_obs` changes.
- **Full suite green** with the flag off ⇒ today's behaviour bit-for-bit.

## 11. Verification

1. `uv run pytest tests/unit tests/architecture` green; ruff, `ruff format --check`, and mypy
   clean on touched files (11–15 pre-existing mypy errors are unrelated — compare, do not
   assume).
2. Fill **one dev cell in a mixed zone** with the rule on, and check: the registry row's
   `n_embedded_px + n_refused_*` equals `n_eligible_px`; the parquet reads with a bare
   `pd.read_parquet`; `index.json` totals match the parquet sums.
3. Rebuild that cell's registry row from the store with the standalone script and confirm it
   matches the row written at assembly. Any divergence means the actor and the store disagree
   about what was embedded.
4. Confirm a resumed fill produces the same registry as an uninterrupted one.

## 12. Why the spend is deferred, not lost — for the ADR

Refusing a pixel means no embedding exists for it, so recovering one is a re-run rather than a
filter change. That is deliberate and it is **not** a bet that the line is right forever; it is
a bet that the *input* will improve. The intent is to revisit refused pixels once Element 84
publishes more imagery, at which point their observation counts rise and they clear the
unchanged 30 on their own. §7's `refused_depth_hist` and `mosaic_identity` exist to make that
pass plannable from the registry rather than from a petabyte scan.

Three things the later pass will face, recorded now so they are not discovered then:

- **A top-up is a new CYCLE, not a re-tag.** Per-cell tags are dropped (§13); the later pass
  appends a run record carrying its cycle label and the release gets one tag. Decided now
  because Icechunk tags are write-once forever and the campaign has not yet tagged anything.
- **The unit of a top-up is a shard, not a pixel.** Rewriting any refused pixel rewrites its
  whole shard, so the economics are set by how refusals cluster — which §1's measurement says
  is heavily (90.4% of them inside chunks already below the line).
- **The top-up's work list is shards whose OWN mean is below the line** (ruled 2026-08-13), not
  every shard containing a refused pixel. That is a much smaller set and it is what makes the
  re-run affordable: a shard that is mostly healthy with a thin corner is left alone, and its
  corner stays refused until some later cycle takes the whole shard. The registry's per-shard
  `s2_obs_mean` is therefore the selection column, and `refused_depth_hist` ranks within it.
- **The saving now is smaller than the pixel share suggests.** Cost is token-denominated and
  refused pixels are the thin ones: roughly 18% of pixels carry closer to 9% of tokens, order
  $50K of a ~$573K inference spend. Real, but half what the headline implies.
- **The recovery has not been priced, and the deferral was accepted anyway** (2026-08-13). Re-running
  a shard later costs more than embedding it once now — a new fleet, re-ingested mosaics, and the
  healthy pixels of every selected shard paid for twice — so the $50K is a deferral with a premium
  rather than a saving. The ruling is that the campaign expects budget left over and the premium is
  affordable; it is recorded here so the ADR does not present the figure as a net gain.

The counter-argument, stated once for the record: `s2_obs_count` is already per-pixel in the
store, so a determined user could always have filtered at 30 themselves, and publishing
everything with a quality flag would have cost nothing to reverse. The decision to refuse
instead is a deliberate choice to protect users who do not read documentation, taken on the
reasoning that a published embedding is taken at face value and unsatisfactory results damage
trust in the whole product.

**Two objections were put to that reasoning on 2026-08-13 and answered.** First, that the branch
being taken is the irreversible one while the reversible one is free — publishing with a flag can be
tightened later, refusing cannot be loosened without a re-run. Second, that a hole is not obviously
kinder to a careless user than a noisy value: a thin embedding degrades a downstream model slightly,
whereas a chunk-shaped hole changes the geometry of a study area and the length of a time series, and
for yield modelling in particular a NaN is not the gentler failure. **The ruling stands**: the
dataset's reputation is the thing being protected, and erring toward high quality serves that even
where it costs a user convenience. Recorded because the objections are the ones a reviewer of the ADR
will raise, and the answer is a judgement about reputation rather than a measurement.

---

## 13. Generations, update cycles, and what tags are actually for

Settled while reviewing this plan, and settled **now** because no cell has been tagged yet.
Once the campaign runs, its tags are write-once forever.

### The mismatch that made tag naming feel impossible

One mechanism was being asked to do five jobs:

| job | asked by | what it wants |
|---|---|---|
| idempotence — has this cell landed? | the campaign work list | a boolean per cell |
| generation — which version, how many? | the top-up plan (§12) | a counter and history, per cell |
| reproducibility — the exact store behind a result | debugging | a snapshot pin |
| release — the published product | users | a few curated names |
| audit — when, from what, under what rule | everyone | structured data per cell |

Tags are good at reproducibility and release, serve idempotence by accident, and serve
generation and audit badly. No naming scheme fixes that, because it is a mechanism mismatch
rather than a naming problem. Two further facts settle it:

- **A per-cell tag pins the WHOLE repo.** All 120 zone groups share one Icechunk repository,
  so `zone-33N-2021` is the entire store as of the moment that cell landed — a snapshot in
  which most other zones are still empty. Nobody wants to read that.
- **Per-cell reproducibility is not wanted** (repo owner, 2026-08-13): the latest state is
  what users ask for, and they think in **update cycles**, not per-cell dynamics.

### The model

**An update cycle is the user-facing unit.** One pass of the campaign — the initial fill, or
a later top-up batch once Element 84 publishes more imagery — is one cycle, labelled
`v1.0`, `v1.1`, and so on.

**Per-cell tags are dropped entirely.** Tags mark cycles and nothing else:

```
release-v1.0            <- what a user cites
release-v1.1            <- after a top-up batch
year-2021-complete      <- already exists, keep
snap-2026-09-14T0000Z   <- optional operational pins
```

Prefixed by kind so `list_tags()` filters cleanly. ISO-8601 without colons, so lexicographic
order is chronological order and shells and URLs stay unharmed. **Never encode a mutable
fact** — a count, a percentage, a coverage figure — in a name that can never be changed.

**Generation lives in the data, not in a name.** `run_provenance` currently ends
`return {**existing, str(year): record}`, which **replaces** the year's record — so a
top-up would silently erase the first fill's `run_id`, `optical_skips`, `input_coverage`
and `code`, which is exactly the evidence a top-up needs to compare against. Change
`runs[year]` to a **list**, append rather than replace, and give each record a `cycle`
field. Then:

- generation of a cell = `len(runs["2021"])`
- what changed in a top-up = the last two records
- what is in a release = every cell whose runs contain that cycle

### The contract: attrs alone must answer every user question

The registry (§8) is an **efficiency layer, never a source of truth**. Anything a data user
needs must be answerable from the store itself:

| question | answered from the store alone |
|---|---|
| what rule produced this product? | root attr `optical_min_obs` |
| which cycle is this store at? | root attr `current_cycle` |
| has this cell been filled — how often, when, by which code? | `runs[year]` list |
| which tiles in this cell were fully skipped? | `runs[year][-1].optical_skips.labels` |
| how much of this cell was refused, and why? | the optical/radar summaries in the same record |
| **is this specific pixel refused?** | `s2_obs_count < optical_min_obs`, and `isnan(scales)` |

What the registry adds is speed and reach, not facts: per-shard rows, the chunk bitmask, the
depth histogram, and cross-zone queries in one file instead of 120 group reads.

**The one honest exception**, which must be stated in the README rather than glossed: a
**fully-skipped** shard writes no arrays at all, so its per-pixel depth is not recoverable
from the store. The attrs still name it (`optical_skips.labels`, which is complete), so a
user can always tell "refused" from "ocean" — the thing that record was invented for. Only
the *distribution* of how thin it was lives solely in the registry, and that is a top-up
planning field rather than a user-facing one.

### Migrating idempotence off tags — the one real hazard

The work list currently reads
`not status.has(z, y) or zone_year_tag(z, y) not in existing_tags`
(`campaign.py` L326, `run_global_campaign.py` L859). `status.has` reads `years_complete`,
which the shard writer advances **atomically with the data**, through exactly one writer that
also writes `runs`. So the attr alone is a sound "the data landed" signal and the tag half is
belt-and-braces.

**But the two halves are not synonyms**, and this is the thing to get right: `years_complete`
means *the data landed*, while the tag means *the cell was finalised* — committed, tagged, and
its validation dispatched. A crash between those two points leaves a cell that `status.has`
calls done and that never got tagged, and today's OR re-runs it. Dropping the tag check
without replacing that distinction would silently skip such a cell.

Replace it with the cycle record, not with nothing: a cell is done **for this cycle** when
`runs[year]` contains a record whose `cycle` matches the running one, and that record is
appended at the point the tag is created today. The work list keys on that. A resume then
behaves exactly as it does now, and a top-up cycle naturally re-selects every cell it wants.

### What this means for the refill tooling (it lives in `yield-embeddings`)

An earlier draft of this section called `scripts/reopen_zone_year.py` a missing tool.
**That was wrong and is retracted** — it exists, along with `drop_run_field.py`,
`validate_zone_group.py`, `campaign_health.py` and `validate_all_cells.py`, in the sibling
**`yield-embeddings`** repository, which is where the validation instrument lives.
`final-data-validation-plan.md`'s paths are relative to that repo, not this one. Do not
rewrite any of them.

What *does* change is how much work `reopen_zone_year.py` has to do. Its docstring describes
a three-step refill whose **middle step is expected to fail**: it clears one year from
`years_complete`, the refill then commits its data and raises at the tag step because the
canonical `zone-<Z>-<Y>` name is already spent, and the operator pins the new snapshot under
a fresh name with `--pin-fresh-tag`.

Under this design that awkwardness disappears at the root rather than being worked around:

- there is no per-cell tag to collide with, so **the refill no longer fails at a tag step**;
- `runs[year]` appends, so a refill no longer erases the original fill's provenance;
- idempotence keys on the **cycle**, so a top-up cycle re-selects every cell it wants without
  anyone clearing marks by hand.

The residual need is narrower: forcing **one** cell to refill outside a cycle, after a defect.
That becomes "clear this cell's record for the current cycle", or simply a `--force` flag on
the fill — no tag surgery and no expected-failure step.

### Which repo owns what

**This is a cross-repo change**: `yield-embeddings` holds the refill and validation tooling,
and it reads the campaign's tags and `years_complete` semantics — both of which move here.
The dependency is one-way and currently clean (`tessera-embeddings` imports `yield_embeddings`
**zero** times); keep it that way.

The boundary is **deployment identity**, not domain versus operations. All five scripts import
`yield_embeddings.config.buckets` — which account, which buckets, which deployment — while
their logic is this repo's data model. The rule that follows, and this repo is **public**, so
it matters:

> **This repo owns the data model and the operations on it. The private repo owns deployment
> identity.** Domain functions take a `BucketPaths`; the private side populates it. Nothing
> carrying an account, a bucket name, or a Prefect deployment name moves here.

Applied to the refill: **do not relocate `reopen_zone_year.py` — dissolve it.** Its logic is
four tessera imports deep in this repo's own campaign model; only its wiring is Arbol's. Under
this design its job becomes a `--force` / `--reopen` parameter on the fill flow, which lives
here and already receives `BucketPaths`. The private repo keeps a thin invocation wrapper or
nothing.

The four validation scripts stay where they are: they need `yield_embeddings.domain` as well
as bucket config.

### The validation modules move here, ideally FIRST (repo owner, 2026-08-13)

**Why, by the written boundary rather than by taste.** `yield-embeddings/README.md` defines
that repo as *"a thin production layer over the OSS `tessera_embeddings` library"*, wrapping it
with CDK infrastructure, the closed-source coarsening flow, Prefect deployments, the Packer AMI,
customer ROI resolution, and operational tooling — with `domain/` for *"closed-source domain
functions (coarsen, …)"*. This repo's own `architecture_tests/allowlist.py` says the same from
the other side, calling closed-source repos *downstream consumers* that extend the OSS rules.

Validation is none of those. It audits the output of the library's own assembly step, which the
README puts in the library's column, and its imports agree: four tessera modules and nothing
private but its own rules file. The CLI drivers and the `validate_zone_year` flow ARE
operational tooling and stay.

**How it ended up there:** born 2026-08-11 in `yield-embeddings` commit `077a7ca`, *"Validate
every cell inside the campaign, as a flow of its own"* — written next to the flow that calls
it, which legitimately lives there. Placement by adjacency, not by decision.

Ruled 2026-08-13: **the move is in scope.** The methodology is not being treated as
commercial know-how.

**Say in the module docstring that the rules are meant to be tuned.** Once this is in the OSS
library, someone running a different campaign — a different sensor mix, a different region, a
different quality bar — should be able to adjust the thresholds without forking. That is
almost entirely already true: every numeric threshold is a keyword parameter whose default is
the module constant (`guard=DEFAULT_EDGE_GUARD`, `ratio_tol=DEFAULT_RATIO_TOL`,
`max_flagged_fraction=DEFAULT_MAX_FLAGGED_FRACTION`,
`dominant_share_floor=DOMINANT_SHARE_FLOOR`, `tolerance=SCALE_SHARE_TOLERANCE`). So this is a
docstring edit.

**DONE 2026-08-13, in `yield-embeddings` — carry it through the move, do not redo it.**
`blocking()` took its policy from the module constant, so *which findings fail a cell* was the
one thing an alternative campaign could not change without editing the file:

```python
def blocking(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.status == CHECK_DISAGREES and f.slug in BLOCKING_SLUGS]
```

It now takes `blocking_slugs` like its five neighbours, and the policy is carried on
`CellAudit` and accepted by `audit_cell()` — so the verdict, the report and the campaign's
`CellValidationFailed` raise all read ONE policy rather than the leaf function offering a knob
nobody could reach. It deliberately did **not** go on `CheckOptions`: that object is the
sampling budget ("how much of the cell to read"), and what counts as failure is a different
question. The module docstring now says the thresholds are defaults rather than laws.

That repo's own scripts (`campaign_health.py`, `validate_all_cells.py`) and the
`validate_zone_year` flow still read `rules.BLOCKING_SLUGS` directly for their own reporting —
correct, because that is one campaign's tooling choosing the default.

The tessera-side threshold `OPTICAL_MIN_OBS` is deliberately NOT among the tunables in
that sense: it is campaign-wide, stamped into the store's root attrs, and enforced (§4). A
downstream campaign sets it for its own store; it is not a per-validation-run argument.


`yield_embeddings.domain.embedding_validation{,_rules}` relocate to this repo. Sizing, so the
implementer is not surprised:

| file | lines | coupling |
|---|---:|---|
| `embedding_validation_rules.py` | 1,201 | **none** — `dataclasses`, `math`, `numpy` only |
| `embedding_validation.py` | 2,075 | four *tessera* modules; from `yield_embeddings`, only its own rules file |
| `tests/unit/embedding_validation/` | 1,303 | paired |

**~4,600 lines with essentially nothing to untangle**: the rules module is pure logic, and the
audit module's sole private dependency is the rules file travelling with it. It violates none
of the enforced architecture rules (the forbidden set is prefect, boto3, botocore,
`tessera_embeddings.profiling`, plus three call names; this uses PIL, fsspec, zarr, numpy,
asyncio).

**Pillow becomes a new optional dependency** — 29 references in a bounded ~300-line rendering
block. Add a `validation` extra rather than splitting the module; the repo already carries
four extras and splitting to dodge one optional dep is the worse shape.

**Stays behind:** the CLI drivers (`validate_cell.py`, `campaign_health.py`), which need
`yield_embeddings.config.buckets`, and `test_campaign_health_detectors.py`.

**The move happens AFTER the campaign (ruled 2026-08-13), and the ordering argument below is
superseded.** An earlier version of this section wanted the move first, so that every consumer of the
constant would sit under one CI and the collapse below would fail a test rather than surface later.
That benefit is real but it is available far more cheaply, and the cost of the move is not: ~4,600
lines across a repository boundary, a new optional Pillow dependency, and the modules the campaign's
own monitoring depends on, four weeks before the deadline.

**What to do instead, now:** the validator already takes the threshold as a parameter whose default
is the constant (`optical_thin_max_obs: int = OPTICAL_THIN_MAX_OBS`), so the logic is safe already
and only two *prose* statements are wrong. Fix those two in the same commit as the rename — which the
rename forces anyway, since the old name stops resolving. No move required, and the collapse cannot
happen silently.

**The two places, which the rename now surfaces as import errors rather than as silent drift:**

- **L458** defines a metric as *"embedded pixels with fewer than `OPTICAL_THIN_MAX_OBS` valid
  optical observations"* — the **empty set by construction** once nothing below the line is
  embedded;
- **L1770** prints *"under {OPTICAL_THIN_MAX_OBS} obs — a PROPERTY OF THE INPUT, not a
  fault"*, which becomes vacuous and contradicts the new policy outright. (It was L1749 when this
  plan was written; the file has since gained the per-window depth fields, which is a standing
  reminder that a line number in a handoff document is a hint, not an address.)

That is §9.2's denominator collapse again, in a repo §9 does not reach — so add both files to §9's
reconciliation list explicitly, since they live where §9's grep does not run.
