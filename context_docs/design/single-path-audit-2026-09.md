# The SINGLE path after the global campaign — audit and plan

**Status:** audit complete, plan proposed. 2026-09-03, against `main` @ `073e12ff`.

Months of work went into the GLOBAL path, whose complexity and cost dwarf a single ROI. Most of
that work is shared code, so single-ROI runs should have inherited the gains. This audit checks
whether they did, and whether anything global-shaped has crept in that a single-ROI user pays for.

**Headline: the single path works, and it is roughly 30× faster than every document says it is.**
The gains arrived; the documentation did not. What global genuinely did *not* hand back is
resumability and staging cleanup.

---

## 1. Verification — it works end-to-end

Full quickstart on a laptop, no GPU, from a clean `/tmp/tessera`:

```
python -m tessera_embeddings.orchestration.runners.plain examples/quickstart/config.yaml
```

Exit 0. Output store written: `outputs/embeddings/quickstart_denver_co.zarr`, 1.3 MB icechunk
repo with snapshots, chunks and transactions.

| stage | measured |
|---|---|
| ROI rasterise + S2 ingest + S1 ingest (both orbits) | ~2.5 min |
| Inference, 1 chunk, CPU | **56.2 s** |
| Assembly | **0.064 s** (1 tile, 1.66 MB, 8 writes, 1 worker) |
| **Total** | **~3.5 min** |

S2 returned ~12 dates; S1 returned 5 ascending and 5 descending. The ROI filter kept 1/1 chunks.

## 2. The documentation is wrong by ~30×, in five places

Every one of these predates the inference work that landed for the campaign — vectorised temporal
resampling, the async two-deep GPU pipeline, the batch-size change, striping:

```
README.md:111                     "Expect ~30+ minutes"
docs/quickstart.md:118            "Full pipeline (30+ minutes on CPU …)"
docs/quickstart.md:132            "Inference: 1 chunk of ROI  (~15-30 min CPU, ~1-2 min GPU)"
examples/quickstart/config.yaml:8 "Expect ~30+ minutes end-to-end on a developer laptop."
runners/plain.py:310              a runtime WARNING: "Expect ~30+ minutes on a developer laptop."
```

Measured: **56 seconds** for inference and assembly, ~3.5 minutes end to end.

This is the single highest-value fix in the audit and it is nearly free. "30+ minutes" reads as
prohibitive to anyone evaluating the project — it is the first number a new user meets in the
README, and it is discouraging them from a run that finishes while they read the page. The
`plain.py` warning is worse than stale: it fires on every run and is simply false.

Note the quickstart's own framing — *"Why the CPU run is slow on purpose … it is the credibility
test"* — no longer describes reality either.

## 3. What global did NOT hand back

### 3.1 The single path cannot resume — it re-does all inference

```python
runners/plain.py:196   run_id = uuid.uuid4().hex[:12]
```

A fresh random identity every run. An interrupted run's staged chunks are orphaned under an id
nothing will look up again, and the next run re-infers from scratch.

The global path solved exactly this. `run_global_campaign._staging_code_identity()` derives a
resumable fingerprint from `inference_code_identity()` plus the code tarball's ETag, with three
levers (`force_staging_reuse`, `force_staging_restage`, an explicitly stated
`staging_code_identity`) and a design document of its own,
`context_docs/design/staging-identity-and-resume.md`. None of it is reachable from the plain
runner.

At quickstart scale this costs a minute. At the scale a real single-ROI user works at — the
reason the path exists — it costs the whole run.

### 3.2 Staging is never cleaned up

Verified on the successful run above:

```
outputs/embeddings/quickstart_denver_co.zarr   1.3 MB   ← the product
outputs/staging/12d5efed74cf/                  1.7 MB   ← left behind, forever
```

The staged intermediate is **larger than the output**, and it survives a *successful* run. Nothing
in `plain.py` deletes it. `storage.object_store.delete_prefix` exists and is called from exactly
two places — `run_global_campaign.py` and `fill_zones_sequential.py`, both global.

Every re-run adds another orphaned prefix, because of 3.1.

### 3.3 The single path is untunable

`ingest_s2_roi_reflectance` takes nine tuning parameters. The Prefect task shell
(`orchestration/prefect/tasks/ingest.py`) passes **all nine**. The plain runner passes **none**:

| parameter | plain runner | global task shell |
|---|---|---|
| `min_valid_coverage` | default | passed |
| `provider` / `collection` | default | passed |
| `stream_stac_monthly` | default | passed |
| `overlap_window_writes` | default | passed |
| `pipeline_dates` | default | passed |
| `batch_dates` | default | passed |
| `allow_ingest_code_mismatch` | default | passed |
| `s3_region` | not accepted | passed |

The defaults are reasonable for a laptop, so this is a capability gap rather than a performance
bug. But it means a single-ROI user cannot trade memory for speed the way the campaign does, and
cannot lower the coverage threshold for a cloudy AOI.

`num_actors=1` is likewise hardcoded at `plain.py:210`, so a single-ROI user with a multi-GPU box
cannot use it.

## 4. Two outright defects

**`--min-valid-coverage` is dead.** `plain.py:377` declares it, documents it, and gives it
`DEFAULT_MIN_VALID_COVERAGE`. `args.min_valid_coverage` is then never read, and `run_plain` does
not accept the parameter. Anyone passing it is silently ignored — they get the default threshold
and no warning. This is the clearest single piece of evidence for §5.

**`s1_use_s3_direct` is undocumented.** `plain.py:342` reads it from the YAML, with a good
laptop-friendly default of `False`. It appears nowhere in `run_plain`'s documented config schema
(lines 268-289), so a working and genuinely useful knob is invisible to users.

## 5. The root cause: the single path is barely tested

| what | coverage |
|---|---|
| `runners/plain.py` | **4 unit tests**, all mocking `_run_ingest` to check whether S1 is called once or twice |
| `run_plain` itself — config parsing, checkpoint resolution, device selection, assembly wiring | **none** |
| the CLI (`main`) | **none** — which is why the dead flag survived |
| full single-path end-to-end | `tests/parity/test_full_pipeline_parity.py`, an `xfail(strict=True)` stub raising `NotImplementedError` |
| the parity suite | compares **Prefect flow ↔ domain function**; never invokes `run_plain` |

The domain functions the single path calls are well covered — that is what the parity suite is
for, and it is genuinely good. What is untested is the runner's own logic: everything between
reading the YAML and calling those functions.

## 6. What global did right, and single already benefits from

Worth recording so nobody "fixes" these:

- `AssemblyConfig.compute_n_workers` floors at 1 explicitly *"so tiny ROIs don't spawn idle
  processes"*. Confirmed on the run: 1 worker requested, 1 used.
- `plain.py` defaults `s1_use_s3_direct=False` because a laptop outside `us-west-2` has no ASF S3
  STS credentials — a deliberate single-path accommodation in shared code.
- `resolve_s1_orbit(..., allow_none=False)` demands radar here with a careful rationale: on one
  named ROI on one machine, no radar is far likelier to be a broken ingest than radar-free
  terrain. Radar-free land goes through the campaign flows instead.
- The inference speed-ups are real and unqualified — 56 s where the docs promise 15-30 min.

---

## Plan

Four phases, ordered by value per unit of risk. Phases 1 and 2 are independent of each other.

### Phase 1 — Correct the timings (highest value, near-zero risk)

Fix the five stale claims in §2 with the measured figures, and delete the false runtime warning in
`plain.py`. Re-time on a GPU box before writing a GPU number, or drop the GPU claim rather than
guess. Revise the quickstart's "why the CPU run is slow on purpose" section, which no longer
describes the software.

**Note:** `plain.py` is inside the inference code-identity closure, so editing it moves the
staged-tile digest. Batch it with Phase 3 or take it at a campaign boundary.

### Phase 2 — Fix the two defects

Wire `--min-valid-coverage` through `run_plain` to `ingest_s2_roi_reflectance`, or delete the flag.
Wiring it is better — it is the one ingest knob a single-ROI user on a cloudy AOI actually needs.
Document `s1_use_s3_direct` in the config schema.

### Phase 3 — Give the single path resume and cleanup

The larger piece, and the one with a real design question.

**Resume.** Derive the staging identity rather than randomising it, so a re-run finds its own
staged tiles. The global mechanism is more than the single path needs — no code tarball, no
campaign restart semantics — so the proposal is a narrow derivation over
`inference_code_identity()` and the ROI name, not a reuse of `_staging_code_identity()` wholesale.
Read `staging-identity-and-resume.md` first; it records why each term is present.

**Cleanup.** Call `delete_prefix` on the run's staging prefix after a successful assemble, matching
what the global flows do. Leave it on failure, which is what makes resume possible.

These two are one change: resume is only safe if cleanup is scoped to a *successful* run.

### Phase 4 — Test the runner, not just the domain functions

Cover `run_plain` end to end with the domain calls mocked: config parsing including every
documented key, checkpoint URL versus directory resolution, device selection, ROI filtering, the
assembly call's arguments, and the CLI's argument wiring. That last one alone would have caught
§4's dead flag.

Then decide on `test_full_pipeline_parity`: implement it, or delete it and stop pretending the
single path has an end-to-end test. It has been an `xfail` stub for months, and the nightly job
that ran it is suspended.

### Not proposed

- **Do not restructure the single path to share the global orchestration.** It is 388 lines
  against the campaign's 2,048, and that ratio is correct — the plain runner's value is that a
  contributor can read it in one sitting.
- **Do not expose all nine ingest knobs in the YAML.** Add `min_valid_coverage` because it has a
  real single-ROI use; the rest are campaign scheduling levers whose defaults are right here.
- **Do not chase the ingest timings.** ~2.5 min for two sensors and three mosaics over a 1 km ROI
  is network-bound and fine.
