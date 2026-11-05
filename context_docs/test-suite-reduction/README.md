# Test-suite reduction: assessment and execution plan

**Status:** plan, 2026-11-05. Not yet executed. Written for an executing session that did not see
the review; everything it needs is here or in `reviews/`.

## 1. Outcome this plan targets

| | now (2026-11-04) | after | change |
|---|---|---|---|
| tests in the default run | 4,177 | ~2,800 | −34% |
| lines under `tests/` | 64,600 | ~46,000 | −29% (plus ~650 lines of dead `src/`) |
| CPU-seconds, default run | 618 | ~150 | −76% |
| wall-clock, 8-core laptop, `-n auto` | 90–98 s | ~25–30 s | |
| CI `unit.yml`, pytest step, per leg | 5.5–6 min | ~1.5–2 min | |
| test-to-source line ratio | 1.19 | ~0.85 | 0.93 on 2026-09-03 (ADR 015) |

Two months ago the suite was 41,800 lines against 45,000 of source. Since then source grew 21% and
tests grew 55%. The growth is not more coverage: it is the same guarantees restated at several layers,
parametrize cases that add ids without adding behaviour, tests of mocks, of log wording, of source
text, and tests of code nothing calls.

## 2. Diagnosis

Three findings, in order of leverage. Each is verified against the source, not inferred.

### 2a. One production choice costs 40% of the suite's CPU

`storage/shard_writer.py::run_forked` (line ~513) opens a **fresh spawn-context
`ProcessPoolExecutor` on every call, even for one payload.** Every spawned child starts an
interpreter and re-imports `tessera_embeddings`, about 0.8 s, of which ~0.74 s is `ingest.s1_roi`
pulled in by the package `__init__`. Every unit test that reaches `ZarrWriter.assemble`,
`assemble_global` or `write_year_shards` pays this: one assemble measured 1.23 s with the spawned pool
and 0.12 s with an in-process pool. It explains `assembly/` (200 CPU-s), `runners/test_zone_fill`
(62 s), `storage/test_campaign` (60 s for 32 tests), `test_zone_update` + `test_plain_update` (62 s).
Roughly **250 of the 618 CPU-seconds**, and the fix deletes no test.

### 2b. Four fixtures account for most of the rest

| fixture / helper | file | cost | cause |
|---|---|---|---|
| `wired` | flows/test_run_global_campaign | ~24 s | leaves `inference_code_identity` real: hashes the import closure (~80 ms) once per dispatch |
| `prefect_test_harness` | flows/test_dask_lifecycle (one test) | 17 s | boots a Prefect API server (uvicorn + SQLite migrations) for one cheap assertion |
| autouse session `prefect_test_harness` | tests/parity/conftest.py | 12.5 s | an **unmarked** test in `parity/` runs in the default suite and triggers it; no CI job runs that test at all |
| `credential_process` issuer | ingest/test_roi_mask_credential_expiry | 12 s | a Python interpreter per credential lookup, credentials expiring every 2 s |
| moto VPC/SSM seed | providers/test_provider_aws_ray | ~10 s | re-seeded per test |
| `LocalCluster(processes=True)` | ingest/test_roi_mask_construction, storage/test_zarr_store | 7 s + 12 s | two-worker process clusters; the zarr_store ones sit in dead region-write tests |
| Hypothesis `ci` profile | unit/test_properties | ~20 s per CI leg only | 200 examples × 5 ms sleeps |

### 2c. Source code that only tests reach

Deleting these removes ~650 lines of `src/` and every test of them. **Each was grepped across
`src/`, `scripts/` and `../yield-embeddings/src` before being listed; the executor re-greps before
deleting, because the downstream repo imports some names the first census called dead
(`daily_times`, `resolve_region` are LIVE there).**

- `inference/shard_record.py` — whole module (ADR 040 already lists it as test-only).
- `storage/region_writes.py` plus `zarr_store.write_region` / `_write_region` — the only callers
  of the module's helpers. `batched_region_writes` is live and unaffected.
- `zarr_store.rollback_commits`, `set_s3_config` / `_s3_config_override` / `S3Config`.
- `campaign.CampaignStatus.pending`, `years_fully_complete`.
- `orchestration/concurrency.py` (`sliding_window_submit`) and its test file.
- `icechunk_logging.set_base_logs_filter`.
- `ZarrWriter` `embeddings_std` / `compute_std` path — `config/inference.py:987` forces it off.
- `sampling.resampler_for`, `sampling.bucket_for_count`; `quantization.quantize_embeddings`,
  `dequantize_embeddings` (a README sentence is the only reference).
- `ingest.roi_processing.filter_low_coverage_dates`, `identify_low_coverage_ds`;
  `opera_query.mgrs_tile_to_bbox`, `mgrs_tile_to_utm_epsg`; `stac.*` `item_filter_fn` parameter
  (never passed); `apply_roi_mask` fallback path and `fill_value`.
- `observe_cluster.PHASE_PARSER` legacy prose branch (~40 lines; only for workers predating 2026-07-21).
- `published_store.sample_live_pixels` moves into the one benchmark script that uses it.
- Not in this PR (ruling §5.5): `student_v2.QKNormEncoderLayer` (unreachable: the only registered
  family sets `enable_qk_norm=False`) and `models/modules.TemporalEncoding` (zero refs) are deleted
  in a separate PR.

## 3. Rules for the executor

1. **The reviews are the work orders.** `reviews/*.md` list, per file, the verdict and every test to
   remove with a one-word reason (DEAD, DUP, MOCK, PARAM, PEDANTIC, IMPROBABLE, HISTORY). Follow them.
   Where a review's `Cross-file` section corrects an earlier section (ingest does this for
   `test_duplicate_granules`), the correction wins.
2. **Verify before deleting.** A DEAD claim is re-grepped across `src/`, `scripts/` and
   `../yield-embeddings/src`. A DUP claim is checked by reading the named surviving test. If a claim
   does not hold, keep the test and note it in §7.
3. **Default is keep.** A test guarding a real behaviour of a live path stays unless the review names
   a reason. Do not cut beyond the lists without the same one-word justification.
4. **Tests reach one layer.** Flow tests guard flow wiring; runner tests guard runner decisions;
   library tests guard the library. A guarantee tested at several layers keeps the lowest one that
   fully states it (exception: `run_global_campaign`, `ingest_zone_year` and
   `fill_zones_sequential` hold real logic with no runner counterpart; their flow tests stay).
5. **No source-text tests.** Anything built on `inspect.getsource` or `read_text()` substring
   matching goes; a rule worth keeping becomes an import-linter contract or a ruff rule.
6. **One pytest run at a time, always `-n auto`.** Never stack suite runs (`feedback: no concurrent
   test runs`). Measure CPU-seconds with `--durations=0` or a JUnit report, not wall-clock.
7. **One PR; commits of 100–500 changed lines, one coherent step each.** Every commit builds and
   passes the files it touches. roborev reviews each commit; run `roborev-suggestions` before any
   push. Push regularly so the review runs as the branch grows rather than all at once.
8. **Docs ship with the change**, per the repo's writing rules. The suite's README is rewritten,
   not patched, at the end (§6).

## 4. Sequence: one PR, seven commit groups

**Everything lands in ONE pull request** (owner's ruling, 2026-11-05: seven PRs is too much review
burden). The groups below are the order of commits on that branch, not separate PRs. Each commit is
one coherent step of roughly 100–500 changed lines that builds and passes the files it touches, so
roborev's per-commit review stays useful. The PR description is written for a reviewer who reads
the description and this plan, not the diff: lead with the before/after table, then one line per
group naming what it cut and why.

The speed group comes first and deletes nothing, so its commits are verified by an unchanged test
set; every later group is verified by its review's expected counts.

### Group 1 — Speed only. No test is deleted.

Gate after this group: `filename::testname` set identical to `main` except the two moved tests;
CPU-seconds measured before and after; expected ~618 → ~300.

1. **In-process forks for unit tests.** One autouse fixture in `tests/unit/conftest.py` (~20
   lines) replaces `shard_writer.ProcessPoolExecutor` with a `ThreadPoolExecutor` subclass whose
   `submit` round-trips the payload through `pickle.dumps`/`pickle.loads` (so each worker still gets
   its own fork session and picklability is still checked) and sets `shard_writer._PROGRESS_SLOTS`
   from `initargs` directly (not via `_init_fork_worker`, which calls `configure_logging()` and
   breaks caplog). `_fork_stall_watchdog` / `_terminate_pool` must tolerate the thread pool. A
   `real_fork_pool` marker opts out the ~3 tests whose subject is the pool itself
   (`TestRunForkedCatchUp`, `TestForkTelemetry` and the drill-death test in `test_shard_writer`).
   Record it as an ADR (§6). Detail: `reviews/assembly-and-runners.md`, top.
2. `flows/test_run_global_campaign`: stub `inference_code_identity` in the `wired` fixture.
3. `flows/test_dask_lifecycle::test_run_on_cluster_logs_onto_the_calling_flow_run` → `tests/integration/`
   with the `integration` marker; the other four tests stay.
4. `parity/test_ingest_s2_roi_date_loop_parity::test_batch_dates_validation_rejects_bad_combinations`
   → `tests/unit/ingest/` (it is the only test of `batch_dates >= 1`; it never ran in CI).
5. `config/test_inference_code_identity`: module-scoped import-closure fixture.
6. `providers/test_provider_aws_ray`: module-scoped moto VPC/SSM seed.
7. `ingest/test_roi_mask_credential_expiry`: issuer as a shell script, not a Python interpreter.
8. `ingest/test_roi_mask_construction`: one `provider-callable` case, one worker.
9. `inference/test_strip_loop`: module-scoped cached serial reference; `inference_config` copied
   per test. `inference/test_data_loading`: real-icechunk concurrency test at 64×64 × 3 repeats.
   `inference/test_sampling_v2_parity`: keep one of the three 8,451-pair loops (the upstream one).
10. `tests/unit/storage/conftest.py`: one session-scoped seeded GLOBAL-layout template,
    `shutil.copytree`'d per mutating test, used by `test_published_store`, `test_global_store`,
    `test_campaign`, `test_fix_published_store_spatial_transform`.
11. `storage/test_campaign`: `_fill` commits year attrs via `campaign.mark_zone_year_empty`; one
    real `write_year_shards` test remains.
12. `src/tessera_embeddings/__init__.py`: the `ingest.s1_roi` and `ingest.s2_roi` re-exports move
    into the existing module-level `__getattr__` (ruling §5.4). Measure the package import before and
    after with `python -X importtime -c "import tessera_embeddings"`.

`addopts` does not gain `-n auto`: a debugger needs serial runs, and `unit.yml` already passes it.

### Group 2 — assembly + runners cuts (`reviews/assembly-and-runners.md`)

551 → 426 tests, 11,800 → ~9,200 lines. Includes: the five-layer stack/encoder refusal reduced to
three; one parametrized `tests/unit/storage/test_run_provenance.py` replacing 3–4 copies of the
optional-field tests across four record files; moving the icechunk_logging, session_catch_up,
registry_rows and storage-options tests to `tests/unit/storage/`; `test_plain_runner` merged into
`test_plain_runner_wiring`; delete `test_orchestration_concurrency` with `orchestration/concurrency.py`;
delete `compute_std` path and `set_base_logs_filter`; `test_zone_update`'s six private imports from
`test_zone_fill` move to `runners/conftest.py`.

### Group 3 — storage + config + root cuts (`reviews/storage-config-and-root.md`)

937 → 568 tests, 10,200 → ~7,200 lines. Includes: `test_zone_grid`'s 120-id parametrize becomes one
loop; `test_zarr_store`'s dead region-write and rollback classes go with `region_writes.py`,
`write_region`, `rollback_commits`, `set_s3_config`; `test_write_days_windows_batched` →
`test_write_day_windows`; `TestSavedManifestPreload` → `test_global_store`;
`TestCheckTimeWindowCoverage` → `inference/test_assessed_window_coverage`; fault-injection AST scans →
`tests/architecture/`; delete `test_imports`, `test_public_api`, `test_prefect_layer` (import-linter
and ruff F822 enforce them) and `test_properties`; `test_rebuild_registry_measurements_script`
trimmed to CLI plumbing; `test_fix_published_store_spatial_transform` and its script deleted (ruling
§5.1).

### Group 4 — ingest cuts (`reviews/ingest.md`, read its Cross-file section first)

1,392 → ~830 tests, 16,400 → ~12,100 lines. Includes: `test_duplicate_granules`' classifier classes
(85 tests) deleted — `test_read_failure_verdict` is the canonical precedence-aware table; the three
`test_roi_mask_*` files → one `test_roi_mask.py`; `test_solar_day_grouping` → `test_solar_days`,
`test_solar_day_labelling` deleted (tests a helper defined in the test); `test_s2_coverage_windows` →
`test_s2_coverage_gate`; `test_ingest_roi_domain` deleted after its two live tests move;
`test_radar_free_zone` → `tests/unit/inference/`; dead `filter_low_coverage_dates`, `mgrs_tile_*`,
`item_filter_fn`, `apply_roi_mask` fallback deleted.

### Group 5 — inference + profiling cuts (`reviews/inference-and-profiling.md`)

587 → ~400 tests, 11,100 → ~7,500 lines. Includes: `test_gpu_starvation.py` (1,482 lines) leaves the
suite — it is the campaign's cost/starvation model, reaching no `src/` beyond the partitioner that
`test_cluster_balance` already covers; relocate to `scripts/` and repoint its five doc references
(ruling §5.2); delete `shard_record.py`,
`resampler_for`, `bucket_for_count`, `quantize_embeddings`/`dequantize_embeddings`, the legacy
`PHASE_PARSER` branch; dead `ray.nodes` patches in `test_scheduling`; the two golden tests stay
(skip without a checkpoint env var, the only real-weights fidelity evidence); `test_observe_cluster_parser`
deleted.

### Group 6 — flows + providers + integration/parity cuts (`reviews/flows-providers-integration.md`)

728 → ~549 unit tests, 11,700 → ~8,900 lines. Includes: the "terminal hook registered on cancel AND
crash" assertion, repeated in six files, becomes one parametrized test in a renamed
`test_terminal_hooks.py`; `_resolve_run_id` tests from two files → one; `_DeploymentCellInputs`
tests split out of `test_fill_zones_sequential_flow`; `test_run_id_model_version` mostly moves to
`config/`; `TestTheRunnerLifecycle` out of `test_fleet_mix`; delete `test_ingest_s1_roi_parity`
(permanent xfail), `adapter_template/test_your_adapter_parity` (always skip),
`integration/test_stac_query_cassette` (tests pystac-client, not us); env-gate
`test_ec2_launch_pacing`; six byte-duplicate or orphaned cassettes under `tests/fixtures/stac_cassettes`
(~1.2 MB) deleted.

### Group 7 — docs and record

See §6. The last commits on the branch; the PR is not opened for review until they are in.

## 5. Owner rulings (2026-11-05)

Decided; the executor does not re-ask.

1. **`fix_published_store_spatial_transform`: DELETE** the script and its 22 tests. The published
   store was repaired; the seeder now gets the transform right by construction (ADR 024).
2. **`test_gpu_starvation.py` → `scripts/`** as a runnable cost/starvation model, out of the test
   suite. Repoint its five doc references.
3. **CI matrix stays 3.12 + 3.13.** After Group 1 each leg is ~2 min, so both are affordable.
4. **Lazy import of `ingest.s1_roi` (and `s2_roi`) from the package `__init__`: DO IT NOW**, in
   Group 1. The package already lazy-loads `run_id_prefix` and `run_inference` through a module-level
   `__getattr__`; the three S1 names and two S2 names join it, `__all__` is unchanged, and no caller
   in `src/`, `scripts/` or `../yield-embeddings/src` imports them from the top level. Saves ~0.74 s per
   spawned worker in production as well as per test child.
5. **`QKNormEncoderLayer` and `TemporalEncoding`: delete, in a SEPARATE PR**, not this one.

## 6. Documentation the executor rewrites

- `tests/README.md`: new counts and timings (replace "3,675 tests in about 23 s"), the
  `real_fork_pool` marker, the moved tests, the merged files, the "tests reach one layer" and "no
  source-text tests" rules from §3. Three links to the deleted `context_docs/test-suite-streamlining.md`
  (tests/README.md ×2, context_docs/README.md ×1) now point at this directory; keep them pointing here.
- New ADR `041-unit-tests-run-the-forked-writer-in-threads.md`: the spawn pool is production
  behaviour; unit tests exercise it through an in-process pool with a pickled payload, and three
  tests opt out to cover the real pool. Why: §2a.
- ADR 040's list of test-only modules loses `shard_record.py` (deleted).
- `storage/shard_writer.py::write_year_shards` docstring says "in-process when 1": stale, fix.
- When done, replace `reviews/` with a ≤60-line `record.md`: what was cut by reason tag, the
  measured before/after table, and which review claims did not hold (§7). The reviews themselves
  are then deleted; git keeps them.

## 7. Claims that did not hold (executor fills in)

_Empty until execution. One line each: file, test, claim, why it was kept._

## 8. Method, for the record

Measured on 2026-11-04 with `pytest -n auto --junitxml` (per-file CPU-seconds, test counts, lines in
`reviews/per-file-timing-2026-11-04.txt`), CI step timings from `gh run view`, a census of public
`src/` definitions with no other `src/` reference, and a suite run with coverage on (90.5 s vs 90–98 s
without: coverage is not the CI cost). Five Opus reviewers each read one slice of the suite in full,
grepped `src/` for callers, and wrote the per-file lists in `reviews/`. Every dead-code claim in §2c
and the pool finding in §2a were re-verified by hand before this plan was written.
