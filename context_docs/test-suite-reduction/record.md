# Test-suite reduction: record

What the plan in [`README.md`](README.md) did when it was executed, 2026-11-05 to 2026-11-07, in
PR #268. The five per-slice reviews that served as work orders are in git history at `431d4073`.

## Before and after

| | before (2026-11-04) | after (2026-11-07) |
|---|---|---|
| tests in the default run | 4,177 | 2,803 |
| CPU-seconds, default run | 618 | 136 |
| wall-clock, 8-core laptop, `-n auto` | 90–98 s | 23 s |
| lines under `tests/` | 64,600 | 51,400 |
| lines under `src/` | 54,500 | 53,200 |
| test files | 145 | 131 |

The line count missed the plan's ~46,000 target: optional helper refactors the reviews proposed
(test_scheduling's mock config, test_strip_loop's store builders, ingest_zone_year's fake
dispatchers) were not done, and the claims below kept tests the reviews would have cut.

## Where the time went

One fixture removed most of it: `run_forked` spawned a fresh process pool per assemble, and every
child re-imported the package (ADR 041). That alone took the suite from 618 to 231 CPU-seconds.
Lazy ingest re-exports cut `import tessera_embeddings` from 0.80 s to 0.06 s, in production too.

## Per slice

| slice | tests | notable |
|---|---|---|
| assembly + runners | 551 → 436 | run-provenance tests in one parametrized file; dead `concurrency.py`, `compute_std` path |
| storage + config + root | 937 → ~570 | dead `region_writes.py`, `rollback_commits`, `S3Config`; repair script deleted; hypothesis removed |
| ingest | 1,393 → 823 | classifier tests collapsed onto one table; three ROI-mask files merged; dead roi_processing helpers |
| inference + profiling | 615 → 430 | `shard_record.py`, quantize/dequantize deleted; GPU starvation model moved to `scripts/scoping/` |
| flows + providers + integration | 764 → 572 | terminal-hook checks in one test; ruff TID253 replaces an AST test; six duplicate cassettes |

The slice counts come from the per-slice reviews, which counted parametrized cases and files shared
between slices differently, so they do not sum to the headline; the headline is the measured default run.

## Claims that did not hold

Each was checked before anything was deleted; these did not survive the check.

- `assembly/test_assembly.py::TestAssembly::test_same_date_overwrite_resets_untouched_vars_to_fill` — DEAD (only `embedding_std` can be dropped) — the reset also covers the other S1 orbit's obs count on a single-orbit refill; rewritten on that variable instead of deleted.
- `storage/test_published_store.py::TestSampleLivePixels` (4 tests) — DEAD (`sample_live_pixels` only reached by a bench script; move it there) — `tests/integration/test_published_store_access.py` also calls it to find a live pixel in the published store, so it stays in `published_store` with its unit tests.
- `inference/test_assessed_window_coverage.py` (whole file) — MERGE into `test_data_loading.py` with the boundary tests in one table — Group 4 had since made it the home of the time-window coverage gate (19 tests, 220 lines, not 8 and 74), and the table saved three lines while adding test ids; left as is.
- `profiling/test_watch_scheduler.py::test_spill_trips_while_every_scheduler_signal_is_nominal` — DUP of `test_alert_worker_spill` — the survivor asserted only that `worker-spill` fires, not that it fires alone; the exclusivity assert was folded into it before the deletion.
- `orchestration/flows/test_flow_ray_teardown.py::test_ray_owning_flows_do_not_import_the_aws_provider_at_module_scope` — express the rule as an import-linter `forbidden` contract — import-linter also counts function-level imports, and these flows import the provider inside functions on purpose, so the contract cannot say "module scope only"; ruff's TID253 (`banned-module-level-imports`, in `ruff.toml`) can, and replaced the AST test (verified against a planted violation).
- `fixtures/stac_cassettes/*` — "byte-identical (same md5)" — the duplicates differ in CDN and date response headers, so their md5s differ; the recorded responses are the same, so they were still deleted as duplicates.
- `providers/test_provider_aws_ray.py` (moto seed made per-worker) — "~10 CPU-s saved" — measured 12.2 → 9.7 CPU-s serially: a `@mock_aws` test between two seeded tests resets moto's backends, so the seed is rebuilt after each one.
- `ingest/opera_query.py::mgrs_tile_to_bbox` — DEAD (no caller in this repo or yield-embeddings `src/`) — yield-embeddings' `scripts/radar_source_sweep.py` imports it; restored, and fixed to take the extent of all four tile edges rather than two corners.

## Left as is

- `scripts/scoping/gpu_starvation.py` imports `tests/unit/zone_density.py`, so it runs from the repo root.
- `student_v2.QKNormEncoderLayer` and `models/modules.TemporalEncoding` have no caller; they are deleted in a separate PR (ruling §5.5).
