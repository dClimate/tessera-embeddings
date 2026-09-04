# Public API

This is the canonical list of names that the `tessera_embeddings`
package commits to. Anything not on this list is implementation
detail and may change in any release without warning.

CI verifies that the names listed here match `tessera_embeddings.__all__`.

## Configuration

- `BucketPaths` — pydantic model for storage URIs (inputs / outputs).
  Pass to flows / runners; never read from env inside the package.
- `InferenceConfig` — frozen-ish dataclass with model, sampling, and
  Ray-actor parameters.
- `AssemblyConfig` — worker-process pool sizing for the
  embedding-assembly phase (raw-zarr fork/merge writes, no Dask).
- `TimeWindow` — 12-month rolling window.
- `parse_time_window(s: str) -> TimeWindow` — parses
  `"Month YYYY"` strings.
- `checkpoint_filename(norm_source: str = "aws") -> str` — canonical
  filename for the bundled model checkpoints (`"aws"` or `"mpc"`).
- `INFERENCE_CHUNK_SIZE` — pixel size of one spatial inference tile (2048).
  One tile is exactly one output shard, and both paths use this constant — the
  global fill rejects any `InferenceConfig.chunk_size` that is not the zone's
  shard pitch, so 2048 is a contract there rather than a default. The whole
  chain divides evenly: a 4096-px ingest chunk is 2×2 tiles, and a tile is 8×8
  of the 256-px inner chunks.
- `EMBEDDING_DIM` — output dimension of one Tessera embedding (128).

## Errors

- `ConfigMismatchError` — store config differs from the manifest.
- `CorruptedStoreError` — Icechunk store cannot be opened cleanly.
- `InsufficientCoverageError` — date range fails the time-window
  coverage check.

## Ingest (domain)

- `ingest_s2_roi_reflectance(*, roi_zarr_path, start_date, end_date,
  store_path, client, ...) -> IngestResult` — pure-domain S2 L2A
  reflectance ingest. Caller supplies a connected
  `dask.distributed.Client`. `stream_stac_monthly` (default **True**)
  queries the catalog one month at a time instead of the whole window up
  front, bounding how many STAC items are retained at once. Loads and
  writes are always restricted to the chunk-aligned windows where the ROI
  mask has land — that behaviour is unconditional and has **no flag** (see
  "Cropping to live windows" in `src/tessera_embeddings/ingest/README.md`
  for why the former `crop_to_live_windows` parameter was removed rather
  than defaulted).
- `ingest_s1_roi_sar(*, roi_zarr_path, start_date, end_date,
  store_path, client, orbit, ...) -> SarIngestResult` — pure-domain
  S1 OPERA RTC ingest with batched windows + per-batch credential
  refresh callback.
- `IngestResult` — dataclass returned from S2 ingest.
- `SarIngestResult` — dataclass returned from S1 ingest.
- `S1Orbit` — `Literal["ascending", "descending"]`.

## Inference (domain)

- `run_inference(*, num_actors, config, chunks, mosaic_base,
  staging_base, run_id, t0, log, on_actor_retire=None) -> list[dict]`
  — pure-domain Ray-based inference run. Caller is responsible for
  having connected to Ray (`ray.init` or attached to a cluster).
  `t0` is **accepted and ignored**. The progress line it fed now times
  inference from the dispatch loop's own start, because a run's start
  includes the ingest look-ahead, cluster bringup and model load — so
  reporting it beside a chunk counter read as though inference had been
  running that long. Kept so this surface does not break; drop it on the
  next deliberate pass here.

## Subpackages with their own surfaces

The following subpackages expose APIs that are documented per-module
rather than re-exported from the top-level package:

- `tessera_embeddings.providers.aws` — AWS Ray + Dask provisioning.
  See `providers/aws/gotchas.md`.
- `tessera_embeddings.providers.local` — single-machine Ray + Dask.
- `tessera_embeddings.orchestration.prefect.flows` — `@flow`
  definitions for the reference Prefect deployment.
- `tessera_embeddings.orchestration.prefect.tasks` — thin task shells.
- `tessera_embeddings.orchestration.runners.plain` — orchestrator-free
  YAML-driven pipeline runner.
- `tessera_embeddings.architecture_tests` — reusable architecture-rule
  checker. Run via
  `python -m tessera_embeddings.architecture_tests --source path/`.
- `tessera_embeddings.profiling` — operator profiling harnesses for the
  ingest (Dask scheduler) and inference (Ray GPU) stages. Installed as
  console scripts: `te-watch-scheduler`, `te-ingest-log-queries`,
  `te-ingest-report`, `te-observe-cluster`, `te-compare-outputs`,
  `te-compare-stores`; each also runs as
  `python -m tessera_embeddings.profiling.<stage>.<tool>`. The AWS-facing
  ones need the `aws` extra. See `profiling/README.md`.
- `tessera_embeddings.storage` — Icechunk/Zarr store management:
  `zarr_store` (open / create / append / region-write / windowed
  per-date batch — `write_day_windows`, the cropped-ingest write path) and
  `empty_store` (all-fill store seeding — `create_empty_store`,
  `create_empty_store_from_coords`, `VarSpec`) and `time_axis` (how a store
  encodes its time axis — `TIME_ENCODING`, `read_time_values`, `time_index_of`,
  `compute_doy`, `daily_times`, and the campaign calendar `CAMPAIGN_YEARS`,
  `year_timestamp`, `year_of`, `calendar_year_times`).

  **`daily_times` moved from `empty_store` to `time_axis`.** An explicit public-API
  change rather than a re-export: it belongs with the other time-axis symbols, and a
  second import path for one name is the sprawl this move exists to remove.

## Privacy conventions

- `_`-prefixed module names → fully private (e.g.
  `orchestration.prefect.flows._dask_runner`).
- `_`-prefixed function / attribute names → private to that module.
- `@typing.final` on public dataclasses / pydantic models → no
  subclassing. mypy enforces statically.
