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
- `AssemblyConfig` — Dask cluster recipe for the embedding-assembly
  phase. Subclasses an internal helper; treat the helper as private.
- `TimeWindow` — 12-month rolling window.
- `parse_time_window(s: str) -> TimeWindow` — parses
  `"Month YYYY"` strings.
- `checkpoint_filename(quantized: bool = True) -> str` — canonical
  filename for the bundled model checkpoints.
- `INFERENCE_CHUNK_SIZE` — pixel size of one spatial chunk (2000).
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
  `dask.distributed.Client`.
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
- `tessera_embeddings.storage` — Icechunk/Zarr store management:
  `zarr_store` (open / create / append / region-write) and
  `empty_store` (all-fill store seeding — `create_empty_store`,
  `create_empty_store_from_coords`, `VarSpec`, `daily_times`).

## Privacy conventions

- `_`-prefixed module names → fully private (e.g.
  `orchestration.prefect.flows._dask_runner`).
- `_`-prefixed function / attribute names → private to that module.
- `@typing.final` on public dataclasses / pydantic models → no
  subclassing. mypy enforces statically.
