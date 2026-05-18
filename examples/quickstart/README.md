# Quickstart

This example runs the full `tessera_embeddings` pipeline locally,
without Prefect, against a tiny ~1 km² ROI in southern Minnesota.

It exists to prove that:

* Every domain function works on a developer laptop with `LocalCluster`
  + Ray local mode.
* The same code path the Prefect flows use is reachable without a
  Prefect runtime.

The input AOI (`roi.geojson`) is a 0.01° × 0.01° box near Minneapolis;
at 10 m resolution that's roughly one 2000×2000 chunk. Choose a real
AOI for production runs.

## Prerequisites

* `tessera_embeddings` installed with the inference group::

      uv sync --group inference

* Earthdata Login credentials in your environment for OPERA RTC-S1
  ingestion::

      export EARTHDATA_USERNAME=...
      export EARTHDATA_PASSWORD=...

* Free disk space at the paths configured in `config.yaml` (default:
  `/tmp/tessera/...`).

## Run it

### Ingest only (fast, ~5 minutes)

```bash
python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml \
    --skip-inference
```

This rasterises the ROI, ingests one month of S2 reflectance and S1
SAR (ascending), and stops before inference. Useful for iterating on
the ingest path.

### Full pipeline (slow, ~30+ minutes)

```bash
python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml
```

Adds CPU inference (one Ray actor with `num_gpus=0`) followed by
Dask-based assembly. The inference checkpoint must already be present
at `${paths.inputs}/models/<filename>` (download from the public model
release; the `checkpoint_filename(quantized=True)` helper returns the
expected filename).

## What you get

* `${paths.preprocessed}/quickstart/reflectance.zarr` — staged S2
  Icechunk store.
* `${paths.preprocessed}/quickstart/sar_ascending.zarr` — staged S1
  store.
* `${paths.outputs}/embeddings/quickstart.zarr` — final 128-dim
  embedding store (full-pipeline mode only).

## Verifying the output

The bundled `expected_output/` directory (when populated) holds a
golden checksum for the embedding store. Parity tests in CI run the
quickstart end-to-end and compare; users running locally may compare
selected pixel values via xarray.
