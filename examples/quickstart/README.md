# Quickstart

This example runs the full `tessera_embeddings` pipeline locally,
without Prefect, against a small AOI over Story County, Iowa — the
densest corn-producing county in the US (USDA NASS).

It exists to prove that:

* Every domain function works on a developer laptop with `LocalCluster`
  + Ray local mode.
* The same code path the Prefect flows use is reachable without a
  Prefect runtime.

The input AOI (`roi.geojson`) is a ~1 km² box (roughly 84×101 pixels /
one inference chunk at 10 m resolution) centred near Ames, Iowa. The AOI was chosen for dense data coverage:

* **Sentinel-2 L2A** (Earth Search): ~12 acquisitions / month over the
  growing season.
* **OPERA RTC-S1** (CMR-STAC ASF): ~6 ascending acquisitions / month.

If you adapt this example to a different AOI, verify both providers
return non-zero results before committing — geographic coverage of
OPERA in particular is uneven (it's a NASA processing program, not a
sensor; some regions have not yet been processed).

## Prerequisites

* `tessera_embeddings` installed with the inference group::

      uv sync --group inference

* Earthdata Login credentials in your environment for OPERA RTC-S1
  ingestion. Two paths:

  Standard EDL account::

      export EARTHDATA_USERNAME=...
      export EARTHDATA_PASSWORD=...

  SAML / Launchpad account (generate a bearer token at
  ``urs.earthdata.nasa.gov/profile`` → "Generate Token")::

      export EARTHDATA_TOKEN=...

  Either way, you must first approve the **"Alaska Satellite Facility
  Data Access"** application under your EDL profile's Authorized
  Apps. See ``docs/quickstart.md`` for the full setup walk-through.

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
release; set `checkpoint_dir` in `config.yaml` to its parent directory.
`checkpoint_filename()` returns the expected filename — or
`checkpoint_filename(norm_source="aws")` for the AWS-normalised encoder).

## What you get

* `${paths.inputs}/quickstart_story_county_ia/reflectance.zarr` — staged S2
  Icechunk store.
* `${paths.inputs}/quickstart_story_county_ia/sar_ascending.zarr` — staged S1
  store.
* `${paths.outputs}/embeddings/quickstart_story_county_ia.zarr` — final
  128-dim embedding store (full-pipeline mode only).

## Verifying the output

The bundled `expected_output/` directory (when populated) holds a
golden checksum for the embedding store. Parity tests in CI run the
quickstart end-to-end and compare; users running locally may compare
selected pixel values via xarray.
