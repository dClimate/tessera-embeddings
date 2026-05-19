# Quickstart

A laptop demo that exercises the full pipeline against a small,
real-world AOI: ~1 km² over Story County, Iowa (the densest
corn-producing county in the United States, picked because both
Sentinel-2 and OPERA RTC-S1 give dense coverage there).

Read time: 5 minutes. Run time: ~5 min for ingest only, ~15 min for
full end-to-end CPU inference.

## Prerequisites

- **`uv`** — install from <https://docs.astral.sh/uv/>.
- **An EDL (Earthdata Login) account** to ingest OPERA RTC-S1. See
  the next section for setup — there are two cases, depending on
  whether your EDL account is a standard account or a SAML /
  Launchpad account.
- **~5 GB of free disk** at the paths in `examples/quickstart/config.yaml`
  (default `/tmp/tessera/...`).
- **A Tessera model checkpoint**, only for the full-pipeline run.
  Download from the public release; set `checkpoint_dir` in
  `examples/quickstart/config.yaml` to its parent directory. The
  expected filename comes from
  `tessera_embeddings.checkpoint_filename()` (or
  `checkpoint_filename(norm_source="aws")` for the AWS-normalised
  encoder).

### Earthdata Login (EDL) setup

Two one-time steps:

1. **Register and approve the ASF Data Access app.**
   Sign up at <https://urs.earthdata.nasa.gov/>. Once logged in, go
   to **Applications → Authorized Apps** and approve **"Alaska
   Satellite Facility Data Access"**. Without this, S1 ingest fails
   with HTTP 401 on every granule.

2. **Pick the auth mode that fits your account.**

   The pipeline reads two env-var triples in this order:

   | Mode | Env vars | Used by |
   | --- | --- | --- |
   | Bearer token | `EARTHDATA_TOKEN` | local quickstart, esp. SAML / Launchpad accounts |
   | Basic auth | `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD` | production pipelines |

   **Standard (password-login) account:** export
   `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` and you're done.

   **SAML / Launchpad account (single-sign-on, no password):** the
   ASF datapool redirect chain returns 401 for these accounts on
   basic auth even with the app approved. Use a bearer token
   instead:

   ```
   1. Log into https://urs.earthdata.nasa.gov/
   2. Profile menu → "Generate Token" (or visit
      https://urs.earthdata.nasa.gov/profile and look for the
      "Generate Token" button under your account).
   3. Copy the long JWT string.
   4. export EARTHDATA_TOKEN=<paste>
   ```

   Bearer tokens have a TTL (typically ~60 days). When yours
   expires, regenerate. **Do not check tokens into git.**

   `EARTHDATA_TOKEN` is a local-development fallback only —
   production deployments must use basic auth (token rotation is
   incompatible with unattended workloads). See
   `src/tessera_embeddings/ingest/auth.py::get_edl_session` for
   the precedence rules.

## Ingest only (5 minutes)

Sets up the env, ingests one month of S2 + S1 over the quickstart
ROI, and stops before inference. Useful for confirming the install
worked and your EDL credentials are fine.

```bash
git clone https://github.com/dClimate/tessera-embeddings
cd tessera-embeddings
uv sync --group inference --group prefect
source .venv/bin/activate

# Pick ONE of:
export EARTHDATA_USERNAME=<your-edl-username>
export EARTHDATA_PASSWORD=<your-edl-password>
# - or, for SAML / Launchpad accounts:
export EARTHDATA_TOKEN=<token-from-urs.earthdata.nasa.gov/profile>

python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml \
    --skip-inference
```

> **macOS note:** use `python -m` after activating the venv, not `uv run python -m`.
> On macOS, `uv run` wraps Python in a subprocess; macOS security callbacks kill the
> Ray GCS C++ process before it finishes starting (Ray issue #54047). Activating the
> venv and calling `python` directly avoids the subprocess wrapper.

What happens:

```
1. Rasterise GeoJSON → ROI Zarr        (~1 s, no cluster)
2. Local Dask cluster (2 workers)
3. S2 L2A ingest from Earth Search     (~2 min, ~12 dates)
4. S1 OPERA RTC ingest from CMR-STAC   (~3 min, ~6 dates)
5. Stop. No inference. No assembly.
```

Output: `${paths.preprocessed}/quickstart_story_county_ia/{reflectance.zarr, sar_ascending.zarr}`.

## Full pipeline (30+ minutes)

Same command without `--skip-inference`:

```bash
source .venv/bin/activate
python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml
```

Now the runner adds two steps:

```
6. Local Ray cluster, num_gpus=0       (1 actor, model loads on CPU)
7. Inference: 1 chunk of ROI           (~15-30 min on a laptop)
8. Local Dask cluster, assemble        (~2 min)
```

Output:
`${paths.outputs}/embeddings/quickstart_story_county_ia.zarr` — a
128-dim per-pixel Tessera embedding store at 10 m resolution.

## Why the laptop run is slow on purpose

CPU torch is slow. **It is the credibility test, not the path you'd
use in production.** If the same domain functions that power the
production GPU pipeline also work — without modification — on a
laptop with `num_gpus=0`, then the inference layer has no GPU-specific
coupling. That's the strongest decoupling proof we can run without
deploying to multiple cloud targets.

For real workloads, run the Prefect flow against
`providers/aws/ray.py`. See
[`docs/providers/aws.md`](providers/aws.md).

## Adapting to a different AOI

The bundled `examples/quickstart/roi.geojson` is a ~1 km² box over
Story County, IA (roughly 84×101 pixels at 10 m resolution, one inference chunk). To use your own AOI:

1. Edit `examples/quickstart/roi.geojson` (any single-feature
   GeoJSON polygon works).
2. Verify both providers return data for your AOI before running:
   ```python
   from pystac_client import Client
   bbox = [your_minx, your_miny, your_maxx, your_maxy]
   s2 = Client.open("https://earth-search.aws.element84.com/v1")
   print(len(list(s2.search(collections=["sentinel-2-l2a"], bbox=bbox,
                            datetime="2024-07-01/2024-07-31").items())))
   opera = Client.open("https://cmr.earthdata.nasa.gov/stac/ASF")
   print(len(list(opera.search(collections=["OPERA_L2_RTC-S1_V1_1"], bbox=bbox,
                                datetime="2024-07-01/2024-07-31").items())))
   ```
   OPERA in particular has uneven geographic coverage; some regions
   are not yet processed.

3. Update `time_range` in `config.yaml` if you need a different
   season.

## Troubleshooting

- **HTTP 401 on `datapool.asf.alaska.edu`** → either
  (a) you haven't approved the ASF Data Access app in your EDL
  profile (see Prerequisites), or (b) your account is SAML /
  Launchpad and basic auth doesn't work — generate a bearer token
  and export `EARTHDATA_TOKEN` instead. Bearer takes precedence
  over `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD` when both are set.
- **`ModuleNotFoundError: ray`** during the inference step → re-run
  `uv sync --group dev --group inference --group prefect`.
- **The runner hangs on "Building Dask graph…"** → check chunk size
  in your config. See [`README.md`](../README.md) §"Why chunk size
  dominates everything".
- **Empty embedding output** → check coverage. The runner skips
  dates with insufficient valid (non-cloud) S2 pixels. The
  `min_valid_coverage` threshold defaults to 5% of the ROI;
  adjustable via `--min-valid-coverage`.
