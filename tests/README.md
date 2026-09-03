# Tests

The `tessera_embeddings` test suite is split by cost and intent. This table describes what is
**actually wired** — where a tier is empty or unrun, it says so, and the Roadmap at the bottom
records what it would take to light it up.

| Directory | Marker | Runs in | Goal |
|---|---|---|---|
| `unit/` | (default) | `unit.yml` — PRs to `main` and pushes to `main`, on Python 3.12 and 3.13, with coverage | Fast, isolated. No test should exceed 30 s. |
| `architecture/` | (default) | `architecture.yml` — PRs to `main` and pushes to `main` | Layer-rule enforcement (no Prefect outside `orchestration/prefect/`, etc.). |
| `integration/` | `integration` | `integration.yml` — PRs to `main`, filtered to `src/`, `tests/integration/`, `tests/parity/`, `tests/fixtures/`, `pyproject.toml`, `uv.lock` | Hit moto / VCR cassettes / LocalCluster. Minutes. |
| `parity/` | `parity` | `integration.yml`, same trigger, as `-m "parity and not slow"` | Prefect flow ↔ plain runner output equivalence. 5–30 min. |
| `slow/` | `slow` | **nothing — the directory is empty.** See Roadmap 1. | Full plain-runner end-to-end. 30+ min. |
| `gpu/` | `gpu` | **nothing — no GPU runner is available.** See Roadmap 2. | Tiny (< 2 min) checks of GPU code paths. |

Default `pytest` invocation only runs unit + architecture, per
`addopts = "-m 'not integration and not parity and not slow and not gpu'"` in
`pyproject.toml`. CI workflows opt in to the heavier markers explicitly.

Two workflows exist but are deliberately dormant, each with the reason and the re-enable
steps in a comment at the top of the file: `nightly.yml` (Roadmap 1) and
`downstream-smoke.yml` (waiting on a stable release of the private downstream consumer).

## Where a new unit test goes

`unit/` is grouped into **subject directories mirroring `src/tessera_embeddings/`**, so an edit
to one subsystem can be verified without running the other 3,700 tests:

```bash
uv run pytest tests/unit/ingest        # ~36 files, a few seconds
```

```
unit/
├── config/  ingest/  inference/  storage/  providers/  profiling/
├── assembly/                 what a published zone-year records about itself
└── orchestration/
    ├── flows/                Prefect flows
    └── runners/              the plain / sequential / zone-fill runners underneath them
```

**Place a file by its primary subject** — what its docstring and test names say it tests, not
which module it imports most. A test importing eight modules still has one subject.

Two departures from a strict `src/` mirror, both deliberate. **`assembly/`** spans
`inference/assembly.py` and `storage/shard_writer.py`, because the publication pipeline is one
theme and mirroring would split it. **`orchestration/flows/` vs `runners/`** keeps a flow's test
away from its runner's — `test_fill_zone_year_flow.py` and `test_zone_fill.py` are different
subjects with confusingly similar names.

**Eleven things stay at `unit/` root** and nothing else should join them: `conftest.py`; the three
shared helpers `zone_density.py`, `mosaic_stores.py` and `coverage_repo.py`, which other tests
import by absolute path and which therefore must not move; and six tests with no single subject —
`test_imports`, `test_public_api`, `test_context_docs_index`, `test_architecture_rules`,
`test_prefect_layer`, `test_properties`.

**Do not locate files by counting directories up from `__file__`.** Import `REPO_ROOT`, `SRC_ROOT`
or `FIXTURES` from `tests/_paths.py`, which anchors on the directory holding `pyproject.toml` and
so works at any depth. Counting levels is what made this regrouping risky: three tests that scan
the source tree fail *silently* when the count is wrong, passing while asserting nothing.

Background in `context_docs/decisions/015-regrouping-the-unit-tests.md`.

## Optional dependencies

Ray and torch are the `inference` extra, not part of the base install, because
torch is large and platform-specific. Anything under `inference/` — the actor
pool, the work-stealing scheduler — therefore cannot even be imported in a plain
checkout, and its tests fail at collection with `ModuleNotFoundError: ray`. That
is a missing extra, not a broken suite. Install them without touching the
lockfile:

```bash
uv sync --extra inference --frozen
```

CI installs `--all-extras`, so it never sees this. To reproduce a CI run exactly:

```bash
uv sync --all-extras --frozen
```

## Hypothesis

Property-based tests live alongside example-based tests in `unit/`.
Profiles are configured in `tests/conftest.py`:

- `local` (default): 20 examples, 500 ms deadline.
- `ci` (when `CI` env var is set): 200 examples, 2 s deadline.

## Fixtures

`tests/fixtures/` holds golden inputs:

- `stac_cassettes/` — VCR.py recordings for STAC queries (`pytest-recording`).
  See its `README.md` for the recording workflow.
- `checkpoints/` — small fake model checkpoints for tests that need to
  load a model. Real production checkpoints are too large for git.

## Running subsets

```bash
# Default (unit + architecture):
uv run pytest

# Property-based tests only:
uv run pytest -k property

# Integration suite (hits VCR cassettes / moto):
uv run pytest -m integration

# Parity contract:
uv run pytest -m parity

# The pipelined CUDA path — NOT covered by CI, run this by hand on a machine
# with a GPU. See Roadmap 2 for when.
uv run pytest tests/unit/inference/test_inference_loop.py -k pipelined -v
```

`-m "slow or parity"` is documented in older notes as the full local end-to-end. It does not
do that today — the only `slow` tests are one Ray-cluster check and an unimplemented stub.
See Roadmap 1.

---

## Roadmap

Things this suite is *designed* to have and does not yet. Each is a deliberate gap with a
known shape, not an oversight — recorded here so they can be picked up rather than
rediscovered. Full analysis in `context_docs/design/test-suite-streamlining-plan.md`.

### 1. The `slow/` tier is empty, and the nightly workflow is suspended

`tests/slow/README.md` names the full plain-runner end-to-end — rasterise ROI, S2 and S1
ingest, CPU inference, assembly — as this tier's canonical occupant. That test exists only as
`tests/parity/test_full_pipeline_parity.py`, whose body is `raise NotImplementedError` under
`@pytest.mark.xfail(strict=True)`. It lives in `parity/` because it needs both markers.

`nightly.yml` was correctly pointed at it and so spent a daily 120-minute runner confirming a
placeholder was still a placeholder. **Its schedule is suspended as of 2026-08-25**;
`workflow_dispatch` is retained. Restore the `schedule:` block when the test is real.

**To pick up:** implement `test_full_pipeline_parity`, then uncomment the schedule.

### 2. The `gpu/` tier is empty, and one GPU path is verified by hand

**No GPU CI runner is available to this project, and one will not be provisioned. This gap is
accepted, as of 2026-08-25.**

`tests/unit/inference/test_inference_loop.py::TestRunInference::test_pipelined_matches_serial` is the
only coverage of the pinned-buffer / CUDA-event / two-deep-drain / backbone-stream-ordering
path. It is gated on `torch.cuda.is_available()`, so it skips on every CI runner — it is the
`1 skipped` you see in each unit run. That skip is the standing signal that this gap exists,
which is why the test keeps its `skipif` rather than taking the `gpu` marker: the marker is
caught by the default deselection and the test would stop being reported at all.

**Verify it by hand** with the command under "Running subsets" above, on any machine with a
CUDA device — after any change to the pipelined loop, the actor pool, or the chunk-staging
path, and once before a campaign starts.

`tests/gpu/` and its README are kept as a dormant spec: they describe what a GPU tier would
look like if a runner ever becomes available, and the `gpu` marker stays declared so those
instructions remain valid.

### Closed, for reference

Two further items were on this list and were fixed in the same change that added it, so they
are recorded here rather than left reading as open work:

- **Two tests no CI job ran.** Both sat in `tests/unit/` behind a marker the default
  `addopts` deselects. The local-Ray smoke test was deleted; the Dask scheduler-plugin test
  moved to `tests/integration/`, where `integration.yml` runs it.
- **A unit test that broke the 30-second bound above.** The source-read resilience test sat
  through the real 8-attempt retry ladder, 61 s of genuine backoff. It now asserts the
  ladder's shape instead of waiting it out, which checks more and takes 0.5 s.
