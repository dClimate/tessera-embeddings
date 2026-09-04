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
| `slow/` | `slow` | **nothing — the directory is empty and no occupant is planned** ([ADR 023](../context_docs/decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md)). | A home for a genuinely slow test, if one ever needs it. |
| `gpu/` | `gpu` | **nothing — no GPU runner is available.** See Roadmap 2. | Tiny (< 2 min) checks of GPU code paths. |

Default `pytest` invocation only runs unit + architecture, per
`addopts = "-m 'not integration and not parity and not slow and not gpu'"` in
`pyproject.toml`. CI workflows opt in to the heavier markers explicitly.

One workflow exists but is deliberately dormant, with the reason and the re-enable steps in a
comment at the top of the file: `downstream-smoke.yml`, waiting on a stable release of the private
downstream consumer. (`nightly.yml` is **deleted** — its only selector matched a stub for a test
that will not be written; see ADR 023.)

## Where a new unit test goes

`unit/` is grouped into **subject directories mirroring `src/tessera_embeddings/`**, so an edit to
one subsystem can be verified without running the whole of it:

```bash
uv run pytest tests/unit/ingest -n auto   # 36 files, 1,387 tests, ~9 s
uv run pytest tests/unit/config -n auto   # 10 files, 170 tests, ~4 s
```

**Measured, so you know what to expect:** the full unit suite is 3,675 tests in about 23 s, and the
directories run 84–1,387 tests in 3–9 s. The saving is real but not dramatic in wall-clock terms —
interpreter start, collection and the `xdist` workers cost roughly three seconds before any test
runs, so even the smallest directory does not drop below that. What you actually get is a **focused
failure list**: a red run names something in the subsystem you touched.

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

**Ten things stay at `unit/` root** and nothing else should join them: `__init__.py` and
`conftest.py`; the three shared helpers `zone_density.py`, `mosaic_stores.py` and
`coverage_repo.py`, which other tests import by absolute path and which therefore must not move;
and five tests with no single subject — `test_imports`, `test_public_api`,
`test_architecture_rules`, `test_prefect_layer`, `test_properties`.

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

**Pass `-n auto`.** It is not in `addopts`, so a bare `uv run pytest` runs serially and is several
times slower than the timings quoted in this file. CI passes it (`unit.yml`), and so does the
coverage gate (`scripts/test_coverage_gate.py`) — parallel and coverage compose fine. Add it to any
run you are waiting on; leave it off when you want a debugger to stop somewhere useful.

```bash
# Default (unit + architecture):
uv run pytest

# Property-based tests only:
uv run pytest -k property

# Integration suite (hits VCR cassettes / moto):
uv run pytest -m integration

# Parity contract:
uv run pytest -m parity

# The pipelined CUDA path — NOT covered by CI. Roadmap 2 has the setup and when to run
# it; the short version is that the lockfile gives you CPU torch, so this SKIPS unless
# you install a CUDA build first, and a skip is not a pass.
uv run --no-sync pytest tests/unit/inference/test_inference_loop.py -k pipelined -v
```

`-m "slow or parity"` is documented in older notes as the full local end-to-end. It is not, and
will not be: **no test carries the `slow` marker**, and the end-to-end path is verified by running
the quickstart by hand ([ADR 023](../context_docs/decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md)).

---

## Roadmap

Things this suite is *designed* to have and does not yet. Each is a deliberate gap with a
known shape, not an oversight — recorded here so they can be picked up rather than
rediscovered. Full analysis in `context_docs/test-suite-streamlining.md`.

### 1. ~~The `slow/` tier and the full-pipeline end-to-end~~ — CLOSED, and not by building it

**Decided 2026-09-03, and final: there will be no automated full-pipeline end-to-end test.
Running the quickstart by hand IS that verification.** See
[ADR 023](../context_docs/decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md).

The `xfail(strict=True)` stub that stood for it and the `nightly.yml` workflow pointed at that stub
are both deleted. **The stub was not a neutral cost:** it read as queued work, and a review in
September 2026 re-proposed the test on the strength of it. That is the reason the decision is
recorded as an ADR rather than as a line here.

What follows for a reader: `run_plain`'s end-to-end path has **no automated coverage, by decision**.
Run the quickstart after changing `run_plain` or the stages it drives — about three and a half
minutes on a laptop, following [`docs/quickstart.md`](../docs/quickstart.md).

**Two ways that run silently checks less than you think**, both in ADR 023: ingest resumes off the
dates already in the store, so a rerun against a populated `/tmp/tessera/inputs` skips ingest
altogether — delete it first; and the shipped config says `device: auto`, so on a GPU box it is not
the CPU path being verified — set `device: cpu`.

### 2. The `gpu/` tier is empty, and one GPU path is verified by hand

**No GPU CI runner is available to this project, and one will not be provisioned. This gap is
accepted, as of 2026-08-25.**

`tests/unit/inference/test_inference_loop.py::TestPipelinedGpuLoop::test_pipelined_matches_serial`
is the only coverage of the pinned-buffer / CUDA-event / two-deep-drain / backbone-stream-ordering
path. It is gated on `torch.cuda.is_available()`, so it skips on every CI runner — it is the
`1 skipped` you see in each unit run. **That skip is the standing signal that this gap exists**,
which is why the test keeps its `skipif` rather than taking the `gpu` marker: the marker is caught
by the default deselection, so a marked test would stop being reported at all. Tidier, and it would
hide the one visible sign that the path is unverified.

#### Running it by hand — the setup is the part that goes wrong

**`uv sync --all-extras --frozen` gives you CPU torch on a CUDA machine.** The lockfile pins torch
to `download.pytorch.org/whl/cpu` for *every* platform, so a bare `--frozen` installs
`torch==2.12.0+cpu`, `torch.cuda.is_available()` is False, and the test **skips** — reporting
exactly what it reports in CI. An operator following that alone would believe they had verified the
path.

**Use the repository's own GPU install** — `docs/environment-setup.md` "GPU installs", which
explains why `--extra-index-url` alone is not enough (PyPI's CPU wheel stays in the candidate pool
and can win). **Check the index before pinning a build**: `torch==2.6.0` publishes `cu118`, `cu124`
and `cu126`, and `https://download.pytorch.org/whl/<cuXXX>/torch/` is the list.

**The `==` in that command is load-bearing, not decoration.** `uv pip install torch` — no version —
is already satisfied by the CPU wheel `uv sync` just installed, so uv audits it and reports *"Would
make no changes"*. You get a clean-looking setup, still on `+cpu`, and a skip. Measured, not
assumed. Pin the version so the requirement genuinely differs from what is installed; do not reach
for an unpinned `--force-reinstall` instead, which takes whatever that index tops out at.

```bash
uv sync --all-extras --frozen
uv pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124

# --no-sync on everything after this. `uv run` re-syncs from the lockfile by default, which
# would put the CPU wheel straight back.
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CPU torch — this will skip'"
uv run --no-sync pytest tests/unit/inference/test_inference_loop.py -k pipelined -v
```

**One thing to know rather than to fix here.** That CUDA build is `2.6.0`, while the lockfile's CPU
build is `2.12.0`, so the manual check does not run against the same torch version CI does. That
mismatch is the repository's, not this test's — the GPU docs and the lock have drifted apart — and
resolving it is a dependency question rather than a test one. **Say which version you ran against
when you report a result.**

**Read the outcome, not the exit code.** `pytest` exits 0 on a skip, so "the command succeeded" is
not evidence. The only result that verifies anything here is a **passed**.

**When:** after any change to the pipelined loop, the actor pool, or the chunk-staging path, and
once before a campaign starts. Manual verification with no trigger attached does not happen.

`tests/gpu/` and its README are kept as a dormant spec: they describe what a GPU tier would look
like if a runner ever becomes available, and the `gpu` marker stays declared so those instructions
remain valid.

### Closed, for reference

Two further items were on this list and were fixed in the same change that added it, so they
are recorded here rather than left reading as open work:

- **Two tests no CI job ran.** Both sat in `tests/unit/` behind a marker the default
  `addopts` deselects. The local-Ray smoke test was deleted; the Dask scheduler-plugin test
  moved to `tests/integration/`, where `integration.yml` runs it.
- **A unit test that broke the 30-second bound above.** The source-read resilience test sat
  through the real 8-attempt retry ladder, 61 s of genuine backoff. It now asserts the
  ladder's shape instead of waiting it out, which checks more and takes 0.5 s.
- **Two tests that had quietly given back a fifth of the suite's speed** (2026-09-03). A fleet-mix
  test reached `ray.kill` on its teardown path, which is wrapped in Ray's auto-init hook, so it
  **booted a real local Ray cluster** every run — the third occurrence of a hazard this repo
  documents. And the documentation-index guard walked the whole working tree (2,975 markdown files,
  of which 66 are the repository's) and took 10.3 s. **That guard is now deleted**, not optimised:
  the documentation tree is not a subject for this suite, and `context_docs/README.md` keeps its
  layout block current by hand. Both in `context_docs/test-suite-streamlining.md` §6.
