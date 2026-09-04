# Reorganising `src/` — what moved, what is held, and why

An audit of all 49,170 lines of `src/tessera_embeddings/` (128 files, 30 of them added since
mid-July) for modules and symbols placed in the wrong file during feature work. Seven sections were
audited in parallel against the symbol index, the inbound-reference graph and the seven
architecture rules.

**Scope rule for the executed work:** move a file, move a symbol verbatim, update the imports those
force. Nothing else. Anything needing a real code edit is listed in §4 and is not in the PR.

## 1. The constraint that decided most of this

`config/code_identity.py::source_identity` hashes each file's **package-relative path** alongside
its contents. A pure move therefore changes the digest even though behaviour is identical.

Two fingerprints are built that way:

| fingerprint | seeded from | what a change costs |
|---|---|---|
| `ingest_code_identity()` | `ingest/s1_roi.py`, `ingest/s2_roi.py` | `IngestManifest.REQUIRED_TO_APPEND` refuses to append to any mosaic recorded under the old identity, unless an operator opts in |
| `inference_code_identity()` | `inference/`, `config/inference.py` | staged tiles from the old code are not reused, so staged-but-unassembled work is re-inferred |

Their import closures cover **66 of the package's 128 files** — every file in `inference/`, 18 in
`ingest/`, 12 in `storage/`, 11 in `config/`, and both root modules:

```
              frozen   movable
config            11         2
inference         23         0     <- the whole directory is a declared source
ingest            18         3
storage           12         1
orchestration      0        31     <- outside both closures
profiling          0        10
providers          0         9
(root)             2         0
```

So `orchestration/` is the only section where a move is free. **Three of the nineteen in-scope
proposals are there; the other sixteen each touch a frozen file** and are held pending a decision
on whether a fingerprint change is acceptable now (§3).

## 2. What moved

Nineteen in-scope moves were proposed. **Eleven were made; five were declined on their merits;
three were held only by §1 and are now unblocked but not yet scheduled.**

**Orchestration** — the three that are free of §1 entirely:

1. `_check_completed` → `flows/_child_runs.py` from `flows/tessera_full_pipeline.py`. A shared
   "did this child run finish?" predicate parked in the single-ROI master flow, which the three
   campaign flows imported for that one function and nothing else. Three module edges → zero.
2. `flows/_fleet_gate.py` moved down from `prefect/` — the seventh shared private flow helper,
   and the only one a level above the other six.
3. `get_task_runner_for_cluster` → `flows/_dask_lifecycle.py`; the 17-line `_dask_runner.py` is
   gone. `_ray_lifecycle.py` is **not** merged with it: disjoint consumers, different substrates.

**Across sections**, once the fingerprint cost was accepted:

4. `inference/conventions.py` → `storage/conventions.py` (398 L). Store metadata, proposed
   independently by two auditors; removes `storage/`'s only runtime upward import.
5. Three exceptions — `DuplicateDateError`, `InconclusiveStoreProbeError`,
   `StoreHoldsCommittedDataError` — from root `errors.py` into `storage/zarr_store.py`, their only
   raiser. Measured first: those are the only three of the module's eight with a single consumer,
   and ten other modules already keep their single-consumer exceptions locally.
6. **`storage/time_axis.py`** gathers nine symbols that answered "how does a store encode time?"
   from three different files — the general store module, the *spatial* zone-grid module, and the
   all-fill seeder. Callers were reaching into two of them on adjacent lines. 25 files repointed.
7. **`inference/read_plan.py`** takes 17 symbols out of `actors.py` (1,663 L → 1,325 L): the
   strip/crop/prefetch arithmetic, which has no Ray, no torch and no actor state, and which four
   other modules already cited by name in their own prose.
8. `download_checkpoint` and three helpers → `models/builder.py`, which already owns
   `load_checkpoint`.
9. `inference/lifecycle.py` folded into `runner.py` and deleted — 133 lines, one public function,
   a docstring naming its sole importer.
10. `StorageOptions` and `resolve_storage_options` → `storage/zarr_store.py`, joining the producer
    half of the same credential-freshness contract.
11. `orchestration/prefect/README.md`'s tree, which listed none of the seven private helpers.

**Declined on their merits**, not blocked:

- `global_store_config()` out of `zarr_store.py` — splits the `RepositoryConfig` builder family,
  and the better answer (a `storage/repo_config.py` holding all three) is blocked behind the
  `zarr_store` split. Doing it now would be undone.
- The coverage-record consolidation — trades one separation for another by splitting
  `read_skip_records` from its two consumers.
- Renaming `inference/profiling.py` — resolves a name collision with `profiling/`, but the
  placement is already right and a rename of an imported module is churn.
- The `roi_processing` retry symbols — the auditor named a cheaper alternative (widen that
  module's docstring), so this is a question of reading, not a defect.

**Four changes to non-import source lines**, each forced by a move and each surfaced by the
symbol-identity check rather than slipped through: `empty_store.__all__` lost a name that moved
out; `read_plan.py` defines its own `logger`, being a new module whose code logs; and two
docstrings carry a `:func:` cross-reference to a symbol that moved
(`providers/aws/credentials.py`, `inference/runner.py`).

## 3. A live bug, found while establishing the above

**`ingest_code_identity()` returns a different value on macOS than on Linux.**
`first_party_import_closure` resolves a candidate module to a file with `Path.exists()`.
`from .providers import PROVIDERS` offers `config.providers.PROVIDERS` as a candidate, and on a
case-insensitive filesystem `config/PROVIDERS.py` exists — it resolves to `providers.py`. The digest
hashes path strings, so the extra entry moves it.

Measured on a dev laptop:

| fingerprint | closure here | phantoms | digest here | digest on Linux |
|---|---|---|---|---|
| `ingest_code_identity()` | 34 | 1 (`config/PROVIDERS.py`) | `81011e3965a35cb5` | `e33a72eacd58a56a` |
| `inference_code_identity()` | 45 | 0 | `70f4d3036d83ccbe` | `70f4d3036d83ccbe` |

**Only the ingest one is affected**; the staged-output closure contains no uppercase candidate and
already agrees across platforms.

Consequence: a mosaic started from a laptop and appended to from the fleet is rejected as built by
different code. **The fix only moves the macOS value**: filtering candidates to case-exact directory
entries makes a laptop compute `e33a72ea…`, which is what Linux computes today — so no mosaic the
fleet has already built changes identity. Not done here; it is a code change, in the mechanism that
gates everything else.

## 4. Held: sixteen moves and the deeper work

Sixteen in-scope moves are held only by §1. The substantive ones, by confidence:

- **`inference/conventions.py` → `storage/conventions.py`** (398 L). Proposed independently by the
  inference and storage auditors. It is store metadata; moving it removes `storage/`'s only runtime
  upward import. **Also breaks `yield-embeddings/src/yield_embeddings/domain/coarsen.py:68`.**
- Three exception classes (`DuplicateDateError`, `InconclusiveStoreProbeError`,
  `StoreHoldsCommittedDataError`) from root `errors.py` into `storage/zarr_store.py`, their only
  raiser and catcher — matching the convention seven other modules already follow.
- Two private subjects buried in `inference/actors.py`: the strip/read planner (four modules
  already reach for it as `actors.<name>`) and the checkpoint downloader.
- Nine time-axis symbols in `storage/` into a new `storage/time_axis.py`; `global_store_config()`
  out of `zarr_store.py`.
- Retry/failure-context symbols out of `ingest/loader_failures.py`; `StorageOptions` out of
  `ingest/roi.py`.

Deeper items (real code edits, a separate PR): splitting `providers/aws/ray.py` (1,316 L, three
subjects) and `dask.py` (979 L, four); splitting `zarr_store.py` (1,965 L); separating
`config/fault_injection.py`'s schema from its runtime half; and a **naming pass on "code
identity"** — five distinct fingerprints of three different things (installed build, staged-content
source hash, campaign-resolved AMI/tarball identity) all share that name across `config/`,
`storage/` and `run_global_campaign.py`. Every one is correctly placed; the confusion is lexical.

## 5. Paths that must not move, whatever else does

- `inference/dataset.py`, `orchestration/prefect/flows/fill_zones_sequential.py` and
  `storage/global_store.py` — `yield-embeddings/scripts/verify_shipped_code.py` checks a shipped
  tarball for those three exact `(path, symbol)` pairs.
- `ingest/solar_days.py` and `ingest/auth.py` — exact-path entries in architecture-rule allowlists.
- Any **flow** module: this repo has no `prefect.yaml`, so deployment entrypoints are named in
  `yield-embeddings` and a moved flow breaks it with no in-repo signal.

## 6. Sections needing nothing

`profiling/` and `config/` are coherent; every candidate adjudicated to "correctly placed".
`inference/profiling.py` in particular **must not** move into `profiling/inference/`: it is
production instrumentation called from the inference hot path, and the move would make
`inference/inference.py`'s import of it violate `no-profiling-imports-outside-profiling`.
