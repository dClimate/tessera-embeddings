# Streamlining the test suite

**Status:** plan, not yet executed. Drawn up 2026-08-25 against `main` @ `76aeda9`.

The suite has grown across many PRs and nobody has looked at it as a whole. This plan says
what to change, what to leave alone, and — most importantly — what has to hold true before
any change lands.

Every measurement below was taken on this machine (10 cores, `-n auto`) against `main`. The
two headline fixes were prototyped and measured, not estimated; where a number is an
estimate it says so.

---

## 1. What the suite looks like today

```
             files    tests    LOC     runs in CI as
tests/unit     120     3367   48387    unit.yml (3.12 + 3.13), downstream-smoke.yml
tests/integration 4       9      602    integration.yml
tests/parity      9       9     1139    integration.yml (fast), nightly.yml (slow)
tests/architecture 2       3       98    architecture.yml
tests/gpu          0       0        0    nothing — no tests exist
tests/slow         0       0        0    nothing — no tests exist
```

Unit suite: **84.3 s wall, 168 s CPU, 3366 passed / 1 skipped.** Line coverage of
`src/tessera_embeddings` is **83 %** (13,038 statements, 2,221 missed).

Two things I expected to find and did not:

- **Collection is not slow.** It looked like 26 s on the first run; that was cold bytecode
  compilation. Warm, it is 2.2 s. No action.
- **There is very little copy-paste between tests.** An AST scan for near-identical test
  bodies inside each file found only 32 collapsible tests across all 120 files, and most of
  those pairs are meaningfully distinct (ascending vs descending, S1 vs S2). A parametrise
  sweep is not worth doing.

---

## 2. Safety model

The suite is the only thing standing between a code change and a corrupted global campaign.
This plan is therefore constrained hard:

**S1 — Test files only.** Nothing under `src/` is touched. Where the review found a
production-code concern it is written up in §7 for a separate decision, not folded in here.

**S2 — Coverage equivalence gate.** Before-and-after, the set of *executed lines* in
`src/tessera_embeddings` must be identical or a superset. Not the percentage — the actual
line set, per file. `scripts/test_coverage_gate.py` (§6) produces and diffs the artefact.
Any line that loses its last covering test is a blocker, and the diff names it.

**S3 — Mutation spot-check.** Coverage equivalence proves a line still *runs*; it does not
prove anything still *asserts* on it. So for any file where tests are deleted or merged, pick
the behaviours those tests named, mutate the production line behind each one, and confirm the
surviving suite goes red. §4 is the worked example of why this gate exists.

**S4 — Every removed test is named.** Test count goes from 3367 to a number that is
accounted for line by line in the PR description, with a reason per test. No net-count
hand-waving.

**S5 — One phase per PR.** Phases 1, 2 and 3 are independent and land separately, so a
revert is cheap and blast radius is one category at a time.

---

## 3. The plan

### Phase 1 — Make it fast (no tests removed)

This is where nearly all the value is. **Six tests account for 59 s of the 84 s wall time.**
Deselecting exactly those six drops the suite to **24.9 s — a 3.4× speedup** (measured, not
projected). Nothing is deleted; each test keeps its assertions.

| # | Test | Now | Cause | Fix |
|---|------|-----|-------|-----|
| 1.1 | `test_source_read_resilience.py::test_a_permanent_read_still_surfaces` | **61.1 s** | Calls the real `source_read_retrying` ladder — 8 attempts with 7 exponential sleeps, which is ~61 s of genuine `time.sleep` | Neutralise the sleep on the `Retrying` object in the test's `_read` helper |
| 1.2 | `test_scheduling.py::TestReplace::test_replace_creates_new_actor_and_marks_initializing` | 10.1 s | Calls `pool.replace()` without patching `ray.kill` | Hoist `patch.object(_sched_mod.ray, "kill")` into the existing autouse fixture |
| 1.3 | `test_pipeline.py::test_abandoning_the_generator_does_not_wait_on_the_in_flight_preparation` | 10.0 s | Two worker threads each block on `release.wait(5)` | Signal `release` before joining, or cut the timeout |
| 1.4 | `test_scheduling.py::test_replace_carries_credentials_and_region_to_new_actor` | 9.1 s | Same as 1.2 | Same as 1.2 |
| 1.5 | `test_assembly.py::TestDetectStagedChunkSize::test_uses_first_available_chunk` | 8.8 s | Writes a real 2000×2000×128 quantised chunk to detect a chunk size | Shrink the `ChunkSpec` extent and assert the smaller size |
| 1.6 | `test_catalogue_refusal.py::TestTheFailingRequestIsNamed::test_a_refusal_before_any_search_is_named_as_the_catalogue_root` | 6.0 s | Retry/backoff on the catalogue-root path | Neutralise the sleep as in 1.1 |

**1.1 in detail.** `SOURCE_READ_ATTEMPTS = 8` with `wait_exponential(multiplier=1, min=2,
max=15)` is about 61 s of backoff, and the docstring in `roi_processing.py` says so
deliberately — that ladder is a production decision and this plan does not touch it. The test
does not need to *experience* the ladder to assert that a permanent read still raises.
Verified:

```python
r = source_read_retrying(log)
r.sleep = lambda _seconds: None   # tenacity calls this between attempts
```

Attempt count stays 8, the `wait_exponential` policy object is untouched, elapsed time goes
from 61 s to 0.000 s. If we also want the ladder's *duration* pinned, assert on the policy
parameters directly — that is a stronger test than sitting through it, and it is free.

**1.2 / 1.4 in detail — this one is a latent hazard, not just slowness.** The file already
carries a comment explaining that an unpatched `ray.kill` from a test silently boots a real
local Ray cluster, which hashes and uploads the whole working directory, and that this once
ate ~60 GB of RAM. The shared `_do_replace` helper patches it. These two tests call
`pool.replace()` directly and do not. Prototyped the fix and measured the file:

```
before:  97 passed in 4.08s
after:   97 passed in 0.85s
```

Checked for vacuity, because a file-wide `ray.kill` patch could silence the tests that assert
on kills. It does not: every such test opens its own `patch.object(...)` which nests inside
and shadows the fixture's. Proved it by mutating the production retire path to not call
`ray.kill` and confirming `TestRetireIdle::test_idle_actor_killed_after_grace` still fails.
Both the source mutation and the test prototype were reverted.

**Expected result: 84.3 s → ~25 s.** Two CI jobs (`unit.yml` × 2 Python versions,
`downstream-smoke.yml`) get most of that back.

### Phase 2 — Remove dead weight (zero behavioural risk)

Each item verified by exhaustive grep across `tests/`, not by inference.

**2.1 — Five unused conftest fixtures, 99 of 377 lines.** `sample_sar_data`,
`roi_mask_array`, `mock_roi_metadata`, `icechunk_s3_config`, `icechunk_s3_store_path`. Each
is referenced only inside `tests/unit/conftest.py`; the last two only reference each other,
so the set is closed. Zero references anywhere else — including as `usefixtures` strings and
`getfixturevalue` strings, which a name-based grep catches because fixture lookup is by name.

**2.2 — The `gpu` marker has zero uses**, and `tests/gpu/` contains only `__init__.py` and a
README. Either delete the marker and the directory, or write the test the README describes.

**2.3 — The nightly workflow runs nothing real.** `nightly.yml` fires daily at 06:00 with a
120-minute timeout and runs `pytest tests/parity -m "parity and slow"`. That selector matches
exactly one test: `test_full_pipeline_parity`, whose body is `raise NotImplementedError`
under `@pytest.mark.xfail(strict=True)`. A scheduled runner every day to confirm a
placeholder is still a placeholder. Recommend disabling the schedule until the test is real
(keep the file — it is a legitimate `xfail` placeholder, just not one worth a nightly).

**2.4 — Two tests that no CI job ever runs.** The root `addopts` deselects
`integration`, `parity`, `slow` and `gpu`; `integration.yml` runs only `tests/integration`
and `tests/parity`. So these two, which live in `tests/unit/`, are executed by nothing:

- `tests/unit/test_provider_local_ray.py::test_local_ray_cluster_enters_and_exits` (`slow`)
- `tests/unit/test_provider_aws_dask.py:585` (`integration`)

They are not protecting anything today. Decide per test: move it to the tier whose job would
run it, or delete it. Do not leave it where it is — a test that never runs reads as coverage
and is not.

### Phase 3 — DRY the stubs (moderate value)

**27 duplicated stub definitions, 217 lines.** Nine separate `_item` STAC-item factories, six
`_Item` classes, and exact clones of `_Roi` (×3), `_snapshots` (×2), `_gdal_logs` (×2),
`_day_ds` (×3). Consolidate into a shared `tests/unit/_stubs.py` — a module, not a conftest,
so it is imported explicitly and a reader can see where the stub came from.

Do this **after** Phase 1 and 2 and behind the S2 gate. The stubs differ in small ways and a
shared one that quietly changes a default is exactly the kind of change that makes a test
pass for a new reason. Estimated saving ~175 lines.

---

## 4. What NOT to do, and why

**Do not consolidate the read-failure cluster.** Six files touch `ingest/duplicates.py` and
`ingest/loader_failures.py` — `test_duplicate_granules.py` (189 tests, 2656 lines),
`test_loader_failure_attribution.py`, `test_read_failure_verdict.py`,
`test_source_read_resilience.py`, `test_catalogue_refusal.py`,
`test_read_failure_cause_over_dask.py`. It is the largest apparent consolidation target in
the suite and it looks compelling: pairwise coverage overlap runs 78–100 %, and
`test_read_failure_verdict.py` (360 lines, 164 tests) has **zero unique line coverage** —
every line it touches is touched by `test_duplicate_granules.py`.

That file is a **characterisation table**: a 20-row matrix pinning the verdict for each
failure cause, plus mutual-exclusivity checks and boundary cases like "a URL port is not
mistaken for a status" and "a signed URL does not outrun the pattern". It has no unique
coverage because it walks the same lines to assert *different properties* about them.
Deleting it on a coverage-subsumption argument would be a serious regression, and coverage
subsumption is precisely the argument that would have justified it.

Two mutation probes on `ingest/duplicates.py` settle the method question:

| Mutant | Caught by |
|---|---|
| Drop the `(?![\d/])` lookahead in `_HTTP_STATUS_RE` | **nobody** — survived all five files |
| Remove `408` from `_TRANSIENT_4XX` | `test_read_failure_verdict` **and** `test_duplicate_granules` |

So the overlap is real for some behaviours and the cluster has genuine gaps for others, and
neither coverage nor a single mutant can tell you which is which per test. The payoff for
merging is LOC only — these files are already fast — and the downside is losing the one
assertion that mattered. Leave the structure alone; Phase 1 and Phase 3 still apply to them.

**Do not do a parametrise sweep.** 32 collapsible tests repo-wide, most meaningfully
distinct. Not worth the churn.

**Do not chase collection time.** 2.2 s warm.

**Do not delete the assertion-free tests.** The AST scan flags 38 tests with no `assert` and
no `pytest.raises`. Nearly all are the legitimate does-not-raise pattern — `test_imports.py`
smoke-imports every module, `test_all_present_passes` calls a validator that raises on
failure. They are cheap and they catch circular imports. Leave them.

---

## 5. Expected outcome

| | before | after |
|---|---|---|
| unit suite wall time | 84.3 s | ~25 s |
| tests | 3367 | 3365 (2 orphans in §2.4, if deleted rather than moved) |
| test LOC | 49,953 | ~49,650 |
| src line coverage | 83 % | unchanged — enforced by the gate |
| daily CI runners doing nothing | 1 | 0 |

**LOC reduction is a small lever here and the plan says so.** About 300 lines of 49,953,
roughly 0.6 %. The suite is not bloated with copy-paste; it is verbose because the behaviours
it pins are genuinely numerous and the naming is deliberately long. The 3.4× speedup and the
two closed CI gaps are the real wins. If a larger LOC cut is wanted, it has to come from
§4 — and §4 is the part I would not do.

---

## 6. The coverage-equivalence gate

```python
# scripts/test_coverage_gate.py — run before and after, then diff
#   uv run pytest tests/unit -n auto -q --cov=src/tessera_embeddings \
#       --cov-report=json:before.json
#   ... make changes ...
#   uv run pytest tests/unit -n auto -q --cov=src/tessera_embeddings \
#       --cov-report=json:after.json
#   uv run python scripts/test_coverage_gate.py before.json after.json
import json, sys

def executed(path):
    files = json.load(open(path))["files"]
    return {(f, l) for f, v in files.items() for l in v["executed_lines"]}

before, after = executed(sys.argv[1]), executed(sys.argv[2])
lost = sorted(before - after)
if lost:
    print(f"BLOCKED: {len(lost)} source lines lost their last covering test")
    for f, l in lost[:50]:
        print(f"  {f}:{l}")
    sys.exit(1)
print(f"OK: {len(before)} lines covered before, {len(after)} after, none lost")
```

The `main` baseline is 10,817 executed lines across 124 files. The artefact takes ~90 s to
produce, so the gate costs one extra coverage run per phase.

Gate S3 has no script — it is a judgement call per file, and §4 is the worked example of
doing it properly.

---

## 7. Found during the review, needs a decision — NOT part of this plan

These touch `src/` and are deliberately excluded. Listed so they are not lost.

**7.1 — `_HTTP_STATUS_RE` has an untested guard.** The `(?![\d/])` lookahead in
`ingest/duplicates.py` stops a 4-digit number or a `403/`-style fragment being read as a
status. Removing it breaks no test in the suite. This is a **test gap, not a bug** — the
guard looks correct, nothing exercises it. Adding a row to the characterisation table in
`test_read_failure_verdict.py` is a test-only fix and could be folded into Phase 2 if wanted.

**7.2 — Tests that assert on source text.** About 20 files read `.py` files with
`Path.read_text()`, `ast.parse` or `inspect.getsource` and assert on the shape of the source
— 8 such assertions in `test_provider_aws_ray.py`, 5 in `test_source_read_resilience.py`.
Some are legitimately architectural ("the package hard-exits in exactly one place") and
belong in `tests/architecture/`. Others are structural assertions that a behavioural test
would cover better. Sorting them requires reading each one against what it is really
protecting; worth a separate pass, and it is test-only, so it could become a Phase 4.
