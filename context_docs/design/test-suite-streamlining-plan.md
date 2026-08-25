# Streamlining the test suite

**Status:** plan, not yet executed. Drawn up 2026-08-25 against `main` @ `76aeda9`, which is
the tip including PRs #133–#140 — the GDAL-forwarder, ranged-reader-status, radar-refusal and
GPU-hours work that landed that day. Every measurement below was taken on that commit.

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
tests/unit     120     3367   48387    unit.yml (3.12 + 3.13)
tests/integration 4       9      602    integration.yml
tests/parity      9       9     1139    integration.yml (fast), nightly.yml (slow)
tests/architecture 2       3       98    architecture.yml
tests/gpu          0       0        0    nothing — no tests exist
tests/slow         0       0        0    nothing — no tests exist
```

Unit suite: **84.3 s wall, 168 s CPU, 3366 passed / 1 skipped.** Line coverage of
`src/tessera_embeddings` is **83 %** (13,038 statements, 2,221 missed).

### The tiers as documented vs. as wired

`tests/README.md` describes a six-tier design. Three of the six are not wired the way it
says, and the drift is the source of most of §2's findings:

| Tier | README says it runs | What actually runs it |
|---|---|---|
| `unit/` | every PR | `unit.yml` (3.12 + 3.13) — correct. Note `downstream-smoke.yml` also runs this suite but is deliberately disabled to manual-only, pending a stable release of the private downstream consumer |
| `architecture/` | every PR | `architecture.yml` — correct |
| `integration/` | path-filtered PRs **+ nightly** | `integration.yml` on path-filtered PRs only. **Never nightly.** |
| `parity/` | flow-touching PRs + nightly | correct — but see §2.3 for what nightly actually selects |
| `slow/` | nightly + on-demand | **directory is empty.** Its documented occupant is an `xfail` stub in `parity/` |
| `gpu/` | inference-touching PRs only | **directory is empty, marker has zero uses, no workflow runs `-m gpu`** |

`tests/README.md` also states that unit tests are "< 30s each". One is 61 s (§3, item 1.1),
so the suite's own documented bound is being broken by the slowest test in it.

Separately, and *not* a defect: collecting `tests/unit` in a plain checkout produces 17
import errors for `ray`, `prefect`, `httpx` and `s3fs`. `tests/README.md` explains this — the
`inference` extra is optional because torch is large. CI installs `--all-extras` and never
sees it. Reproducing CI locally needs `uv sync --all-extras --frozen`.

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
deliberately — that ladder is a production decision and this plan does not touch it. Note
that at 61 s this single test breaks the "< 30s each" bound `tests/README.md` sets for the
unit tier, by a factor of two. The test
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

**Expected result: 84.3 s → ~25 s**, on every PR and every push to `main`, across both
Python versions in the `unit.yml` matrix — and on every local `uv run pytest`, which is where
it will be felt most.

### Phase 2 — Remove dead weight (zero behavioural risk)

Each item verified by exhaustive grep across `tests/`, not by inference.

**2.1 — Five unused conftest fixtures, 99 of 377 lines.** `sample_sar_data`,
`roi_mask_array`, `mock_roi_metadata`, `icechunk_s3_config`, `icechunk_s3_store_path`. Each
is referenced only inside `tests/unit/conftest.py`; the last two only reference each other,
so the set is closed. Zero references anywhere else — including as `usefixtures` strings and
`getfixturevalue` strings, which a name-based grep catches because fixture lookup is by name.

**2.2 — The `gpu` tier is empty, and the one real GPU test is filed outside it.** The `gpu`
marker has zero uses and `tests/gpu/` holds only `__init__.py` and a README. But that README
is not junk — it is a carefully written contract (tiny inputs, tiny model, compare GPU output
to a CPU reference, ≤ 2 CI minutes per run). It is an unimplemented design, not dead code,
so **do not just delete it.**

Meanwhile `tests/unit/test_inference_loop.py::TestRunInference::test_pipelined_matches_serial`
is gated by `@pytest.mark.skipif(not torch.cuda.is_available())` rather than
`@pytest.mark.gpu`, and its own docstring says it is "the only coverage of the pinned-buffer /
CUDA-event / two-deep-drain / backbone-stream-ordering path". No CI runner has a GPU, so it
skips everywhere — it is the `1 skipped` in every run. **That inference hot path has no CI
coverage at all today.**

**Decision, 2026-08-25: the gap is accepted.** A GPU CI runner is not available to this
project and will not be provisioned. The pinned-buffer / CUDA-event / stream-ordering path in
the pipelined inference loop is therefore **verified manually, not by CI**. What follows from
that is a documentation and ergonomics job, not a coverage one:

- **Keep the `skipif`; do not add `@pytest.mark.gpu` to this test.** The marker would be
  caught by the default `addopts` deselection and the test would vanish from the unit run
  entirely. The `skipif` keeps it collected, so it reports as `1 skipped` on every CI run —
  which is the only standing signal that the gap exists. Losing that is worse than the tidiness
  of a used marker.
- **State the gap where a reader will hit it.** `tests/README.md` and the class docstring
  should say plainly that this path has no automated coverage and is checked by hand. A skip
  that nobody explains reads as "not applicable here", which is how an accepted gap turns
  into an assumed one.
- **Make the manual run one documented command**, so "verified manually" is a thing someone
  can actually do on any CUDA machine:
  ```bash
  uv sync --all-extras --frozen
  uv run pytest tests/unit/test_inference_loop.py -k pipelined -v
  ```
- **Name when to run it**: after any change to the pipelined loop, the actor pool, or the
  chunk-staging path, and once before a campaign starts. Manual verification with no trigger
  attached to it does not happen.
- **Keep `tests/gpu/` and its README.** With the decision made, that directory is a dormant
  spec rather than a plan-in-progress — it describes what a GPU tier would look like if a
  runner ever appears. Say that in the README so it is not mistaken for work in flight, and
  leave the `gpu` marker declared in `pyproject.toml` so the README's instructions stay valid.

Consider adding this to `context_docs/decisions/` as an ADR. It is a standing decision with a
consequence a future reader will trip over ("why is the CUDA path untested?"), and the ADR
tree is where this repo answers that kind of question.

**2.3 — The nightly workflow runs one unimplemented test. — DONE 2026-08-25.** `nightly.yml` fires daily at
06:00 with a 120-minute timeout and runs `pytest tests/parity -m "parity and slow"`. That
selector matches exactly one test: `test_full_pipeline_parity`, whose body is `raise
NotImplementedError` under `@pytest.mark.xfail(strict=True)`.

The wiring is not an accident — `tests/slow/README.md` names the full plain-runner end-to-end
as that tier's canonical occupant, and this stub is it, written in `parity/` because it needs
both markers. So the honest description is: the nightly job is correctly pointed at a test
nobody has written yet. Recommend suspending the schedule (keep `workflow_dispatch`) until
the test is real, rather than paying a daily runner to confirm a placeholder is still a
placeholder.

**2.4 — `tests/README.md` needs correcting either way. — DONE 2026-08-25.** Its tier table
claimed integration runs nightly (it does not), and that `slow/` and `gpu/` are populated
(they are not). It also listed `downstream-smoke.yml` as a runner of the unit suite; that
workflow is deliberately disabled to manual-only, pending a stable release of the private
downstream consumer.

The table now describes what is wired, and a **Roadmap** section at the end of
`tests/README.md` carries the four undone things — the empty `slow/` tier and the suspended
nightly, the accepted GPU gap, the two orphaned tests from §2.5, and the 30-second bound
broken by §3 item 1.1 — so they can be scooped up rather than rediscovered. Status banners
were added to `tests/slow/README.md` and `tests/gpu/README.md` so the tier docs no longer
contradict the top-level one.

**2.5 — Two tests that no CI job ever runs.** The root `addopts` deselects
`integration`, `parity`, `slow` and `gpu`; `integration.yml` runs only `tests/integration`
and `tests/parity`. So these two, which live in `tests/unit/`, are executed by nothing:

- `tests/unit/test_provider_local_ray.py::test_local_ray_cluster_enters_and_exits` (`slow`)
- `tests/unit/test_provider_aws_dask.py:585` (`integration`)

They are not protecting anything today. Decide per test: move it to the tier whose job would
run it, or delete it. Do not leave it where it is — a test that never runs reads as coverage
and is not.

### Phase 3 — DRY the stubs — MOSTLY NOT DONE, and the estimate was wrong

The plan sized this at "27 duplicated stub definitions, 217 lines" and estimated ~175 lines
saved. **That measurement counted definitions sharing a NAME as duplication.** Comparing the
actual bodies by AST digest shows most are not duplicates at all:

| Stub | Definitions | Distinct bodies |
|---|---|---|
| `_Item` | 6 | **6** |
| `_item` | 9 | 7 |
| `_day_ds` | 3 | 3 |
| `_snapshots` | 4 | 3 |
| `_Roi` | 3 | 1 |
| `_gdal_logs`, `_mask_store`, `_completed_run`, `_stage_quickstart_roi` | 2 each | 1 each |

The `_Item` and `_item` stubs are **deliberately minimal**, each carrying exactly the surface
the code under test reads, and several say so in their own docstrings — "carrying only what
the baseline decision reads", "the minimum of a STAC item the date loop touches", "minimum
surface both ingests' items expose to the ownership filter". That is a feature: each stub
documents what the production function actually requires of its input. A shared superset stub
would delete that information, hand every test fields it does not use, and create a coupling
point where a change made for one file silently reaches nine.

The genuinely identical helpers are 2–5 lines each. Hoisting the five that live in
`tests/unit/` would save ~20 lines gross and cost ~26 in a new module header plus import
lines — a net loss, for the price of making a reader jump files to see what a four-line stub
is. **Not done.**

**One consolidation was worth it and is done:** `_stage_quickstart_roi` was byte-identical
across the two ROI-parity tests, both of which already import from `tests/parity/helpers.py`.
It moves there as `stage_quickstart_roi`, with the duplicated `FORCE_CRS` constant. No new
module, and the parity suite is green after the move (6 passed, 2 skipped, 121 s).

**The finding to carry forward: this suite is not repetitive.** Reducing its line count is not
an available lever, and the review's own first pass overstated it by measuring names instead
of bodies.

### Phase 4 — the source-text assertion pass

45 functions across `tests/` read source with `read_text`, `ast.parse` or `inspect.getsource`.
Read one at a time against what each is really protecting, they sort into four groups.

**Not source assertions at all — the §7.2 count was wrong.** The six
`test_provider_aws_ray.py` hits read the YAML that `_resolve_ray_config` **writes**. That is
its output, so those are ordinary behavioural tests and nothing needs doing. The same is true
of `test_context_docs_index.py` (reads a Markdown index), `test_public_api.py`'s doc check,
and the code-identity tests, whose whole subject is hashing source — reading it is the point.

**One was superseded by a linter, and is deleted.**
`test_source_read_resilience.py::test_placeholders_match_arguments` walked the AST of
`ingest/s1_roi.py` counting `%`-placeholders against arguments. Ruff's `PLE1205` and `PLE1206`
make exactly that check across every file in the repo. Both rules are now enabled in
`ruff.toml`, verified to fire nothing on `src/`, `tests/` and `scripts/` as they stand, and
verified **equivalent by mutation**: a too-many mutant and a too-few mutant in `s1_roi.py`
each fail the lint exactly as they used to fail the test. Net effect is wider coverage, one
module to all of them, for 18 fewer lines of test.

**The rest are project-specific invariants with no linter equivalent, and they stay put.**
Things like "no Prefect import outside the orchestration subtree", "every caller of
`apply_roi_mask` supplies the mask", "no ingest module builds its own retry policy". Each
encodes a real past incident, and each already runs on every PR in `unit.yml`. Moving them to
`tests/architecture/` would relocate the same hand-rolled AST walks into a folder whose job
runs on the same trigger — churn with no gain, and a real chance of losing context in the
move. The one thing that would genuinely improve them is expressing them as rules in
`src/tessera_embeddings/architecture_tests/`, which is production code and out of scope here.

**The pass turned up an asymmetry, now ruled on.** Two of these rules read only
`ingest/s1_roi.py`, the radar path:

- `test_placeholders_match_arguments` — moot, since the linter covers both sensors.
- `test_every_informational_line_carries_the_roi` — "a line without `roi=` cannot be tied to
  a cell: the log stream is a task id". `ingest/s2_roi.py` has 12 informational lines and 6
  carry no ROI, so widening the test would fail today.

**Ruling 2026-08-25: radar-only is correct, keep it.** Attribution earns its cost where dates
are actually abandoned, and radar is that path — `s1_roi.py` carries 25 mentions of data
loss, giving up and skipping against `s2_roi.py`'s 7. The optical path is the more forgiving
one, so a log line there is far less likely to be the only surviving record of a lost date.

Note for anyone reading the test later: the ``roi=`` it demands is the **region of interest**,
the cell being processed, taken from the ROI Zarr's filename. It is unrelated to the AWS
region locality that decides which copy of a granule to read — an easy collision, since both
are called "region" in this codebase and radar is genuinely the in-region-only sensor.

**One narrower observation left open.** Four of the six unattributed optical lines are
progress telemetry, where the run context is already obvious. The other two are warnings:

```
s2_roi.py:485  ROI has no live pixels — every date will fail the coverage gate
s2_roi.py:652  Load failed on asset-incomplete STAC item(s): %s
```

Those two are the ones where the radar argument transfers — a warning you would want to trace
to a cell. Naming them is a change to `src/` and out of scope here; recorded so the decision
is a decision rather than an omission.

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

## 5. Outcome — MEASURED, all phases landed

| | before | after |
|---|---|---|
| unit suite wall time | 84.3 s | **24.5 s** (3.4×) |
| tests, all tiers | 3398 | 3402 |
| tests deleted | — | **1**, plus 1 moved and 1 superseded (§5.1) |
| test LOC | 49,953 | 49,796 |
| src lines covered | 10,817 | **10,817 — none lost**, by the §6 gate |
| daily CI runners doing nothing | 1 | 0 |
| tests no CI job ran | 2 | 0 |
| GPU inference path | uncovered, unstated | uncovered, **stated and runnable by hand** (§2.2) |

### 5.1 Every test movement, named

The S4 gate: collect all tiers with markers disabled, before and after, and diff the IDs.

**Removed (3)**
- `test_provider_aws_dask.py::TestSchedulerResourceLoggerOnCluster::test_registered_plugin_emits_on_real_scheduler`
  — **moved**, not deleted; reappears under `tests/integration/`.
- `test_provider_local_ray.py::test_local_ray_cluster_enters_and_exits` — **deleted.** The only
  genuine deletion in the whole exercise. Marked `slow` inside `tests/unit/`, so no CI job had
  ever run it; `test_imports.py` already imports the module, and the coverage gate confirms
  nothing was lost.
- `test_source_read_resilience.py::TestZeroDateOutcomeIsAttributable::test_placeholders_match_arguments`
  — **superseded** by ruff `PLE1205`/`PLE1206`, which check the same thing repo-wide (Phase 4).

**Added (4 tests, 6 IDs)**
- the moved scheduler-plugin test, in its new tier;
- `test_the_retry_ladder_is_the_one_production_pays_for` — pins the retry budget that Phase 1
  stopped waiting out;
- `test_a_longer_number_is_not_a_status_with_its_tail_ignored` (4 parametrised rows) — closes
  the untested regex guard from §7.1;
- one auto-parametrised row of `test_every_document_is_listed`, because this document exists.

### 5.2 What the exercise was actually worth

**The speedup is the whole prize.** 84.3 s to 24.5 s, on every PR, both Python versions, and
every local `uv run pytest`. Five of the six offending tests were *waiting*, not working.

**Line count was not an available lever, and the plan's own first estimate of it was wrong.**
157 lines net, against an estimate of ~300, and Phase 3's share of that estimate rested on
counting helper NAMES rather than comparing their bodies (§3, Phase 3). This suite is verbose
because it pins a great many behaviours in deliberately long names, not because it repeats
itself.

**Three CI gaps closed, one accepted.** A nightly runner that confirmed a placeholder; two
tests no job executed; a regex guard nothing exercised. The GPU path stays uncovered by
decision, and is now written down in the three places a reader will meet it.

**LOC reduction is a small lever here and the plan says so.** About 300 lines of 49,953,
roughly 0.6 %. The suite is not bloated with copy-paste; it is verbose because the behaviours
it pins are genuinely numerous and the naming is deliberately long. The 3.4× speedup and the
two closed CI gaps are the real wins. If a larger LOC cut is wanted, it has to come from
§4 — and §4 is the part I would not do.

**One gap is accepted rather than closed.** Per §2.2 the pipelined CUDA path stays outside
CI, because no GPU runner is available. Nothing in this plan changes that; what changes is
that it stops being silent. The honest reading of the "after" column is *84 % of the codebase
verified automatically, one hot path verified by hand on a documented trigger* — which is a
weaker guarantee than the coverage number alone suggests, and is written down here so the
number is not read as more than it is.

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

**7.2 — Tests that assert on source text. — DONE as Phase 4 (§3).** One test was replaced by a
ruff rule; the rest stay. The radar-only scope of the ROI-logging rule was **ruled correct**
on 2026-08-25 — attribution earns its cost where dates are abandoned, and radar is that path.
Two optical *warnings* remain unattributed and would be worth naming; that is a `src/` change
and is not made here.
