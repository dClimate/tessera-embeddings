# The test suite: what has been done to it, and what is left

**This is the one record here about the repository's own engineering rather than the data
pipeline.** Three rounds of work have landed on the test suite:

| round | what it did | when |
|---|---|---|
| **1 — streamlining** | 84.3 s → 24.5 s, three CI gaps closed, one accepted | 2026-08-25, PR #143 |
| **2 — regrouping** | flat `tests/unit/` → subject directories mirroring `src/` | 2026-09-03, PR #171, [ADR 015](decisions/015-regrouping-the-unit-tests.md) |
| **3 — follow-ups** | the two regressions round 1's gains had lost, the standing list re-assessed, and the end-to-end gap settled as [ADR 023](decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md) | 2026-09-03, §6 |

**Read §1 for what the suite is now, §6 for what round 3 changed, and §7 for the standing list —
most of which is closed, and two entries of which are closed *by decision* rather than by being
built.** Only one of those two is an ADR; the other, the CUDA gap, lives in `tests/README.md`
Roadmap 2 (§6.3).
§2–§5 are round 1's method and rulings, kept because they are what a fourth round would need: the
safety model, the coverage gate, the things that were measured and ruled *not* worth doing, and why.

---

## 1. What the suite is now

Measured 2026-09-03 on **this branch after round 3** (10 cores, `-n auto`) — not on its parent
`main` @ `692655d`, which still carries the deleted documentation-index test and so has 134 unit
files rather than 133.

**Recount it rather than trusting the LOC column**, which moves whenever a test file is edited —
it drifted by 11 lines within this PR when a docstring grew:

```bash
git ls-files 'tests/unit/**/*.py' 'tests/unit/*.py' | xargs wc -l | tail -1
```

```
                  files   tests    LOC    runs in CI as
tests/unit          133    3675   54136   unit.yml (3.12 + 3.13), with coverage
tests/integration     8      —     1108   integration.yml, path-filtered PRs
tests/parity         10      —     1152   integration.yml as -m "parity and not slow"
tests/architecture    3       3       98  architecture.yml
tests/slow            1       0        0  nothing, and none planned — ADR 023 (§7.1)
tests/gpu             1       0        0  nothing — no GPU runner exists (Roadmap 2)
```

**Unit suite: 24.5 s wall, 3,674 passed / 1 skipped** (3,675 collected). Line coverage of `src/tessera_embeddings` is
**84%** (13,950 statements, 2,205 missed). The slowest single test is **3.3 s**, so the tier's
documented "no test should exceed 30 s" bound holds with an order of magnitude to spare.

The `1 skipped` is the pipelined CUDA path, and it is deliberate: no GPU CI runner is available to
this project and one will not be provisioned, so that path is verified by hand. The setup, the
trigger, and why the test keeps its `skipif` rather than the `gpu` marker are in `tests/README.md`
Roadmap 2.

### The layout, after round 2

`tests/unit/` is grouped into **subject directories mirroring `src/tessera_embeddings/`**, so an
edit to one subsystem can be verified without running the whole 3,675. Measured with `-n auto`: the
directories are 84–1,387 tests and run in 3–9 s against the full suite's 23 s, with about three of
those seconds being startup that no subset avoids. `tests/README.md` is
the authority on where a new file goes; ADR 015 carries the reasoning and the two departures from a
strict mirror. Every `src/` subpackage that holds testable code has a matching directory, and the
ten files that stay at `tests/unit/` root are named in `tests/README.md` and should not grow.

**Round 1's §1 described a flat `tests/unit/` of 120 files.** That is no longer the shape of the
thing, which is why this section was rewritten rather than annotated.

### Two things worth not re-discovering

**Collection is not slow.** It looks like ~26 s on a first run; that is cold bytecode compilation.
Warm, it is ~2 s. Do not chase it.

**The suite barely repeats itself — 18 tests in 8 groups share an identical body**, out of 3,675.
Round 1 found 32 collapsible tests across 120 files and judged most meaningfully distinct; that
holds, and the same is true of these.

> **Corrected 2026-09-03: this said "zero pairs", and that was a measurement error of exactly the
> kind round 1 warned about.** Round 1's first estimate compared helper *names* and called distinct
> things duplicates. Mine hashed each `FunctionDef` *including its name*, so two tests with
> different names and identical bodies came out distinct — the same mistake inverted. **Comparing
> bodies is the whole point; do not let the name into the digest.**

**They are retained deliberately, and the two shapes are worth knowing.** Some are one assertion
standing for two different *claims* — `test_a_harmonised_bucket_is_recognised` is the basic case,
while `test_the_live_catalogue_serves_the_configured_alias_keys` carries a docstring explaining why
`UNKNOWN` never fires in production. Same line, different property, which is §5's characterisation
argument in miniature. The rest are `parametrize` pairs whose bodies are identical *by construction*
because the inputs are the subject — `test_parse_cell_name_rejects_malformed` against
`..._rejects_invalid_coords`. Collapsing either shape trades a name that says what is being claimed
for a line of saved text.

**So line count is still not an available lever here**, and §5 is the evidence for the
strongest-looking counter-example.

---

## 2. Safety model

The suite is the only thing standing between a code change and a corrupted global campaign. Any
work on it is constrained hard, and these five held for rounds 1 and 3 alike:

**S1 — Test files only.** Nothing under `src/` is touched. Where a review finds a production-code
concern it is written up in §7 for a separate decision, not folded in.

**S2 — Coverage equivalence gate.** Before and after, the set of *executed lines* in
`src/tessera_embeddings` must be identical or a superset. Not the percentage — the actual line set,
per file. `scripts/test_coverage_gate.py` (§4) produces and diffs the artefact. Any line that loses
its last covering test is a blocker, and the diff names it.

**S3 — Mutation spot-check.** Coverage equivalence proves a line still *runs*; it does not prove
anything still *asserts* on it. For any file where tests are deleted or merged, pick the behaviours
those tests named, mutate the production line behind each, and confirm the surviving suite goes red.
§5 is the worked example of why this gate exists.

**S4 — Every removed test is named**, with a reason, in the PR description. No net-count
hand-waving.

**S5 — One phase per PR**, so a revert is cheap and the blast radius is one category at a time.

---

## 3. Round 1 — what it did, measured

| | before | after |
|---|---|---|
| unit suite wall time | 84.3 s | **24.5 s** (3.4×) |
| tests, all tiers | 3398 | 3402 |
| tests deleted | — | **1**, plus 1 moved and 1 superseded |
| src lines covered | 10,817 | **10,817 — none lost**, by the S2 gate |
| daily CI runners doing nothing | 1 | 0 |
| tests no CI job ran | 2 | 0 |

**The speedup was the whole prize, and five of the six offending tests were *waiting*, not
working.** The worst sat through a real 8-attempt retry ladder — about 61 s of genuine
`time.sleep`, breaking the tier's own 30 s bound by a factor of two — to assert that a permanent
read still raises. It now asserts the ladder's *shape* instead, which checks more and takes 0.5 s.
Two others called `pool.replace()` without patching `ray.kill`; see §6.2, because that one came
back.

**Every test movement, named** (the S4 gate: collect all tiers with markers disabled, before and
after, and diff the IDs):

- **moved** — the Dask scheduler-plugin test, from `tests/unit/` to `tests/integration/`, where a
  job actually runs it.
- **deleted** — `test_local_ray_cluster_enters_and_exits`, the only genuine deletion. Marked `slow`
  inside `tests/unit/`, so no CI job had ever run it.
- **superseded** — an AST walk counting `%`-placeholders against arguments, replaced by ruff
  `PLE1205`/`PLE1206`, which check the same thing across every file in the repo. Verified equivalent
  by mutation: a too-many and a too-few mutant each fail the lint exactly as they used to fail the
  test.
- **added** — the retry-ladder shape assertion, four parametrised rows closing an untested regex
  guard, and one auto-parametrised row because this document exists.

**Three CI gaps closed, one accepted.** A nightly runner that confirmed a placeholder; two tests no
job executed; a regex guard nothing exercised. The GPU path stays uncovered by decision —
`tests/README.md` Roadmap 2.

**Line count was not the lever, and round 1's own first estimate of it was wrong** — 157 lines net
against an estimate of ~300, because the estimate counted helper *names* rather than comparing their
bodies. See §5.

---

## 4. The coverage-equivalence gate

`scripts/test_coverage_gate.py` diffs two `--cov-report=json` runs by executed-line **set**:

```bash
uv run pytest tests/unit -n auto -q --cov=src/tessera_embeddings --cov-report=json:before.json
# ... make changes ...
uv run pytest tests/unit -n auto -q --cov=src/tessera_embeddings --cov-report=json:after.json
uv run python scripts/test_coverage_gate.py before.json after.json
```

A file can gain lines and lose others while the percentage holds, which is why this compares sets
rather than ratios. The artefact takes about 40 s to produce, so the gate costs one extra coverage
run per change. Round 3's run: **11,745 lines before, 11,747 after, none lost** — the
gate's verdict is *none lost*, not the total, which drifts by a line or two with parallel execution
order.

Gate S3 has no script — it is a judgement call per file, and §5 is the worked example of doing it
properly.

---

## 5. What NOT to do, and why

Each of these was measured and turned down. They are here so the same idea is not re-proposed on
the same evidence.

**Do not consolidate the read-failure cluster.** Six files touch `ingest/duplicates.py` and
`ingest/loader_failures.py`, and it is the largest apparent consolidation target in the suite:
pairwise coverage overlap runs 78–100%, and `test_read_failure_verdict.py` has **zero unique line
coverage** — every line it touches is touched by `test_duplicate_granules.py`.

That file is a **characterisation table**: a 20-row matrix pinning the verdict for each failure
cause, plus mutual-exclusivity checks and boundary cases like "a URL port is not mistaken for a
status". It has no unique coverage because it walks the same lines to assert *different properties*
about them. Two mutation probes settle the method question:

| Mutant in `ingest/duplicates.py` | Caught by |
|---|---|
| Drop the `(?![\d/])` lookahead in `_HTTP_STATUS_RE` | **nobody** — survived all five files |
| Remove `408` from `_TRANSIENT_4XX` | `test_read_failure_verdict` **and** `test_duplicate_granules` |

So the overlap is real for some behaviours and the cluster has genuine gaps for others, and neither
coverage nor a single mutant can tell you which is which per test. **Coverage subsumption is not
redundancy**, and it is precisely the argument that would have justified deleting the one file whose
assertions mattered.

**Do not DRY the stubs.** Round 1 sized this at "27 duplicated definitions, ~175 lines saved" by
counting definitions that shared a *name*. Comparing actual bodies by AST digest: `_Item` has 6
definitions and **6 distinct bodies**; `_item` has 9 and 7. They are deliberately minimal, each
carrying exactly the surface the code under test reads, and several say so in their own docstrings.
That is a feature — each stub documents what the production function requires of its input. A shared
superset stub would delete that information and create a coupling point where a change for one file
silently reaches nine. The genuinely identical helpers are 2–5 lines each; hoisting them would save
~20 lines and cost ~26. **Not done, and not to be re-proposed.**

**Do not do a parametrise sweep**, do not chase collection time, and **do not delete the
assertion-free tests** — the AST scan flags 38 with no `assert` and no `pytest.raises`, and nearly
all are the legitimate does-not-raise pattern. `test_imports.py` smoke-imports every module and
catches circular imports for almost nothing.

**Do not move the source-text assertions to `tests/architecture/`.** 45 functions across `tests/`
read source with `read_text`, `ast.parse` or `inspect.getsource`. Most are not source assertions at
all — they read YAML that the code under test *writes*, or a Markdown index, or (for the
code-identity tests) source whose hashing is the entire subject. The rest are project-specific
invariants with no linter equivalent — "no Prefect import outside the orchestration subtree", "every
caller of `apply_roi_mask` supplies the mask" — each encoding a real past incident and each already
running on every PR. Relocating them would move the same hand-rolled AST walks into a folder whose
job runs on the same trigger. The one thing that would genuinely improve them is expressing them as
rules in `src/tessera_embeddings/architecture_tests/`, which is production code and out of scope.

**Keep the ROI-logging rule radar-only.** `test_every_informational_line_carries_the_roi` reads only
`ingest/s1_roi.py`. Widening it to the optical path would fail today, and **radar-only is correct**:
attribution earns its cost where dates are actually abandoned, and `s1_roi.py` carries 25 mentions
of data loss against `s2_roi.py`'s 7. Note for anyone reading that test later — the `roi=` it
demands is the **region of interest**, the cell being processed, not the AWS region locality that
picks a granule copy. Both are called "region" here and radar is separately the in-region-only
sensor, so the collision is easy to make.

---

## 6. Round 3 — the follow-ups, 2026-09-03

Round 1's gains had partly eroded, and both causes were introduced *after* it. The unit suite had
drifted from 24.5 s back to **30.7 s**, and 13.3 s of that was two tests.

### 6.1 The documentation-index guard — deleted

`tests/unit/test_context_docs_index.py` asserted that `context_docs/README.md` names every file
under it and names nothing that is gone. **It is deleted: the documentation tree is not a subject
for the test suite** (repo owner, 2026-09-03). The layout block in `context_docs/README.md` is now
kept current by hand, and that file says so.

Two things about it are worth keeping, because both are general.

**It was the slowest test in the suite, at 10.3 s — a third of the whole run** — and for a reason
that had nothing to do with what it asserted. It excused an index row whose file exists elsewhere in
the repo, implemented as `REPO.rglob("*.md")`, which walks **2,975 markdown files** on a checkout
carrying git worktrees. Sixty-six of those are the repository's; the rest are `.venv` and
`.claude/worktrees/`.

**And it was wrong, in the direction that made it useless.** Because a file deleted on this branch
was still on disk under a worktree, the excuse fired for it — nine stale index rows stayed green
locally in PR #175 while CI, which has neither directory, would have failed on all nine. **A guard
that asks the filesystem "does this still exist anywhere?" is answered by build artifacts,
worktrees and caches.** `git ls-files` answers the question actually being asked, in 8 ms rather
than 9.5 s. That is the transferable part, and it applies to any check of this shape.

### 6.2 A unit test booted a real local Ray cluster

`tests/unit/providers/test_fleet_mix.py::TestTheRunnerLifecycle::test_a_drought_that_kills_the_actor_wait_has_still_asked`
cost ~3 s and printed `Started a local Ray instance`. Traced to its exact cause rather than guessed:

```
runner.py:252   ray.kill(actor)
ray/_private/auto_init_hook.py:21   auto_init_wrapper → auto_init_ray → ray.init()
```

The test makes `wait_for_actors` raise, but the actor factory has already handed back stand-ins, so
`run_inference`'s `finally` reaches `ray.kill(actor)` — and `ray.kill` is wrapped in Ray's auto-init
hook. **This is the third occurrence of a hazard this repository already documents**: Ray's init
hashes and uploads the whole working directory, which here includes every sibling worktree, and the
same hazard once ate ~60 GB of RAM across three concurrent runs. Round 1 fixed two instances of it
(`test_scheduling.py`, which now patches `ray.kill` in an autouse fixture); this one was written
afterwards and did not inherit the patch.

Fixed by patching `ray.kill` inside the helper's existing `with` block. Not file-wide, so nothing
that asserts on kills is silenced. **4.18 s → 1.17 s for the file**, and no Ray instance is started
anywhere in `tests/unit` afterwards — checked by running every candidate file with `-s` and grepping
for the banner.

### 6.3 The GPU decision stays in the test tree

Round 1 recommended promoting it to an ADR — *"a standing decision with a consequence a future
reader will trip over"*. One was written and then **removed**: the decision was already stated in
`tests/README.md` Roadmap 2, in the test class's own docstring, and here, and a fourth copy in the
decision tree earned nothing (repo owner, 2026-09-03).

What the exercise did surface, and what the Roadmap entry now carries, is that **the documented
manual command did not work.** `uv sync --all-extras --frozen` installs CPU torch on a CUDA machine
— the lockfile pins the CPU wheel registry for every platform — so `torch.cuda.is_available()` is
False and the test **skips**, reporting exactly what it reports in CI. `uv run` then re-syncs from
the lockfile by default, which would undo a manual CUDA install, so every invocation after it needs
`--no-sync`. And the node ID in the instructions named the wrong class. **A manual verification
whose command silently does nothing is worse than no instruction**, because it converts "unverified"
into "believed verified".

### Round 3, measured

| | before | after |
|---|---|---|
| unit suite wall time | 30.7 s | **24.5 s** |
| slowest single test | 10.3 s | **3.2 s** |
| unit tests that boot a real Ray cluster | 1 | **0** |
| src lines covered | 11,745 | **11,747 — none lost**, by the S2 gate |
| tests removed | — | **the documentation-index guard, and the end-to-end stub** (§6.1, §7.1) |

**No surviving test changed what it asserts.** The Ray fix is to how a test reaches its subject;
the two removals are subjects ruled out of the suite. The S2 gate reports no source line lost by
either, which is the S4 accounting: the documentation guard covered no `src/` line, and the stub
never ran.

---

## 7. The standing list

Five entries. **Two are closed by decision rather than by being built** (§7.1, §7.2) — the
end-to-end one as [ADR 023](decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md)
precisely so it is not re-proposed, the CUDA one in `tests/README.md` Roadmap 2. One is a `src/`
change deliberately not made here (§7.3); one is a guard considered and declined (§7.4); and one is
a non-finding recorded so it is not mistaken for one (§7.5). **Nothing on this list is queued
work.**

### 7.1 The full-pipeline end-to-end — CLOSED, and not by building it

**Decided 2026-09-03, and final: there will be no automated full-pipeline end-to-end test. Running
the quickstart by hand IS that verification.**
[ADR 023](decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md) carries it, including
the two alternatives that were declined.

The reasoning that led here is worth keeping, and so is the fact that **half of it was wrong**.
Round 1 recorded the item as pending on two stated blockers, and this round claimed both had lapsed:
that the upstream per-stage parity tests pass, and that the single-ROI path runs in **~3.5 minutes**
end to end rather than the "30+ minutes" five documents claimed.

**The second is right; the first is not, and review caught it.** `6 passed, 2 skipped` does not mean
S1 parity passes. `test_ingest_s1_roi_parity.py` does not run at all, for **two reasons that are
independent of each other** — a distinction review had to make three times before all three copies
of this claim stated it:

* a `skipif` on Earthdata credentials, accepting **either** `EARTHDATA_TOKEN` **or**
  `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` together, so it skips only when neither form is set;
* and separately an `xfail(raises=Exception)`, because the committed cassette was recorded against
  the old CMR-STAC search path while OPERA now resolves items through the native CMR granule API, so
  it no longer matches on replay (issue #45,
  [ADR 009](decisions/009-native-cmr-granule-query.md)). Re-recording trips the credential-safety
  guard at ~116 MB.

Supplying credentials therefore changes the *label*, not the verdict — measured both ways: **no
credentials → `6 passed, 2 skipped`; credentials → `6 passed, 1 skipped, 1 xfailed`**, six
comparisons either way. The other skip is the deliberately-skipped `adapter_template`, not a second
credential-gated arm. **So the S1 precondition was never verified, and "both blockers lapsed"
overstated the evidence by one.**

**The decision is unaffected and would have been the same either way** — it rests on the manual
check being sufficient, not on the stub being unblocked. But it is recorded because the proposal
that got overruled was argued partly on a claim that does not hold, and a reader coming back to this
should not inherit it.

**The `xfail(strict=True)` stub and `nightly.yml` are deleted**, and that is the operative half of
the decision rather than tidying. A placeholder for work that will not be done is a standing claim
that the work is pending, and it produced exactly that: this round re-proposed the test on the
strength of the stub and the roadmap entry describing it. `nightly.yml`'s only selector matched that
one stub, so after the deletion a dispatched run would have exited on "no tests collected".

`tests/slow/` and the `slow` marker stay, as a **category rather than a plan** — "this test is
genuinely slow" may need a home again, and there is no CI job for it today.

### 7.2 `run_plain`'s end-to-end path has no automated coverage — accepted

The consequence of §7.1, stated separately because it is the thing a reader needs to know rather
than the decision that produced it. `tests/unit/orchestration/runners/test_plain_runner_wiring.py`
covers config precedence, the CLI, the staging identity and the cleanup, all with the domain calls
mocked; the parity suite compares flows to domain functions per stage and never invokes `run_plain`.
**So a green suite is weak evidence about the path an outside user of this library is most likely to
take.** The quickstart run is the evidence — after a change to `run_plain` or the stages it drives.

**This is one of two things in this repository verified by hand rather than by CI**, and the two
should be read together: the pipelined CUDA path (`tests/README.md` Roadmap 2), which has no GPU
runner to run on, and this one, which is
[ADR 023](decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md). Neither is a
placeholder for automation that is coming, and **neither is to be re-proposed.**

### 7.3 Two optical warnings carry no ROI attribution

`s2_roi.py:443` ("ROI has no live pixels — every date will fail the coverage gate") and
`s2_roi.py:595` ("Load failed on asset-incomplete STAC item(s)") are warnings a reader would want to
trace to a cell, and the radar argument in §5 transfers to them even though the blanket optical rule
does not. **This is a `src/` change and is deliberately not made here** (gate S1); recorded so the
omission is a decision.

### 7.4 Considered and not proposed: a guard against the Ray-boot class

§6.2 is the third occurrence, so the obvious move is an autouse fixture that fails any unit test
which initialises Ray. It would be about ten lines and it would work.

**It is not proposed, because the standing direction on this repository is to reduce gates rather
than add them**, and this one guards a hazard whose cost is seconds of local wall clock plus a
memory risk that has materialised once in the project's life. The cheaper mitigation is already in
place: the cause is now documented at all three sites, and the detection is one command —

```bash
uv run pytest tests/unit -q -s | grep "Started a local Ray instance"
```

Worth running after any change that touches `inference/runner.py` or the actor pool. If a fourth
occurrence appears, that is the evidence for the guard, and this paragraph is what it supersedes.

### 7.5 Two large files, still not split

`tests/unit/assembly/test_assembly.py` (2,884 lines) and `tests/unit/inference/test_scheduling.py`
(2,392) are the largest in the suite. Round 2 put them in the right directories and deliberately did
not split them. Neither is slow and neither is duplicated (§1), so the only argument for splitting
is navigability — which the subject directories already improved. **No action proposed**; noted so
that "these files are large" is not mistaken for a finding.
