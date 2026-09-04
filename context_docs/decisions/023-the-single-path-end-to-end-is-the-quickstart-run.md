# 023 — The end-to-end path is verified by running the quickstart, not by an automated test

**Status:** Accepted (2026-09-03, repo owner). **Final — this is not an open item and is not to be
re-proposed.**

## Context

`tests/slow/README.md` named a **full plain-runner end-to-end** — rasterise the ROI, ingest
Sentinel-2 and Sentinel-1, run CPU inference, assemble — as its canonical occupant. That test
existed only as `tests/parity/test_full_pipeline_parity.py`, an `xfail(strict=True)` stub whose body
was `raise NotImplementedError`, filed under `parity/` because it needed both markers.
`nightly.yml` was pointed at exactly that stub, so a daily 120-minute runner spent its time
confirming that a placeholder was still a placeholder; its schedule was suspended on 2026-08-25.

The 2026-08 test-suite review recorded it as pending work. The 2026-09-03 follow-up then observed
that both stated blockers had lapsed and proposed building it.

> **One of those two observations was wrong, and it is recorded here so this ADR does not carry a
> premise its own supporting document has withdrawn.** The single-ROI path had genuinely been
> measured at about **three and a half minutes** end to end on a CPU laptop, against the "30+
> minutes" the tier README claimed. But "the upstream per-stage parity tests pass" was read off a
> `6 passed, 2 skipped` summary line and does not hold: `test_ingest_s1_roi_parity.py` does not run,
> for **two independent reasons that are easy to read as one**. It carries a credentials `skipif`
> that accepts **either** supported Earthdata Login form — `EARTHDATA_TOKEN`, or
> `EARTHDATA_USERNAME` **and** `EARTHDATA_PASSWORD` together — so it skips only when *neither* is
> present; refreshing the cassette does not remove that skip. Supply either and it starts, then
> `xfail`s, for the separate reason that its committed cassette predates the native CMR granule
> query ([ADR 009](009-native-cmr-granule-query.md), issue #45). The other skip is the adapter
> template, skipped on purpose. Measured both ways: **no credentials → `6 passed, 2 skipped`;
> credentials → `6 passed, 1 skipped, 1 xfailed`** — six comparisons either way. **So the S1
> precondition was never verified**, and clearing either reason alone would not have verified it. The decision below is unaffected — it rests on the manual check being sufficient, not
> on the stub being unblocked — but the proposal it declined was argued partly on a claim that does
> not hold.

**That proposal was declined.** The reason is that the verification it would automate already
happens: **running the quickstart on a laptop IS this test.** It exercises the same path, over the
same inputs, and a person looking at the result learns more than a green tick would.

## Decision

**There will be no automated full-pipeline end-to-end test. The end-to-end path is verified by
running the quickstart, by hand, and that is sufficient.**

Three things follow.

**The stub is deleted.** An `xfail(strict=True)` placeholder for work that will not be done is not a
neutral cost. It is a standing claim that the work is pending, and it caused exactly that: a review
in September 2026 re-proposed the test on the strength of the stub and the roadmap entry that
described it. Removing it removes the claim.

**`nightly.yml` is deleted.** Its only selector was `-m "parity and slow"`, which matched that one
stub and now matches nothing — a dispatched run would exit on "no tests collected". A workflow that
fails when triggered is worse than no workflow. If a slow test is ever written for another reason,
the file is four lines of boilerplate plus a selector and is in git history.

**`tests/slow/` stays, as a category rather than a plan.** The tier means **more than 30 seconds**
— the threshold `pyproject.toml` declares for the `slow` marker, and the complement of the unit
tier's rule that no test may exceed 30 s. What the tier no longer has is a canonical occupant
waiting to be written. The marker stays declared so the tier remains usable if something genuinely
slow ever needs a home.

## Rejected alternatives

**Build it, with Earthdata credentials as repository secrets.** This is what the follow-up
proposed. It buys a nightly signal on a path that a person already exercises, at the cost of
credentials in CI, a 120-minute runner, and a test whose failures would most often be the archive
rather than the code.

**Build it as a local-only instrument that skips in CI.** The parity tier already has that pattern
for credential-gated tests. But a test that always skips reports nothing while looking like
coverage, and here there is nothing to make visible anyway: the manual check is not a gap being
tolerated, it is the chosen method.

**Keep the stub as a design record.** The design is not lost: this ADR states it and the tier README
describes the shape. What the stub added over those was a false signal that it was queued work.

## Consequences

**The plain runner's end-to-end path has no automated coverage, by decision.**
`tests/unit/orchestration/runners/test_plain_runner_wiring.py` covers config precedence, the CLI,
the staging identity and the cleanup, with the domain calls mocked; the parity tests compare flows
against domain functions per stage and never invoke `run_plain`. **A green suite is therefore weak
evidence about that path specifically** — the quickstart run is the evidence.

**When to run it:** after a change to `run_plain` or to the stages it drives, and before relying on
the single-ROI path for anything that matters. It takes about three and a half minutes on a laptop.
Follow [`docs/quickstart.md`](../../docs/quickstart.md) rather than improvising the command — it
carries the `source .venv/bin/activate` step and the reason `uv run python -m` must not be used
(the subprocess it spawns kills Ray's GCS on macOS).

**Three things make the difference between a real check and a green-looking no-op.** The first two
come from the same property that makes the run pleasant to repeat — it resumes — and resume is
exactly what a verification run does not want:

* **Delete the input stores first.** Resume is per-store and per-date, not per-run: `s2_roi` reads
  the dates already committed and treats everything at or below the newest one as closed, and
  `s1_roi` narrows its query to the window after the last written date and can return without
  fetching anything. So a second run against a populated `/tmp/tessera/inputs` **skips the ingest
  stages entirely** and verifies nothing about them. `rm -rf /tmp/tessera/inputs` — or point
  `paths.inputs` somewhere fresh; the key is `paths`, and a `stores` section would simply be
  ignored — before a run meant to check an ingest change.
* **Delete the staging prefix too, if the last run did not finish.** Staging is removed only after
  `assemble` *returns* (`plain.py`), so a crashed or interrupted run leaves it behind on purpose,
  for the next attempt to resume onto. And `_staging_run_id` **deliberately excludes the inference
  code identity** — that is the documented single-ROI trade-off, not an oversight, and its own
  docstring states the consequence: *edit inference code between a crash and its retry and the
  retry reuses tiles the old code staged.* On the one-chunk quickstart that is the whole product,
  so the rerun can report success having executed none of the changed inference code. The prefix is
  logged at the top of every run — `Staging prefix for this run: … (delete it to force a clean
  re-inference)` — so copy it from there rather than guessing. **"Inference staging cleans itself
  up" is true only of a run that succeeded**, which is not the run you are most likely to be
  retrying.
* **Pin the device.** `examples/quickstart/config.yaml` ships `device: auto`, which resolves to one
  GPU whenever `torch.cuda.is_available()`. On a GPU box the run therefore exercises the CUDA path,
  not the CPU path it is being cited for. Set `device: cpu` when the CPU path is what you mean to
  verify.

The blunt version of all three: **for a verification run, start from nothing** — fresh inputs, no
staging, device stated. Resume is a feature of the ordinary run, not of this one.

**This is one of two things in this repository verified by hand rather than by CI**, the other
being the pipelined CUDA path, which has no GPU runner to run on (`tests/README.md`, Roadmap 2).
Both are deliberate, both are cheap to run, and neither is a placeholder for automation that is
coming.

## Related

- `tests/slow/README.md` — what the tier means now
