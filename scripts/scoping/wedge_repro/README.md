# Publication-density reproduction

`publication_density.py` drives the real write path — `write_year_shards` with its fork pool,
periodic catch-up, fork-phase watchdog (#181) and publication spacing (#183) — from several
coordinator processes at once into one icechunk repository seeded at production geometry, with
terminal marks interleaved at the campaign's rate. It measures, rather than infers, the quantity
the assembly wedges track: how far behind the branch tip each coordinator is at every catch-up.

## What it reproduces, and what it does not

It reproduces the **condition** — publications landing close enough together that an unspaced
coordinator's catch-up finds itself four or more snapshots behind — and lets each mechanism be
watched doing its job against real S3 and real snapshot volume. It does **not** inject the
icechunk hang itself: that hang has never reproduced at small scale (ten standalone attempts and
the 2026-08-30/31 harness all committed normally at every depth). The two `--wedge` arms
substitute a deterministic hang — a catch-up that never returns, a fork worker that never
returns — so the recovery paths run end to end in a multi-process, real-store setting.

## Arms

| arm | flags | what it shows |
|---|---|---|
| control | `--spacing off` | today's `main`: depth histogram at the ticks; expect ≥4 when finishes cluster |
| spacing | `--spacing on` | depth ≤2 at every tick; min publication gap from the store's own timestamps ≥ `PUBLICATION_SPACING_S` |
| wedged catch-up | `--wedge catch_up` | `CatchUpDidNotStopError` → re-home → the cell still publishes (#165), neighbours untouched |
| wedged worker, once | `--wedge worker --fork-stall-timeout-s 90` | `ASSEMBLY FORK PHASE STALLED`, stacks on stderr; the stalled partition is re-run once (`partitions_rerun`) and the cell publishes; neighbours unaffected |
| wedged worker, always | `--wedge worker_always --fork-stall-timeout-s 90` | the re-run stalls too: the cell fails as `ForkPhaseStalledError` (a deterministic fault) after exactly one re-run; neighbours publish |
| wedged publish | `--wedge publish --publish-step-timeout-s 60` | `PUBLISH WEDGED`: the child is killed (its stacks on stderr), the publish retried once on a fresh session, the cell publishes (`publish_retries`) |

Every arm reads its data back: each coordinator writes its own id into its chunks, so a merged
wrong fork or a neighbour's overwrite is a wrong number, not a silent pass.

## Running in the dev account

```
AWS_PROFILE=global-tessera-dev uv run python scripts/scoping/wedge_repro/publication_density.py \
    --run-id density-01 --coordinators 10 --seed-snapshots 300 --fork-seconds 120 --spacing on --cleanup
```

Outputs land in `temp/wedge_repro/<run-id>/`: `report.json` (the verdicts), one `.jsonl` per
coordinator (per-cell record with every tick's depth), one `.log` and one `.stderr` per process
(the stack dumps are on stderr). `--cleanup` deletes the store afterwards.

A credential-free local smoke: `--store-uri /tmp/x.icechunk --coordinators 3 --fork-seconds 6 --seed-snapshots 8`.

## Results, dev account, 2026-09-08

Real S3 (`global-tessera-embeddings-dev`), 120 zone groups at production layout, 300 (arms 1-2) or
100 (arms 3-4) terminal marks pre-seeded for snapshot volume, terminal marks interleaved every 3 s.

| arm | published / failed | max depth | ticks | ticks ≥ 4 | min publication gap | watchdog | stacks | re-homed | data intact |
|---|---|---|---|---|---|---|---|---|---|
| control, spacing off, 10 coordinators | 10 / 0 | **4** | 100 | 1 | 0.79 s | 0 | 0 | 0 | yes |
| spacing on, 10 coordinators | 10 / 0 | **1** | 110 | 0 | 12.1 s¹ | 0 | 0 | 0 | yes |
| wedged catch-up, 6 coordinators | 6 / 0 | 1 | 43 | 0 | 13.0 s¹ | 0 | 0 | **1** | yes |
| wedged worker, 6 coordinators, 90 s timeout | 5 / **1** | 2 | 53 | 0 | 13.2 s¹ | **1** | **4** | 0 | yes |

The unspaced control reached the hang's precondition (depth 4) against a real store; the spaced
arm never reached depth 2. The wedged catch-up re-homed and published. The wedged worker's cell
failed as `ForkPhaseStalledError` after the watchdog fired and dumped four thread stacks, while its
five neighbours published. Nothing was left in the bucket after `--cleanup`.

¹ Measured from slot acquisition at the time, so a 3 s commit ate into the gap. The hold is now a
full spacing after the last commit; re-run with that change (`density-spaced-v2`, `wedge-catchup-v2`):

| arm | published / failed | max depth | ticks | min publication gap | re-homed (report / log) | data intact |
|---|---|---|---|---|---|---|
| spacing on, 10 coordinators | 10 / 0 | 2 | 110 | **15.96 s** | 0 / 0 | yes |
| wedged catch-up, 6 coordinators | 6 / 0 | 2 | 45 | **15.90 s** | **1 / 1** | yes |

Every gap by the store's clock is now at or above the 15 s spacing, and the re-home reaches the
summary telemetry as well as the log.

### Recovery arms, dev account, 2026-09-08 (after the recovery landed in #181)

| arm | published / failed | partitions re-run | publish retries | max depth | min gap | watchdog | stacks | data intact |
|---|---|---|---|---|---|---|---|---|
| wedged publish, 6 coordinators, 60 s step timeout | 6 / 0 | 0 | **1** | 1 | 16.2 s | 0 | **1** (the killed child's) | yes |
| wedged worker, hangs once, 6 coordinators, 90 s | 6 / 0 | **1** | 0 | 2 | 16.1 s | 1 | 6 | yes |
| wedged worker, hangs always, 6 coordinators, 90 s | 5 / **1** | 0 | 0 | 2 | 16.1 s | 2 | 12 | yes |

The publish wedge was killed at 60 s and retried on a fresh session; the cell published and the
killed child's stack dump names the wedged frame. The once-only worker hang was recovered: the
stalled partition re-ran and the cell published. The deterministic worker hang failed exactly one
cell, as `ForkPhaseStalledError` after one re-run, with its neighbours unaffected. Nothing was left
in the bucket.
