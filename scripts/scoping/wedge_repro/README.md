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
| wedged worker | `--wedge worker --fork-stall-timeout-s 90` | `ASSEMBLY FORK PHASE STALLED`, stacks on stderr, the cell fails cleanly, neighbours publish |

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
