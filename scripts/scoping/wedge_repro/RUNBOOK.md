# Runbook: production-scale reproduction of the assembly wedge

Goal: provoke the **real** icechunk hang (`rebase → list_nodes → fetch_snapshot` never returning)
at production scale on dev S3, capture its **native stack** with py-spy, then run the identical load
against the fix stack and show it does not hang — or, if a wedge is injected or occurs, that it is
recovered without losing shards. Two arms, one driver, one box, one afternoon.

Everything below is executed by the operator (an agent). It changes no code. It runs in the dev
account only (658132200637); the scripts refuse any other account.

## 0. Ground rules

- `export AWS_PROFILE=global-tessera-dev AWS_REGION=us-west-2`; `aws sts get-caller-identity` must
  show 658132200637 before anything else.
- The box self-terminates after 8 h regardless. Run `teardown.sh` yourself at the end anyway.
- Never delete `scoping/prod-repro/results/`: that is the evidence.
- Bound every remote command; nothing here should block a terminal indefinitely.
- Report facts. No interpretation of root cause.

## 1. Provision (laptop, ~1 min + ~10 min bootstrap)

```
cd /Users/rbanick/dev/tessera-embeddings && git checkout dev/publication-density-harness
scripts/scoping/wedge_repro/provision.sh          # prints the instance id
until aws s3 ls s3://global-tessera-embeddings-dev/scoping/prod-repro/box/READY >/dev/null 2>&1; do sleep 30; done
```
If READY has not appeared after 20 min, read the bootstrap log via SSM (section 2) and report it.

## 2. Driving the box: SSM, non-interactive

Use `aws ssm send-command` with a bounded timeout and poll for output. Helper (bash function):

```
ssm() {  # usage: ssm <instance-id> "<command>" [timeout-s]
  local id=$(aws ssm send-command --instance-ids "$1" --document-name AWS-RunShellScript \
    --parameters "commands=[\"$2\"]" --timeout-seconds "${3:-600}" --query 'Command.CommandId' --output text)
  for i in $(seq 1 120); do s=$(aws ssm get-command-invocation --command-id "$id" --instance-id "$1" --query Status --output text 2>/dev/null); case "$s" in Success|Failed|TimedOut|Cancelled) break;; esac; sleep 5; done
  aws ssm get-command-invocation --command-id "$id" --instance-id "$1" --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
}
```
Sanity: `ssm $IID "tail -5 /var/log/repro-bootstrap.log; ls /opt/repro; /opt/repro/fix/.venv/bin/python -c 'import tessera_embeddings, icechunk; print(icechunk.__version__)'"`.

Long runs are started **detached** and polled; SSM must not own them:

```
ssm $IID "cd /opt/repro && nohup <venv>/bin/python prod_scale_repro.py <args> > results/<run-id>.out 2>&1 &"
```

## 3. Arm A — REPRODUCE, on `main` (pre-fix)

```
RUN=repro-main-01
ssm $IID "cd /opt/repro && nohup main/.venv/bin/python prod_scale_repro.py --arm reproduce --run-id $RUN \
  --coordinators 10 --cells-per-coordinator 2 --live-shards 1500 --n-workers 12 \
  --stagger-seconds 2 --marker-rate-s 3 --seed-target-snapshot-kb 180 --seed-cells 60 --seed-shards 100 \
  --stall-seconds 900 --stall-dumps 5 --results-dir results/$RUN \
  --s3-results s3://global-tessera-embeddings-dev/scoping/prod-repro/results/$RUN > results/$RUN.out 2>&1 &"
```
Expected timeline: seeding 20–40 min (it stops when the snapshot object reaches 180 KB; production's
mean is 246 KB — if the target is not reached by the cap, note the achieved size), then two rounds
of ten 1500-shard cells, each round ~20–30 min network-bound. Total ~1.5–2 h.

**Poll every 10 minutes**, and record each poll:
```
ssm $IID "tail -3 /opt/repro/results/$RUN.out; grep -h 'shards written' /opt/repro/results/$RUN/coord-*.log | tail -10; ls /opt/repro/results/$RUN/*.pyspy-* 2>/dev/null; free -g | head -2; uptime"
```
**The moment a `COORDINATOR N STALLED` line or a `coord-N.pyspy-*.txt` file appears**, fetch and
save the dump verbatim (it is also mirrored to S3 by the driver): the frames under `_rebase`,
`fetch_snapshot`, `list_nodes`, or anything in `icechunk` are the result this whole exercise exists
for. Also immediately run one extra dump yourself for corroboration:
`ssm $IID "py-spy dump --native --subprocesses --pid <coordinator pid>"` (the pid is in the STALLED line).

The run ends on its own; `results/$RUN/report.json` appears and is mirrored to S3. If it has not
ended 30 min after the last coordinator's expected finish, record the state and move on (the driver
kills stuck coordinators at its own deadline).

## 4. Arm B — FIXED, on the PR stack

Same load, same seeding target, the fix's spacing and recovery active:
```
RUN=fixed-01
ssm $IID "cd /opt/repro && nohup fix/.venv/bin/python prod_scale_repro.py --arm fixed --run-id $RUN \
  --coordinators 10 --cells-per-coordinator 2 --live-shards 1500 --n-workers 12 \
  --stagger-seconds 2 --marker-rate-s 3 --seed-target-snapshot-kb 180 --seed-cells 60 --seed-shards 100 \
  --stall-seconds 900 --stall-dumps 5 --results-dir results/$RUN \
  --s3-results s3://global-tessera-embeddings-dev/scoping/prod-repro/results/$RUN > results/$RUN.out 2>&1 &"
```
Poll the same way. Expected: no STALLED lines, no py-spy dumps, all cells published, `publish_retries`
and `partitions_rerun` zero, and every publication gap ≥ 15 s from the store's own timestamps.

## 5. If arm A did NOT hang

That is itself a result. Record it, then run **one** escalation and stop: arm A again with
`--cells-per-coordinator 3 --live-shards 2500 --stagger-seconds 1 --marker-rate-s 2`
(`RUN=repro-main-02`), which lengthens the fork phase and tightens the publication clustering.
Do not improvise further arms.

## 5b. Run 2 (2026-09-09): production snapshot weight, depth measured

Run 1 (sections 3-5, results in the README) never reached production weight: the snapshot object
sat at 127 KB in every arm because its size is the store skeleton plus ~110 B per published FILL,
not per shard, and the 60-fill cap added almost nothing. Run 1 also could not say whether the
depth-4 precondition ever occurred, because the driver did not record catch-up depth. Both are
fixed in the driver (`_recording_catch_up`, `--seed-pool-cells`, per-fill seeding). Arms run in
SERIES on one box: the concurrent attempt in run 1 hit an instance-profile credential failure.

Refresh the driver on the box first (the venvs are unchanged; only the driver moved):
```
ssm $IID "aws s3 cp --only-show-errors s3://global-tessera-embeddings-dev/scoping/prod-repro/box/prod_scale_repro.py /opt/repro/prod_scale_repro.py && grep -c _recording_catch_up /opt/repro/prod_scale_repro.py"
```
Expect `4` or more; `0` means the old driver.

Arm A, `main`:
```
RUN=repro-main-03
ssm $IID "cd /opt/repro && nohup main/.venv/bin/python prod_scale_repro.py --arm reproduce --run-id $RUN \
  --coordinators 10 --cells-per-coordinator 4 --live-shards 1500 --n-workers 12 \
  --stagger-seconds 2 --marker-rate-s 3 --seed-target-snapshot-kb 300 --seed-cells 2000 --seed-pool-cells 200 \
  --stall-seconds 900 --stall-dumps 5 --results-dir results/$RUN \
  --s3-results s3://global-tessera-embeddings-dev/scoping/prod-repro/results/$RUN > results/$RUN.out 2>&1 &"
```
Seeding now takes 60-90 min (≈1,600 one-shard fills at 2-3 s each; the log line
`seed fill N ... snapshot_max` shows the weight climbing every ten fills), then four rounds of ten
1500-shard cells, ~40 min. Poll as in section 3. The report gains `catch_up_depth` (ticks, max,
histogram, `at_or_above_4`) and `condition_reached`; a run with `condition_reached: false` did not
even present the precondition and must be reported as such, not as "no hang".

Arm B, the fix stack, the same flags with `--arm fixed --run-id fixed-03` in `fix/.venv`, started
only after arm A's `report.json` exists.

Escalation, ONLY if arm A ended with `condition_reached: true` and no stall: `RUN=repro-main-04`
with `--cells-per-coordinator 4 --live-shards 2500 --stagger-seconds 1 --marker-rate-s 2`, other
flags unchanged. Do not improvise further arms.

Budget: the box self-terminates 8 h after launch (this box: launched 01:27Z, terminates ~09:27Z).
A + B + escalation is about 6 h. If the schedule threatens the deadline, say so in a message
rather than shortening a run; only the operator (Robert) extends the box.

## 6. Collect, per arm

From `report.json`: `seed` (achieved `snapshot_max_bytes`), `cells`, `catch_up_depth`, `condition_reached`, `publication_gaps_from_store`,
`coordinators_stuck_at_teardown`, `pyspy_dumps`, `hang_captured`, `integrity`, `wall_s`. From the
logs: every `STALLED` line verbatim; for each py-spy dump, the first 60 lines verbatim, and separately
every frame line containing `icechunk`, `rebase`, `fetch_snapshot`, `list_nodes`, `commit`,
`merge`, or `tokio`. Any `Traceback` in the `.out` files verbatim. Wall-clock per arm.

## 7. Teardown (laptop)

```
scripts/scoping/wedge_repro/teardown.sh
aws s3 ls --recursive --summarize s3://global-tessera-embeddings-dev/scoping/prod-repro/results/ | tail -2
```
Confirm the instance is terminating and only `results/` remains. Then report.

## 8. Final report shape

One table: arm | hang captured (yes/no) | condition reached (ticks at depth ≥ 4 / ticks) | max depth |
stalled coordinators | py-spy dumps | published / failed | publish_retries | partitions_rerun |
achieved snapshot KB | min publication gap | integrity | wall.
Then the verbatim STALLED lines and dump excerpts, then tracebacks, then the teardown confirmation.
