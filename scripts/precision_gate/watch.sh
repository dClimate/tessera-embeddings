#!/usr/bin/env bash
# Watch precision-gate boxes until every one has finished, or any one fails.
#
# usage: watch.sh <s3 run prefix> <role>=<instance-id> [...]
#
# Each pass, for each box still running: its status file (ok ends it; not-passed ends it with a gate
# result that is not a pass; failed:* is a failure), its EC2
# state (stopped or terminated without a status is a crash), its uploaded log (a traceback, CUDA error
# or OOM kill is a failure), and its progress (log lines must keep growing: 20 minutes without a new
# line is a stall). Polls every 30 s for the first 10 minutes, when most failures happen, then every 2 min.
# Exits 0 when every box reports ok, 3 when every box has finished and some gate did not pass, 1 on the
# first failure.
set -uo pipefail
PREFIX=$1
shift
: "${AWS_REGION:?}"
declare -A ID LINES SEEN
not_passed=""
for pair in "$@"; do
  ID[${pair%%=*}]=${pair#*=}
  LINES[${pair%%=*}]=0
  SEEN[${pair%%=*}]=$(date +%s)
done
t0=$(date +%s)
log=$(mktemp)
trap 'rm -f "$log"' EXIT
while true; do
  running=0
  for role in "${!ID[@]}"; do
    status=$(aws s3 cp --quiet "$PREFIX/status/$role" - 2>/dev/null || true)
    case $status in
      ok) continue ;;
      not-passed)
        [[ " $not_passed " == *" $role "* ]] || { not_passed="$not_passed $role"; echo "GATE NOT PASSED $role"; }
        continue ;;
      failed*) echo "FAIL $role: $status"; aws s3 cp --quiet "$PREFIX/logs/$role.log" - 2>/dev/null | tail -30; exit 1 ;;
    esac
    running=$((running + 1))
    state=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "${ID[$role]}" \
      --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo unknown)
    case $state in
      pending | running | unknown) ;;
      *) sleep 20; [ -z "$(aws s3 cp --quiet "$PREFIX/status/$role" - 2>/dev/null)" ] && {
        echo "CRASH $role: box $state with no status file"; exit 1; } ;;
    esac
    now=$(date +%s)
    if aws s3 cp --quiet "$PREFIX/logs/$role.log" "$log" 2>/dev/null; then
      if grep -qE 'Traceback|CUDA error|illegal memory|out of memory|OutOfMemoryError|Killed|ERROR:' "$log"; then
        echo "ERROR $role:"; grep -nE -B3 -A12 'Traceback|CUDA error|illegal memory|out of memory|OutOfMemoryError|Killed|ERROR:' "$log" | head -40
        exit 1
      fi
      n=$(wc -l < "$log")
      if [ "$n" -gt "${LINES[$role]}" ]; then
        echo "$(date +%H:%M) $role ($state): $(grep -E 'PROGRESS|done |: [0-9]+s,|STEP|staged|ALL COMPLETE' "$log" | tail -1)"
        LINES[$role]=$n
        SEEN[$role]=$now
      fi
    fi
    if [ $((now - SEEN[$role])) -gt 1200 ]; then
      echo "STALL $role: no new log line for 20 min ($state)"
      exit 1
    fi
  done
  if [ $running -eq 0 ]; then
    [ -n "$not_passed" ] && { echo "ALL FINISHED; GATE NOT PASSED on:$not_passed"; exit 3; }
    echo "ALL OK"
    exit 0
  fi
  [ $(($(date +%s) - t0)) -lt 600 ] && sleep 30 || sleep 120
done
