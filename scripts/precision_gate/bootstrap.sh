#!/bin/bash
# User data for one precision-gate box. launch.sh fills the __PLACEHOLDERS__.
#
# Roles:
#   gpu-<card>     embed the lattice on this card (l40s, a10g, l4) with __VARIANTS__, one npz per tile
#   cpu-record     re-score ADR 043's recorded embeddings in this environment (the environment-only control)
#   cpu-rebase     score this run's embeddings, which the gpu-* boxes must already have written
#
# Every exit, success or failure, uploads the log and a status file (ok, or failed:<step>:rc=<n>) and
# shuts the box down; the log is also uploaded every minute, so a watcher sees progress and tracebacks.
# A hard shutdown is scheduled first, in case the trap itself never runs.
set -euo pipefail
RUN=__RUN__
ROLE=__ROLE__
CODE=__CODE__
OUT=__OUT__/$RUN
INPUTS=__INPUTS__
CHECKPOINT=__CHECKPOINT__
VARIANTS=__VARIANTS__
TILES="cpu-ab-uk cpu-ab-iowa fp8-ecuador fp8-punjab fp8-finland fp8-kansas"
LOG=/var/log/gate-$ROLE.log
exec > >(tee -a "$LOG") 2>&1
shutdown -h +__CAP_MIN__
export HOME=/root DEBIAN_FRONTEND=noninteractive
command -v aws || { apt-get update -q && apt-get install -y -q unzip curl; curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/a.zip && unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install; }

STEP=setup
finish() {
  rc=$?
  status=ok
  [ $rc -ne 0 ] && status="failed:$STEP:rc=$rc"
  echo "FINISH $status"
  aws s3 cp --quiet "$LOG" "$OUT/logs/$ROLE.log" || true
  echo "$status" | aws s3 cp - "$OUT/status/$ROLE" || true
  shutdown -h now
}
trap finish EXIT
(while sleep 60; do aws s3 cp --quiet "$LOG" "$OUT/logs/$ROLE.log" || true; done) &

STEP=env
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
mkdir -p /opt/te && cd /opt/te
aws s3 cp --quiet "$OUT/code/$CODE.tar.gz" /tmp/te.tar.gz
tar xzf /tmp/te.tar.gz
uv sync --frozen --extra inference --extra aws --python 3.12 -q
export TE_CODE_SHA=$CODE PY=/opt/te/.venv/bin/python

label_args() {  # tile -> "<labels file> <label key>"
  case $1 in
    cpu-ab-uk) echo "labels_uk.npz worldcover" ;;
    cpu-ab-iowa) echo "labels_iowa.npz cdl" ;;
    fp8-kansas) echo "labels_fp8-kansas.npz cdl" ;;
    *) echo "labels_$1.npz worldcover" ;;
  esac
}

gpu() {
  local card=${ROLE#gpu-}
  STEP=torch
  uv pip install -q --python $PY --reinstall-package torch "torch==2.14.1+cu130" \
    --index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  nvidia-smi
  $PY -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.__version__)"
  STEP=inputs
  mkdir -p /opt/ckpt /opt/lat /opt/emb
  aws s3 cp --quiet "$CHECKPOINT" /opt/ckpt/v2_student_large.pt
  for t in $TILES; do
    STEP=embed:$t
    aws s3 cp --quiet "$INPUTS/fp8-lattice/lattice_$t.pt" /opt/lat/
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/precision_gate/embed_lattice.py \
      "$card" /opt/lat /opt/emb "$t" --variants "$VARIANTS"
    aws s3 cp --quiet "/opt/emb/lattice_${card}_$t.npz" "$OUT/embed/"
    rm -f "/opt/lat/lattice_$t.pt"
  done
}

cpu() {
  local set=${ROLE#cpu-} ns=10 pids=() t s
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  STEP=inputs
  mkdir -p /opt/s/labels /opt/s/emb /opt/s/rec /opt/s/logs && cd /opt/s
  aws s3 sync --quiet "$INPUTS/fp8-mixed/labels/" labels/
  if [ "$set" = record ]; then
    aws s3 sync --quiet "$INPUTS/fp8-mixed/" rec/ --exclude '*' --include 'mixed_g6e_*.npz' --include 'mixed_g6_*.npz' \
      --include 'mixed_g5_*.npz' --include 'mixed_g5emu_*.npz'
    $PY /opt/te/scripts/precision_gate/stage_record_embeddings.py rec emb $TILES
  else
    aws s3 sync --quiet "$OUT/embed/" emb/
  fi
  STEP=calibrate
  for t in $TILES; do
    for s in $(seq 0 $((ns - 1))); do
      # shellcheck disable=SC2046
      UPLOAD=$OUT/$set/pred/$t $PY -u -W ignore /opt/te/scripts/precision_gate/calibrate.py \
        $(label_args "$t" | sed 's|^|labels/|') emb "$t" "$s" $ns "out/$t" > "logs/${t}_$s.txt" 2>&1 &
      pids+=($!)
    done
  done
  # The shards log to their own files; report progress and any traceback into the box log the watcher reads.
  (while sleep 120; do
    echo "PROGRESS calibrate: $(cat logs/*.txt 2>/dev/null | grep -c ' done ') variants done"
    for f in $(grep -l Traceback logs/*.txt 2>/dev/null); do echo "ERROR: traceback in $f"; tail -20 "$f"; done
  done) &
  local progress=$! failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
  kill $progress || true
  aws s3 sync --quiet logs/ "$OUT/$set/logs/" || true
  [ $failed -eq 0 ] || { echo "ERROR: $failed calibrate shards failed"; return 1; }
  STEP=gate
  mkdir -p reports
  for t in $TILES; do
    # Exit 3 is a variant outside the exception: a result, not a box failure. Anything else non-zero is a crash.
    rc=0
    # shellcheck disable=SC2046
    $PY /opt/te/scripts/precision_gate/apply_gate.py $(label_args "$t" | sed 's|^|labels/|') "out/$t" "$t" \
      "reports/$t.json" --bounds /opt/te/scripts/precision_gate/fp8_exception_bounds.json \
      --hold fp8_stage1=stage1:fp8 --hold mixed_stage1=stage1:mixed --hold fp8_all4=all4:fp8 \
      --hold mixed_all4=all4:mixed > "reports/$t.txt" || rc=$?
    [ $rc -eq 0 ] || [ $rc -eq 3 ] || { echo "ERROR: apply_gate crashed on $t (rc $rc)"; cat "reports/$t.txt"; return $rc; }
    [ $rc -eq 3 ] && echo "gate: $t has a variant outside the exception"
    cp "out/$t/envelope.json" "reports/${t}_envelope.json"
  done
  aws s3 sync --quiet reports/ "$OUT/$set/reports/"
}

case $ROLE in
  gpu-*) gpu ;;
  cpu-*) cpu ;;
  *) echo "unknown role $ROLE"; exit 2 ;;
esac
echo "ALL COMPLETE"
