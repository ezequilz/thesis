#!/usr/bin/env bash
# Submit one GPU sleep-hold job and optionally write its id into
# configs/lrz.local.yaml. Requires an open ControlMaster
# (scripts/lrz/ssh-session.sh). One sbatch — never an squeue loop.
#
#   scripts/lrz/allocate.sh              # 8h, both A100 partitions, start now
#   scripts/lrz/allocate.sh 24h
#   scripts/lrz/allocate.sh 6h
#   scripts/lrz/allocate.sh 8h --begin tomorrow
#   scripts/lrz/allocate.sh 24h --begin 2026-09-10T09:00
#   scripts/lrz/allocate.sh 8h --partition lrz-hgx-h100-94x4
#   scripts/lrz/allocate.sh 24h --after  # queue until the current job_id ends
#   scripts/lrz/allocate.sh --widen      # one scontrol: selected/default partitions
#   scripts/lrz/allocate.sh --use 5777470
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$LRZ_ROOT"
lrz_load_config

HOURS=8
AFTER=""
AFTER_SET=0
USE=""
WIDEN=0
BEGIN=""
PARTITION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --after)
      AFTER_SET=1
      if [ "${2:-}" != "" ] && [[ "${2}" =~ ^[0-9]+$ ]]; then
        AFTER="$2"
        shift 2
      else
        AFTER=""
        shift
      fi
      ;;
    --begin)
      BEGIN="${2:?usage: --begin tomorrow|YYYY-MM-DDTHH:MM}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:?usage: --partition lrz-dgx-a100-80x8[,...]}"
      shift 2
      ;;
    --use)
      USE="${2:?usage: scripts/lrz/allocate.sh --use <job-id>}"
      shift 2
      ;;
    --widen) WIDEN=1; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    [0-9]*h|[0-9]*)
      HOURS="${1%h}"
      if ! [[ "$HOURS" =~ ^[0-9]+$ ]]; then
        echo "Unknown argument: $1"
        exit 2
      fi
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: scripts/lrz/allocate.sh [Nh] [--begin TIME] [--partition LIST] [--after [JOBID]] | --widen | --use JOBID"
      exit 2
      ;;
  esac
done

export PYTHONPATH="$LRZ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "$USE" ]; then
  JOB="$USE" "$LRZ_PYTHON" - <<'PY'
import os
from splat_explorer.repair_lrz import use_lrz_job
print(use_lrz_job(os.environ["JOB"])["message"])
PY
  exit 0
fi

if ! lrz_mux_alive; then
  echo "Open the LRZ SSH session first: scripts/lrz/ssh-session.sh"
  exit 2
fi

if [ "$WIDEN" = 1 ]; then
  export LRZ_ALLOC_PARTITION="$PARTITION"
  "$LRZ_PYTHON" - <<'PY'
import os
from splat_explorer.repair_lrz import widen_lrz_job
body = widen_lrz_job(partition=os.environ.get("LRZ_ALLOC_PARTITION") or None)
print(body["command"])
print(body["message"])
if body.get("stdout"):
    print(body["stdout"])
PY
  echo
  echo "Confirm once (do not loop):"
  lrz_ssh "squeue --me"
  exit 0
fi

export LRZ_HOLD_HOURS="$HOURS"
export LRZ_AFTER_SET="$AFTER_SET"
export LRZ_AFTER_JOB="$AFTER"
export LRZ_BEGIN="$BEGIN"
export LRZ_ALLOC_PARTITION="$PARTITION"
"$LRZ_PYTHON" - <<'PY'
import os
from splat_explorer.repair_lrz import allocate_lrz_gpu
after = False
if os.environ.get("LRZ_AFTER_SET") == "1":
    after = os.environ.get("LRZ_AFTER_JOB") or True
body = allocate_lrz_gpu(
    os.environ["LRZ_HOLD_HOURS"],
    after=after,
    begin=os.environ.get("LRZ_BEGIN") or None,
    partition=os.environ.get("LRZ_ALLOC_PARTITION") or None,
)
print(body["command"])
print(body["message"])
if body.get("stdout"):
    print(body["stdout"])
print(f"job_id={body['job_id']} switched={body['switched']}")
PY

echo
echo "Confirm once (do not loop):"
lrz_ssh "squeue --me"
echo
echo "If ST is PD (Priority): scripts/lrz/allocate.sh --widen"
echo "GPU shell: scripts/lrz/gpu-shell.sh"
echo "Review:    scripts/lrz/status.sh --sinfo   or http://localhost:8090/repair/gpu"
