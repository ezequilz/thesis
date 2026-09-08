#!/usr/bin/env bash
# Submit one 6h GPU hold job and write its id into configs/lrz.local.yaml.
# Requires an open ControlMaster (scripts/lrz/ssh-session.sh). No squeue loops.
#
#   scripts/lrz/allocate.sh
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$LRZ_ROOT"
lrz_load_config

if ! lrz_mux_alive; then
  echo "Open the LRZ SSH session first: scripts/lrz/ssh-session.sh"
  exit 2
fi

LOCAL="$LRZ_ROOT/configs/lrz.local.yaml"
if [ ! -f "$LOCAL" ]; then
  echo "Copy configs/lrz.example.yaml to configs/lrz.local.yaml first."
  exit 2
fi

# Both A100 partitions: HGX-only sat in PD (Priority) while a DGX A100 was free.
PARTITION="${LRZ_PARTITION:-lrz-hgx-a100-80x4,lrz-dgx-a100-80x8}"
echo "Submitting 6h hold job on $PARTITION (one sbatch, no wait loop)."
OUT="$(lrz_ssh "sbatch --job-name=gs-debug \
  --partition=$PARTITION \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G \
  --time=06:00:00 --output=gs-debug-%j.log \
  --wrap='sleep 21600'")"
echo "$OUT"
JOB="$(echo "$OUT" | awk '/Submitted batch job/{print $4}')"
if [ -z "$JOB" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi

JOB="$JOB" LOCAL="$LOCAL" python3 - <<'PY'
import os, re
from pathlib import Path
path = Path(os.environ["LOCAL"])
job = os.environ["JOB"]
text = path.read_text()
new, n = re.subn(r"(?m)^job_id:\s*.*$", f'job_id: "{job}"', text, count=1)
if n != 1:
    raise SystemExit(f"Could not replace job_id in {path}")
path.write_text(new)
print(f"Wrote job_id: \"{job}\" → {path}")
PY

echo
echo "Confirm once (do not loop):"
lrz_ssh "squeue --me --job=$JOB"
echo
echo "If ST is PD (Priority), on the login node run once:"
echo "  scontrol update JobId=$JOB Partition=$PARTITION"
echo "Then one more: squeue --me --job=$JOB"
echo
echo "GPU shell: scripts/lrz/gpu-shell.sh"
