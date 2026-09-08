#!/usr/bin/env bash
# Push a packed CUDA repair job to the allocated LRZ GPU, run it, pull results.
#
#   scripts/lrz/run-repair.sh <job-id>
#
# Type your LRZ password when ssh asks (once; ControlMaster is only for this
# run). Requires eduVPN. The dashboard packs jobs into outputs/lrz-jobs/<id>/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
JOB_ID="${1:?usage: scripts/lrz/run-repair.sh <job-id>}"
JOB_DIR="$ROOT/outputs/lrz-jobs/$JOB_ID"
if [ ! -d "$JOB_DIR" ]; then
  echo "No packed job at $JOB_DIR"
  echo "Start a repair from http://localhost:8090/repair with backend gsplat CUDA first."
  exit 1
fi

SOCK="${LRZ_SSH_CONTROL_PATH:-$HOME/.ssh/cm-lrz}"
ASKPASS=""
TARGET=""

eval "$(JOB_ID="$JOB_ID" PYTHONPATH="$ROOT/src" python3 -c "
from splat_explorer.repair_lrz import load_lrz_config, lrz_configured, remote_job_dir, srun_worker_command
import os, shlex, sys
c = load_lrz_config()
if not lrz_configured():
    sys.stderr.write(
        'LRZ is not configured. Copy configs/lrz.example.yaml to '
        'configs/lrz.local.yaml and set job_id to the running sbatch id.\\n'
    )
    sys.exit(2)
jid = os.environ['JOB_ID']
print('export LRZ_USER=' + shlex.quote(c['user']))
print('export LRZ_HOST=' + shlex.quote(c['host']))
print('export LRZ_JOB_ID=' + shlex.quote(str(c['job_id'])))
print('export LRZ_WORKSPACE=' + shlex.quote(c['workspace']))
print('export LRZ_REMOTE_JOB=' + shlex.quote(remote_job_dir(c, jid)))
print('export LRZ_SRUN=' + shlex.quote(srun_worker_command(c, jid)))
")"
TARGET="$LRZ_USER@$LRZ_HOST"

cleanup() {
  rm -f "$ASKPASS"
}
trap cleanup EXIT

if ssh -o ControlPath="$SOCK" -O check "$TARGET" >/dev/null 2>&1; then
  echo "==> Reusing ControlMaster $SOCK (from scripts/lrz/ssh-session.sh)"
else
  SSH_MASTER=(
    ssh -4 -F /dev/null
    -o PubkeyAuthentication=no
    -o PreferredAuthentications=password
    -o NumberOfPasswordPrompts=3
    -o StrictHostKeyChecking=accept-new
    -o ControlMaster=yes
    -o ControlPath="$SOCK"
    -o ControlPersist=8h
  )
  if [ -n "${LRZ_SSH_PASSWORD:-}" ]; then
    ASKPASS="$(mktemp /tmp/lrz-askpass-XXXXXX)"
    printf '#!/bin/sh\nprintf "%%s\\n" "$LRZ_SSH_PASSWORD"\n' > "$ASKPASS"
    chmod 700 "$ASKPASS"
    export SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force
    export DISPLAY="${DISPLAY:-:0}"
    echo "==> Connecting to $TARGET (password from this repair run)…"
    "${SSH_MASTER[@]}" -fN "$TARGET" </dev/null
  else
    echo "==> Connecting to $TARGET — type your LRZ password (or run scripts/lrz/ssh-session.sh first)."
    "${SSH_MASTER[@]}" -fN "$TARGET"
  fi
fi

SSH=(ssh -4 -F /dev/null -o ControlPath="$SOCK" -o ControlMaster=no)
RSYNC_E="ssh -4 -F /dev/null -o ControlPath=$SOCK -o ControlMaster=no"

echo "==> Checking Slurm job $LRZ_JOB_ID (one squeue call)"
STATE="$("${SSH[@]}" "$LRZ_USER@$LRZ_HOST" "squeue --me --job=${LRZ_JOB_ID} -h -o %T" | awk '{print $1}')"
if [ "$STATE" != "R" ]; then
  echo "Job $LRZ_JOB_ID is '${STATE:-unknown}', not running."
  echo "Allocate a GPU (sbatch) and put the new id in configs/lrz.local.yaml"
  exit 3
fi

echo "==> Uploading code + job $JOB_ID"
"${SSH[@]}" "$LRZ_USER@$LRZ_HOST" \
  "mkdir -p '$LRZ_WORKSPACE/inputs/$JOB_ID' '$LRZ_WORKSPACE/code' '$LRZ_WORKSPACE/outputs' '$LRZ_WORKSPACE/logs' '$LRZ_WORKSPACE/containers'"
rsync -az --delete -e "$RSYNC_E" "$ROOT/src/" "$LRZ_USER@$LRZ_HOST:$LRZ_WORKSPACE/code/src/"
if [ -f "$ROOT/pyproject.toml" ]; then
  rsync -az -e "$RSYNC_E" "$ROOT/pyproject.toml" "$LRZ_USER@$LRZ_HOST:$LRZ_WORKSPACE/code/pyproject.toml"
fi
rsync -az -e "$RSYNC_E" --exclude status.json "$JOB_DIR/" "$LRZ_USER@$LRZ_HOST:$LRZ_WORKSPACE/inputs/$JOB_ID/"

echo "==> srun CUDA refine on the allocated GPU"
"${SSH[@]}" "$LRZ_USER@$LRZ_HOST" "$LRZ_SRUN"

echo "==> Downloading repaired splat"
rsync -az -e "$RSYNC_E" "$LRZ_USER@$LRZ_HOST:$LRZ_WORKSPACE/inputs/$JOB_ID/" "$JOB_DIR/"

if [ ! -f "$JOB_DIR/scene_repaired.ply" ] || [ ! -f "$JOB_DIR/metrics.json" ]; then
  echo "Remote job finished without scene_repaired.ply / metrics.json"
  ls -la "$JOB_DIR"
  exit 4
fi

JOB_DIR="$JOB_DIR" PYTHONPATH="$ROOT/src" python3 -c "
from pathlib import Path
import os
from splat_explorer.repair_lrz import write_status
write_status(Path(os.environ['JOB_DIR']), phase='done', message='Results pulled from LRZ.')
"
echo "Done. Results in $JOB_DIR"
echo "The repair page will pick them up if a CUDA replay is waiting."
