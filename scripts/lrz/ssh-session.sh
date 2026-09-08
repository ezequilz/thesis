#!/usr/bin/env bash
# Open a multiplexed SSH session to LRZ. Type your password once; the dashboard
# reuses ~/.ssh/cm-lrz for rsync/srun until ControlPersist expires (8h) or you
# run: ssh -o ControlPath=$HOME/.ssh/cm-lrz -O exit go73kaf2@login.ai.lrz.de
#
#   scripts/lrz/ssh-session.sh
#
# Requires eduVPN. Does not allocate a GPU — that is still a manual sbatch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SOCK="${LRZ_SSH_CONTROL_PATH:-$HOME/.ssh/cm-lrz}"
mkdir -p "$(dirname "$SOCK")"

eval "$(PYTHONPATH="$ROOT/src" python3 -c "
from splat_explorer.repair_lrz import load_lrz_config
import shlex
c = load_lrz_config()
print('export LRZ_USER=' + shlex.quote(c['user']))
print('export LRZ_HOST=' + shlex.quote(c['host']))
print('export LRZ_JOB_ID=' + shlex.quote(str(c.get('job_id') or '')))
")"

TARGET="$LRZ_USER@$LRZ_HOST"

if ssh -o ControlPath="$SOCK" -O check "$TARGET" >/dev/null 2>&1; then
  echo "ControlMaster already up: $SOCK"
else
  echo "Connecting to $TARGET — type your LRZ password once."
  ssh -4 -F /dev/null \
    -o PubkeyAuthentication=no \
    -o PreferredAuthentications=password \
    -o NumberOfPasswordPrompts=3 \
    -o StrictHostKeyChecking=accept-new \
    -o ControlMaster=yes \
    -o ControlPath="$SOCK" \
    -o ControlPersist=8h \
    -fN "$TARGET"
  echo "Session stored at $SOCK (ControlPersist 8h)."
fi

echo "==> squeue --me (one call)"
ssh -4 -F /dev/null -o ControlPath="$SOCK" -o ControlMaster=no "$TARGET" "squeue --me"
echo
echo "Dashboard CUDA repair can use this socket now."
echo "Close with: ssh -o ControlPath=$SOCK -O exit $TARGET"
