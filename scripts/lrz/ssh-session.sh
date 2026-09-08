#!/usr/bin/env bash
# Open a multiplexed SSH session to LRZ. Type your password once; the dashboard
# reuses ~/.ssh/cm-lrz for rsync/srun until ControlPersist expires (8h) or you
# run: /usr/bin/ssh -o ControlPath=$HOME/.ssh/cm-lrz -O exit go73kaf2@login.ai.lrz.de
#
#   scripts/lrz/ssh-session.sh
#
# Same flags as the login that already worked:
#   /usr/bin/ssh -4 -F /dev/null -o PubkeyAuthentication=no \
#     -o PreferredAuthentications=password go73kaf2@login.ai.lrz.de
# Do NOT use -fN here — that backgrounds ssh and breaks the password prompt
# in Cursor's terminal (Permission denied even with the right password).
#
# Requires eduVPN. Does not allocate a GPU — that is still a manual sbatch
# (scripts/lrz/allocate.sh). Full handover: docs/lrz.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SSH="${LRZ_SSH_BIN:-/usr/bin/ssh}"
SOCK="${LRZ_SSH_CONTROL_PATH:-$HOME/.ssh/cm-lrz}"
mkdir -p "$(dirname "$SOCK")"

if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY=python3; fi
eval "$(PYTHONPATH="$ROOT/src" "$PY" -c "
from splat_explorer.repair_lrz import load_lrz_config
import shlex
c = load_lrz_config()
print('export LRZ_USER=' + shlex.quote(c['user']))
print('export LRZ_HOST=' + shlex.quote(c['host']))
print('export LRZ_JOB_ID=' + shlex.quote(str(c.get('job_id') or '')))
")"

TARGET="$LRZ_USER@$LRZ_HOST"
AUTH=(
  -4 -F /dev/null
  -o PubkeyAuthentication=no
  -o PreferredAuthentications=password
  -o NumberOfPasswordPrompts=1
  -o KbdInteractiveAuthentication=no
)

if "$SSH" -o ControlPath="$SOCK" -O check "$TARGET" >/dev/null 2>&1; then
  echo "ControlMaster already up: $SOCK"
else
  echo "Connecting to $TARGET with $SSH (interactive, same as your working login)."
  echo "Type your LRZ password once. You will not stay in a remote shell."
  "$SSH" "${AUTH[@]}" \
    -o ControlMaster=yes \
    -o ControlPath="$SOCK" \
    -o ControlPersist=8h \
    "$TARGET" "echo session-ok; hostname"
  echo "Session stored at $SOCK (ControlPersist 8h)."
fi

echo "==> squeue --me (one call)"
"$SSH" -4 -F /dev/null -o ControlPath="$SOCK" -o ControlMaster=no "$TARGET" "squeue --me"
echo
echo "Dashboard CUDA repair can use this socket now."
echo "Close with: $SSH -o ControlPath=$SOCK -O exit $TARGET"
