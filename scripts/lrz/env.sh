# Shared LRZ helpers. Source from other scripts in this directory.
# Does not allocate a GPU and does not loop on squeue.
LRZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -x "$LRZ_ROOT/.venv/bin/python" ]; then
  LRZ_PYTHON="$LRZ_ROOT/.venv/bin/python"
else
  LRZ_PYTHON="${LRZ_PYTHON:-python3}"
fi
SSH="${LRZ_SSH_BIN:-/usr/bin/ssh}"
SOCK="${LRZ_SSH_CONTROL_PATH:-$HOME/.ssh/cm-lrz}"
LRZ_AUTH=(
  -4 -F /dev/null
  -o PubkeyAuthentication=no
  -o PreferredAuthentications=password
  -o NumberOfPasswordPrompts=1
  -o KbdInteractiveAuthentication=no
)

lrz_load_config() {
  eval "$(cd "$LRZ_ROOT" && PYTHONPATH="$LRZ_ROOT/src" "$LRZ_PYTHON" -c "
from splat_explorer.repair_lrz import load_lrz_config
import shlex
c = load_lrz_config()
print('LRZ_USER=' + shlex.quote(c['user']))
print('LRZ_HOST=' + shlex.quote(c['host']))
print('LRZ_JOB_ID=' + shlex.quote(str(c.get('job_id') or '')))
print('LRZ_WORKSPACE=' + shlex.quote(c['workspace']))
print('LRZ_CONTAINER=' + shlex.quote(str(c.get('container') or '')))
print('LRZ_CONTAINER_NAME=' + shlex.quote(str(c.get('container_name') or 'splat-repair')))
print('LRZ_CPUS=' + shlex.quote(str(int(c.get('cpus') or 4))))
")"
  TARGET="$LRZ_USER@$LRZ_HOST"
}

lrz_mux_alive() {
  [ -n "${TARGET:-}" ] || return 1
  "$SSH" -o ControlPath="$SOCK" -O check "$TARGET" >/dev/null 2>&1
}

lrz_ssh() {
  if lrz_mux_alive; then
    "$SSH" -4 -F /dev/null -o ControlMaster=no -o ControlPath="$SOCK" "$TARGET" "$@"
  else
    echo "No ControlMaster at $SOCK — type your LRZ password." >&2
    echo "Prefer: scripts/lrz/ssh-session.sh" >&2
    "$SSH" "${LRZ_AUTH[@]}" "$TARGET" "$@"
  fi
}

lrz_ssh_tty() {
  if lrz_mux_alive; then
    "$SSH" -4 -t -F /dev/null -o ControlMaster=no -o ControlPath="$SOCK" "$TARGET" "$@"
  else
    echo "No ControlMaster at $SOCK — type your LRZ password." >&2
    "$SSH" -t "${LRZ_AUTH[@]}" "$TARGET" "$@"
  fi
}

lrz_require_job() {
  if [ -z "${LRZ_JOB_ID:-}" ] || ! [[ "$LRZ_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "No running job_id in configs/lrz.local.yaml." >&2
    echo "Run: scripts/lrz/allocate.sh" >&2
    return 2
  fi
}
