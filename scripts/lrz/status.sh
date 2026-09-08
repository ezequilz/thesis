#!/usr/bin/env bash
# One-shot review of SSH, your Slurm jobs, workspace, and (optional) partitions.
# Never loops squeue/sinfo — LRZ treats that as a DoS.
#
#   scripts/lrz/status.sh
#   scripts/lrz/status.sh --sinfo
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$LRZ_ROOT"
lrz_load_config

SINFO=0
case "${1:-}" in
  --sinfo) SINFO=1 ;;
  "" ) ;;
  -h|--help)
    sed -n '2,8p' "$0"
    exit 0
    ;;
  *)
    echo "Usage: scripts/lrz/status.sh [--sinfo]"
    exit 2
    ;;
esac

echo "==> ControlMaster"
if lrz_mux_alive; then
  echo "alive  $SOCK  $TARGET"
else
  echo "down   $SOCK"
  echo "Run: scripts/lrz/ssh-session.sh"
  exit 2
fi

echo
echo "==> squeue --me (one call)"
lrz_ssh "squeue --me"
echo
echo "configs/lrz.local.yaml job_id=${LRZ_JOB_ID:-empty}"

echo
echo "==> DSS workspace + Enroot image (login node)"
lrz_ssh "mkdir -p $(printf %q "$LRZ_WORKSPACE")/{containers,inputs,outputs,logs,code}; \
ls -ld $(printf %q "$LRZ_WORKSPACE") $(printf %q "$LRZ_WORKSPACE")/containers; \
if [ -f $(printf %q "$LRZ_CONTAINER") ]; then ls -lh $(printf %q "$LRZ_CONTAINER"); else echo CONTAINER_MISSING; fi; \
if [ -s \"\$HOME/enroot/.credentials\" ]; then echo NGC_CREDS_OK; else echo NGC_CREDS_MISSING; fi"

if [ "$SINFO" = 1 ]; then
  echo
  echo "==> sinfo (one call; MIXED may still have a free GPU)"
  lrz_ssh "sinfo -p lrz-v100x2,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4"
fi

echo
echo "Connect:  scripts/lrz/gpu-shell.sh"
echo "Reserve:  scripts/lrz/allocate.sh 8h | 24h"
echo "Dashboard: http://localhost:8090/repair/gpu"
