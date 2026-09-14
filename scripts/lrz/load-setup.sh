#!/usr/bin/env bash
# Once per GPU allocation: check occupancy, start a job-scoped Pyxis
# container from DSS pytorch.sqsh, rsync code, and install gsplat onto
# YOUR shared drive (never a colleague's files). Refuses to run if a
# process that is not yours is on the allocated GPU.
#
#   scripts/lrz/load-setup.sh
#
# Requires eduVPN + ControlMaster (scripts/lrz/ssh-session.sh) and a
# running job_id in configs/lrz.local.yaml (Use on /repair/gpu).
set -euo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$LRZ_ROOT"
lrz_load_config
lrz_require_job

if ! lrz_mux_alive; then
  echo "Open the LRZ SSH session first: scripts/lrz/ssh-session.sh"
  exit 2
fi

echo "==> Loading GPU setup on job $LRZ_JOB_ID (named container + torch/gsplat on DSS)"
PYTHONPATH="$LRZ_ROOT/src" "$LRZ_PYTHON" -c "
from splat_explorer.repair_lrz import setup_lrz_gpu
detail = setup_lrz_gpu()
print('SETUP_OK', detail)
"
echo
echo "Dashboard CUDA repair can use this allocation now."
echo "Reload later with the same command, or Load GPU setup on /repair/gpu."
