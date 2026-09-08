#!/usr/bin/env bash
# Interactive shell on the allocated GPU node. --overlap is required because
# the sbatch hold job is `sleep` and already occupies the allocation.
#
#   scripts/lrz/gpu-shell.sh
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

echo "srun --overlap on job $LRZ_JOB_ID (Ctrl-D to leave the GPU shell)."
lrz_ssh_tty "srun --jobid=$LRZ_JOB_ID --overlap --nodes=1 --ntasks=1 --cpus-per-task=$LRZ_CPUS --gres=gpu:1 --pty bash"
