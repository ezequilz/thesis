#!/usr/bin/env bash
# One-time (per allocation) setup ON the GPU node. Run this from a project
# terminal — you will type your LRZ password. Then paste the commands it
# prints into `srun --jobid=… --pty bash` if Enroot is only on the compute node.
#
#   scripts/lrz/bootstrap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

eval "$(PYTHONPATH="$ROOT/src" python3 -c "
from splat_explorer.repair_lrz import load_lrz_config, lrz_configured
import shlex, sys
c = load_lrz_config()
print(f\"export LRZ_USER={shlex.quote(c['user'])}\")
print(f\"export LRZ_HOST={shlex.quote(c['host'])}\")
print(f\"export LRZ_JOB_ID={shlex.quote(str(c.get('job_id') or ''))}\")
print(f\"export LRZ_WORKSPACE={shlex.quote(c['workspace'])}\")
print(f\"export LRZ_CONTAINER={shlex.quote(str(c.get('container') or ''))}\")
")"

if [ -z "${LRZ_JOB_ID:-}" ] || ! [[ "$LRZ_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Set job_id in configs/lrz.local.yaml to the running sbatch job first."
  echo "Example:"
  echo "  cp configs/lrz.example.yaml configs/lrz.local.yaml"
  echo "  # edit job_id, then re-run this script"
  exit 2
fi

echo "This opens SSH to $LRZ_USER@$LRZ_HOST — type your password."
echo "Prefer scripts/lrz/ssh-session.sh first so ControlMaster is reused."
echo
ssh -4 -F /dev/null \
  -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password \
  -o NumberOfPasswordPrompts=3 \
  -o StrictHostKeyChecking=accept-new \
  "$LRZ_USER@$LRZ_HOST" bash -s <<EOF
set -euo pipefail
echo "==> job \$USER"
squeue --me --job=$LRZ_JOB_ID
mkdir -p "$LRZ_WORKSPACE"/{containers,inputs,outputs,logs,code}
echo "==> DSS workspace"
ls -ld "$LRZ_WORKSPACE" "$LRZ_WORKSPACE"/containers || true
echo
echo "Enroot is only on the *compute* node. From the login node run:"
echo
echo "  srun --jobid=$LRZ_JOB_ID --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --pty bash"
echo
echo "Then, on the GPU node (once per allocation):"
echo
echo "  cd $LRZ_WORKSPACE"
echo "  # Import NGC PyTorch if the squashfs is missing (slow, ~once):"
echo "  if [ ! -f containers/pytorch.sqsh ]; then"
echo "    enroot import -o containers/pytorch.sqsh docker://nvcr.io/nvidia/pytorch:24.10-py3"
echo "  fi"
echo "  srun --jobid=$LRZ_JOB_ID --gres=gpu:1 \\\\"
echo "    --container-image=$LRZ_WORKSPACE/containers/pytorch.sqsh \\\\"
echo "    --container-name=splat-repair \\\\"
echo "    --container-mounts=$LRZ_WORKSPACE:/workspace --pty bash"
echo "  # inside the container:"
echo "  pip install -e /workspace/code '.[gpu]'"
echo "  python -c 'import torch,gsplat; print(torch.cuda.get_device_name(0), gsplat.__version__)'"
echo
echo "After src/ is rsynced by run-repair.sh, /workspace/code/src is the package."
echo "If pip install -e fails, PYTHONPATH=/workspace/code/src is enough for the worker."
EOF
