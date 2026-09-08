#!/usr/bin/env bash
# Check DSS workspace and print the Enroot import commands to run on the GPU node.
# Does not import the image (multi-GB, compute-node only).
#
#   scripts/lrz/bootstrap.sh
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

echo "==> login node: job + DSS dirs"
lrz_ssh "squeue --me --job=$LRZ_JOB_ID; mkdir -p $(printf %q "$LRZ_WORKSPACE")/{containers,inputs,outputs,logs,code}"
lrz_ssh "ls -ld $(printf %q "$LRZ_WORKSPACE") $(printf %q "$LRZ_WORKSPACE")/containers; ls -lh $(printf %q "$LRZ_CONTAINER") 2>/dev/null || echo CONTAINER_MISSING"

CREDS="$(lrz_ssh 'test -s "$HOME/enroot/.credentials" && echo NGC_CREDS_OK || echo NGC_CREDS_MISSING')"
echo "==> NGC credentials on cluster home: $CREDS"
echo
echo "Enroot is only on the compute node. Open a GPU shell:"
echo "  scripts/lrz/gpu-shell.sh"
echo
echo "Then:"
echo "  cd $LRZ_WORKSPACE"
echo
echo "  # NGC (needs ~/enroot/.credentials). Use '#' not '/':"
echo "  #   docker://nvcr.io/nvidia/pytorch:... hits Docker Hub and 401s."
echo "  enroot import -o containers/pytorch.sqsh 'docker://nvcr.io#nvidia/pytorch:24.10-py3'"
echo
echo "  # No NGC key: public CUDA devel image (includes nvcc for gsplat):"
echo "  enroot import -o containers/pytorch.sqsh docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
echo
echo "Then named Pyxis container (once per allocation):"
echo "  srun --jobid=$LRZ_JOB_ID --overlap --gres=gpu:1 \\"
echo "    --container-image=$LRZ_CONTAINER \\"
echo "    --container-name=splat-repair \\"
echo "    --container-mounts=$LRZ_WORKSPACE:/workspace --pty bash"
echo "  python -c 'import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)'"
echo
echo "Full write-up: docs/lrz.md"
