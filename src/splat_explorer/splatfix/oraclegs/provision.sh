#!/usr/bin/env bash
# Explicit installation on a Linux CUDA worker; never touches existing methods.
set -euo pipefail
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Requires Linux x86_64 and NVIDIA CUDA'; exit 2; }
base=${1:-/workspace}
bootstrap="$base/oraclegs-bootstrap"
mkdir -p "$bootstrap"
exec 9>"$bootstrap/setup.lock"
flock 9
if [[ ! -f "$bootstrap/bin/micromamba" ]]; then
  curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/2.3.3 -o "$bootstrap/micromamba.tar.bz2"
  tar -xjf "$bootstrap/micromamba.tar.bz2" -C "$bootstrap" bin/micromamba
fi
chmod u+x "$bootstrap/bin/micromamba"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 MAMBA_ROOT_PREFIX="$bootstrap/root"
python "$(dirname "$0")/provision.py" "$base" "$bootstrap/bin/micromamba"
