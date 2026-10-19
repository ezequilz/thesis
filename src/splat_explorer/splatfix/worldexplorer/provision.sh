#!/usr/bin/env bash
set -euo pipefail
base=${1:-/workspace}
bootstrap="$base/oraclegs-bootstrap"
[[ -x "$bootstrap/bin/micromamba" ]] || { echo 'OracleGS micromamba bootstrap required'; exit 1; }
mkdir -p "$base/worldexplorer-cache"
exec 8>"$base/worldexplorer-cache/setup.lock"
flock 8
unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST PIP_FIND_LINKS
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
export MAMBA_ROOT_PREFIX="$bootstrap/root" MAX_JOBS=1
python "$(dirname "$0")/provision.py" "$base" "$bootstrap/bin/micromamba"
