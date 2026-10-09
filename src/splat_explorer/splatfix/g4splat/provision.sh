#!/usr/bin/env bash
# Runs inside the GPU container, in a separate persistent prefix.
set -euo pipefail
repo=$1
prefix=$2
revision=$3
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'G4Splat provisioning requires Linux x86_64 with NVIDIA CUDA'; exit 2; }
mkdir -p "$(dirname "$repo")" "$(dirname "$prefix")"
echo "Waiting for G4Splat installation lock: ${prefix}.setup.lock"
exec 9>"${prefix}.setup.lock"
flock 9
echo "Checking pinned G4Splat source at $repo"
if [[ ! -d "$repo/.git" ]]; then
  git clone --recursive https://github.com/DaLi-Jack/G4Splat.git "$repo"
  git -c core.fileMode=false -C "$repo" checkout "$revision"
fi
[[ $(git -C "$repo" rev-parse HEAD) == "$revision" ]] || { echo 'Existing G4Splat revision differs; refusing to overwrite'; exit 2; }
[[ -z $(git -c core.fileMode=false -C "$repo" status --porcelain --untracked-files=no) ]] || { echo 'Existing G4Splat source is modified'; exit 2; }
git -C "$repo" submodule update --init --recursive
unset PYTHONPATH PYTHONHOME PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# DSS ACLs can strip executable bits and confuse Git's dependency checkouts.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.fileMode GIT_CONFIG_VALUE_0=false
build_tmp=$(mktemp -d /tmp/g4splat-setup.XXXXXX)
trap 'rm -rf -- "$build_tmp"' EXIT
export TMPDIR="$build_tmp" TMP="$build_tmp" TEMP="$build_tmp"
bootstrap="${prefix}-bootstrap"
mkdir -p "$bootstrap"
if [[ ! -f "$bootstrap/bin/micromamba" ]]; then
  curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/2.3.3 -o "$bootstrap/micromamba.tar.bz2"
  tar -xjf "$bootstrap/micromamba.tar.bz2" -C "$bootstrap" bin/micromamba
fi
chmod u+x "$bootstrap/bin/micromamba"
echo "Preparing isolated CUDA 11.8 environment at $prefix"
export MAMBA_ROOT_PREFIX="$bootstrap/root"
if [[ ! -f "$prefix/conda-meta/history" ]]; then
  "$bootstrap/bin/micromamba" create -y -p "$prefix" --strict-channel-priority -c nvidia/label/cuda-11.8.0 -c conda-forge \
    python=3.9 pip 'cmake<4' gmp 'cgal<6' 'gxx_linux-64=11' cuda-version=11.8 cuda-toolkit=11.8 cuda-nvcc=11.8 cuda-compiler=11.8 cuda-command-line-tools=11.8 cuda-cccl=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 cuda-driver-dev=11.8
fi
# CUDA toolkit meta-packages have loose dependencies; pin the compiler too.
if [[ ! -f "$prefix/.g4splat-cuda118" ]]; then
  "$bootstrap/bin/micromamba" install -y -p "$prefix" --strict-channel-priority -c nvidia/label/cuda-11.8.0 -c conda-forge \
    cuda-version=11.8 cuda-toolkit=11.8 cuda-nvcc=11.8 cuda-compiler=11.8 cuda-command-line-tools=11.8 cuda-cccl=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 cuda-driver-dev=11.8
  touch "$prefix/.g4splat-cuda118"
fi
# Removing newer split CUDA packages can remove files shared with retained nvcc.
if [[ ! -x "$prefix/bin/nvcc" ]]; then
  "$bootstrap/bin/micromamba" install -y --force-reinstall -p "$prefix" \
    --strict-channel-priority -c nvidia/label/cuda-11.8.0 -c conda-forge cuda-nvcc=11.8
fi
"$prefix/bin/nvcc" --version | grep 'release 11.8'
[[ -x "$prefix/bin/python" ]] || { echo "Incomplete G4Splat Python prefix: $prefix"; exit 2; }
export CONDA_PREFIX="$prefix" CUDA_HOME="$prefix" CUDA_PATH="$prefix"
export PATH="$prefix/bin:$PATH"
# The NGC image exports its own incompatible torch libraries. Never inherit them.
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export CMAKE_PREFIX_PATH="$prefix:$prefix/lib/python3.9/site-packages/torch"
export CC="$prefix/bin/x86_64-conda-linux-gnu-cc" CXX="$prefix/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
cd "$repo"
if [[ ! -f "$prefix/.g4splat-dependencies" ]]; then
  python -m pip install 'pip<25' 'setuptools<70' 'cmake<4' wheel ninja
  python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  # NGC advertises architectures (e.g. 8.7) unsupported by this pinned torch.
  export TORCH_CUDA_ARCH_LIST
  TORCH_CUDA_ARCH_LIST=$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
  unset CUDAARCHS
  python -m pip install -r requirements.txt
  python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5' 'git+https://github.com/facebookresearch/segment-anything.git' 'git+https://github.com/facebookresearch/detectron2.git'
  python -m pip install --no-build-isolation -e 2d-gaussian-splatting/submodules/diff-surfel-rasterization -e 2d-gaussian-splatting/submodules/simple-knn
  (
    cd 2d-gaussian-splatting/submodules/tetra-triangulation
    cmake . -DCONDA_PREFIX="$prefix" -DCMAKE_CUDA_COMPILER="$prefix/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CXX"
    cmake --build . --parallel 2
    python -m pip install --no-build-isolation -e .
  )
  # Cython regenerates a tracked hamming.c. Build a copy to keep upstream clean.
  asmk_build="${prefix}-build/asmk"
  mkdir -p "$asmk_build"
  cp -a mast3r/asmk/. "$asmk_build/"
  (cd "$asmk_build/cython" && cythonize *.pyx)
  python -m pip install --no-build-isolation "$asmk_build"
  (cd mast3r/dust3r/croco/models/curope && python setup.py build_ext --inplace)
  touch "$prefix/.g4splat-dependencies"
fi
# Each file is published only after a successful download; interrupted transfers resume.
download() {
  local url=$1 target=$2
  mkdir -p "$(dirname "$target")"
  if [[ ! -s "$target" ]]; then
    curl --fail --location --retry 3 --continue-at - "$url" -o "$target.partial"
    mv "$target.partial" "$target"
  fi
}
download https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
for name in MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl; do
  download "https://download.europe.naverlabs.com/ComputerVision/MASt3R/$name" "mast3r/checkpoints/$name"
done
download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth checkpoint/segment-anything/sam_vit_h_4b8939.pth
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('bruiiii/See3D', revision='cbf13b6f813137134907408e40d3f2a17d6f0a80',
                  local_dir='.', local_dir_use_symlinks=False,
                  allow_patterns=['checkpoint/MVD_weights/*'],
                  ignore_patterns=['*/unet/single/*', '*/unet/SR/*', '*/open_clip_pytorch_model.bin'], max_workers=2)
PY
python -m pip freeze > "$prefix/g4splat-packages.txt"
echo 'G4Splat provisioning finished; validating CUDA and imports next.'
