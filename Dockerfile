# CPU image: SOG loading, CPU splat rendering, viser debug viewer, agent harness.
# Runs anywhere (including Docker Desktop on macOS). This Linux image cannot
# run gsplat-mlx / Metal; scripts/start.sh installs the [apple] extra into
# .venv and starts the dashboard on the Mac host so Apple Silicon refine works.
# For CUDA 3DGS rasterization on an NVIDIA GPU, see docker/Dockerfile.gpu.

FROM python:3.11-slim

WORKDIR /app

# ffmpeg encodes the on-demand episode video (H.264); DejaVu is the overlay font.
# git is needed to pip-install GitHub extras (gsplat-mlx) if you add them here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-dejavu-core git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in their own layer so code edits don't re-download them.
# Keep this list in sync with pyproject.toml (core + viewer + vlm extras).
# Do not install [apple] here: mlx-metal is macOS-only and will fail on Linux.
RUN pip install --no-cache-dir \
    "numpy>=1.26" \
    "pillow>=10.0" \
    "pyyaml>=6.0" \
    "scipy>=1.11" \
    "viser>=0.2.7" \
    "openai>=1.40"

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps .

# Splat assets and outputs are bind-mounted by docker-compose:
#   ./3dgs_rooms -> /app/3dgs_rooms (ro),  ./outputs -> /app/outputs

ENTRYPOINT ["splat-explorer"]
CMD ["render-test"]

