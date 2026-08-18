# splat-explorer

A harness that lets a Vision-Language Model (VLM) explore 3D Gaussian Splat
scenes like an embodied agent and hunt for rendering artifacts.

The loop: the harness renders the scene from the agent's current camera pose,
sends the screenshot to the VLM, the VLM answers with a tool call
(`move`, `rotate`, `look`, `report_artifact`, `done`), the harness applies it
and renders the next frame. Every episode is logged as frames + an action
trace for later evaluation.

```
 ┌────────────┐  screenshot  ┌────────────┐  tool call  ┌────────────┐
 │  Renderer   │ ───────────▶ │  Harness   │ ──────────▶ │  VLM (API) │
 │ (gsplat /   │              │ (episode   │ ◀────────── │  move 1.2m │
 │  cpu_points)│ ◀─────────── │  loop)     │             │  rotate 30°│
 └────────────┘  new camera   └────────────┘             └────────────┘
        ▲                            │
   GaussianScene                     ▼
   (SOG loader)              outputs/episodes/…
```

## Repository layout

```
3dgs_rooms/                     Splat assets (.sog bundles; mounted read-only in Docker)
configs/default.yaml            All knobs: scene, renderer, camera, agent, viewer
src/splat_explorer/
  scene/                        Asset loading
    ply_loader.py                 Standard 3DGS .ply decoder (uncompressed, default)
    sog_loader.py                 SOG v2 (.sog) decoder — compressed bundles
    types.py                      GaussianScene dataclass + robust bounds helpers
  rendering/
    base.py                       Camera model (OpenCV convention) + Renderer protocol
    cpu_splat_renderer.py         Full gaussian splatting on CPU (EWA + soft z-buffer)
    cpu_point_renderer.py         Dot-based debug renderer (legacy)
    gsplat_renderer.py            Real 3DGS rasterization [STUB — needs CUDA, untested]
    viser_viewer.py               Browser-based interactive debug viewer
  agent/
    actions.py                    Action space + OpenAI tool schemas
    camera_rig.py                 Embodied camera: yaw/pitch/position, applies actions
    vlm.py                        Policies: scripted (works) / OpenAI-compatible [STUB]
    cli_relay.py                  CliRelay backend: Gemini/Claude/OpenAI via one proxy
    loop.py                       observe → decide → act episode loop + logging
  tasks/
    artifact_hunt.py              Task prompt + scoring placeholder [STUB]
  cli.py                        Entrypoints: render-test / explore / viewer
Dockerfile                      CPU image (loader, cpu renderer, viewer, harness)
docker/Dockerfile.gpu           CUDA image for gsplat [STUB — untested]
docker-compose.yml              Services: render-test, viewer, harness
scripts/start.sh                Start/restart the stack: CliRelay + build + viewer
```

## Quick start (Docker)

```bash
./scripts/start.sh
```

Safe to re-run at any time; it (re)starts the whole stack: CliRelay at
http://localhost:8317 (cloned to `~/CliRelay` on first run; panel at
`/manage`, admin password printed from its `.env`), builds the image, and
(re)starts the interactive viewer at http://localhost:8080 — killing a stale
locally-run viewer holding the port if needed. Optional one-off jobs:
`--render-test` renders a sanity panorama to `outputs/test_views/`,
`--episode` runs an agent episode into `outputs/episodes/<timestamp>/`, and
`--stop` shuts everything down (project + CliRelay).

Individual services:

```bash
docker compose run --rm render-test     # panorama sanity render
docker compose up viewer                # viser debug viewer on :8080
docker compose run --rm harness         # agent episode (scripted policy)
```

## Quick start (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[viewer]"
splat-explorer render-test
splat-explorer explore
splat-explorer viewer
```

## Connecting a real VLM

The default policy is `scripted` (no API needed). Two live backends exist:

### CliRelay (Gemini / Claude / OpenAI through one endpoint)

[CliRelay](https://github.com/kittors/CliRelay) is a self-hosted proxy that
turns AI CLI subscriptions and OAuth credentials (Gemini CLI, Claude Code,
Codex, ...) into a single OpenAI-compatible endpoint at
`http://localhost:8317/v1`. The backend (`agent/cli_relay.py`) sends the tool
catalog as text and parses one JSON action per turn out of the reply, so it
works with any vision model the relay routes to regardless of how well the
upstream translates native tool calls.

Set up the relay once (see its README: `docker compose up -d`, add your
provider credentials in the panel at `http://localhost:8317/manage`, create
an API key), then:

```bash
pip install -e ".[vlm]"
export CLIRELAY_API_KEY=sk-...   # key created in the CliRelay panel
# Verify relay + model routing + parsing with a single request first:
python scripts/test_cli_relay.py
# Then run an episode:
splat-explorer --config configs/cli_relay.yaml explore
```

Pick the model in `configs/cli_relay.yaml` (`agent.model`) — any
vision-capable model ID your relay routes, e.g. `gemini-2.5-pro`,
`claude-sonnet-4-5`, or `gpt-4o`. If the relay runs elsewhere, set
`agent.relay_base_url` (from inside Docker, a host-local relay is
`http://host.docker.internal:8317/v1`).

### OpenAI-compatible endpoints

Create a config override, e.g. `configs/live.yaml`:

```yaml
agent:
  vlm_backend: openai
  model: gpt-4o          # any OpenAI-compatible vision model with tool calling
  base_url: ""           # set for vLLM / OpenRouter / Ollama endpoints
```

then:

```bash
export OPENAI_API_KEY=sk-...
splat-explorer --config configs/live.yaml explore
```

The OpenAI client is a stub: written against the chat-completions API but
not yet exercised live (see `agent/vlm.py` TODOs on history management).

## Renderer backends

| Backend      | Quality | Requirements | Status |
|--------------|---------|--------------|--------|
| `cpu_splats` | full anisotropic splats, alpha blending, soft depth buffer (~40s/frame at 960x720, full 5.7M-splat scene) | none | working, default |
| `gsplat`     | reference 3DGS rasterization | NVIDIA GPU, `pip install ".[gpu]"`, `docker/Dockerfile.gpu` | stub, untested |
| `cpu_points` | dot-based debug (speckled; legacy) | none | working |
| viser viewer | interactive browser splats (full scene, `viewer.max_splats: 0` = no subsampling) | `pip install ".[viewer]"` | working |

The renderer is behind a small `Renderer` protocol (`rendering/base.py`), so
backends are swappable via `renderer.backend` in the config.

## Scene assets

Two formats are supported (`scene.path` dispatches on extension):

- **`.sog` (default)** — SOG v2 bundles (PlayCanvas' compressed splat format,
  [spec](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/)).
  ~17x smaller than the raw export (78MB vs 1.4GB here) and git-friendly;
  quantized, but decode statistics match the PLY to ~4 decimals on this asset.
- **`.ply`** — standard uncompressed 3DGS exports as written by the INRIA
  trainer, nerfstudio, gsplat, etc. Full float precision, higher-order SH
  present in the file (not decoded yet). Useful as lossless reference when
  you need to rule out compression as an artifact source. Too big for GitHub
  (>100MB limit) — `3dgs_rooms/*.ply` is gitignored, keep these local.

Both loaders decode positions, scales, orientations, base color and opacity;
higher-order spherical harmonics (view-dependent color) are not decoded yet.

Note: the provided asset is y-down (like most COLMAP-trained scenes) even
though the SOG spec says y-up — `camera.up_axis` in the config handles this.

Keep `.ply` assets out of git pushes to GitHub (100MB file limit) — track
only the compact `.sog` or use Git LFS.

## Roadmap / open stubs

- [ ] `gsplat` renderer: validate on a CUDA machine, wire up SH decoding
- [ ] VLM client: exercise against a live API, conversation-history trimming,
      malformed tool call handling, retries
- [ ] Task evaluation: ground-truth artifact annotations, precision/recall
      scoring, exploration-coverage metrics (`tasks/artifact_hunt.py`)
- [ ] Collision handling: stop the agent walking through walls (occupancy
      from gaussian density)
- [x] PLY loading for uncompressed 3DGS exports
- [ ] Viewer: overlay agent trajectory + camera frustum during episodes
- [ ] Synthetic artifact injection for controlled benchmarks
