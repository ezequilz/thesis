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
    sog_loader.py                 SOG v2 (.sog) decoder — means/scales/quats/color/opacity
    types.py                      GaussianScene dataclass + robust bounds helpers
  rendering/
    base.py                       Camera model (OpenCV convention) + Renderer protocol
    cpu_point_renderer.py         Dependency-free debug renderer (works everywhere)
    gsplat_renderer.py            Real 3DGS rasterization [STUB — needs CUDA, untested]
    viser_viewer.py               Browser-based interactive debug viewer
  agent/
    actions.py                    Action space + OpenAI tool schemas
    camera_rig.py                 Embodied camera: yaw/pitch/position, applies actions
    vlm.py                        Policies: scripted (works) / OpenAI-compatible [STUB]
    loop.py                       observe → decide → act episode loop + logging
  tasks/
    artifact_hunt.py              Task prompt + scoring placeholder [STUB]
  cli.py                        Entrypoints: render-test / explore / viewer
Dockerfile                      CPU image (loader, cpu renderer, viewer, harness)
docker/Dockerfile.gpu           CUDA image for gsplat [STUB — untested]
docker-compose.yml              Services: render-test, viewer, harness
scripts/start.sh                Build → test render → viewer → episode
```

## Quick start (Docker)

```bash
./scripts/start.sh
```

This builds the image, renders a sanity panorama from the scene center to
`outputs/test_views/` (open `contact_sheet.png`), starts the interactive
viewer at http://localhost:8080, and runs a scripted agent episode into
`outputs/episodes/<timestamp>/`.

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

The default policy is `scripted` (no API needed). To drive the agent with a
VLM, create a config override, e.g. `configs/live.yaml`:

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

The VLM client is a stub: written against the OpenAI chat-completions API but
not yet exercised live (see `agent/vlm.py` TODOs on history management).

## Renderer backends

| Backend      | Quality | Requirements | Status |
|--------------|---------|--------------|--------|
| `cpu_points` | debug (depth-sorted point cloud, no alpha blending) | none | working |
| `gsplat`     | faithful 3DGS rasterization | NVIDIA GPU, `pip install ".[gpu]"`, `docker/Dockerfile.gpu` | stub, untested |
| viser viewer | interactive browser splats | `pip install ".[viewer]"` | working (subsampled) |

The renderer is behind a small `Renderer` protocol (`rendering/base.py`), so
backends are swappable via `renderer.backend` in the config.

## Scene assets

Scenes are SOG v2 bundles (`.sog` — PlayCanvas' compressed splat format,
[spec](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/)).
The loader decodes positions, scales, orientations, base color and opacity;
higher-order spherical harmonics (view-dependent color) are not decoded yet.

Note: the provided asset is y-down (like most COLMAP-trained scenes) even
though the SOG spec says y-up — `camera.up_axis` in the config handles this.

## Roadmap / open stubs

- [ ] `gsplat` renderer: validate on a CUDA machine, wire up SH decoding
- [ ] VLM client: exercise against a live API, conversation-history trimming,
      malformed tool call handling, retries
- [ ] Task evaluation: ground-truth artifact annotations, precision/recall
      scoring, exploration-coverage metrics (`tasks/artifact_hunt.py`)
- [ ] Collision handling: stop the agent walking through walls (occupancy
      from gaussian density)
- [ ] PLY loading for uncompressed 3DGS exports
- [ ] Viewer: overlay agent trajectory + camera frustum during episodes
- [ ] Synthetic artifact injection for controlled benchmarks
