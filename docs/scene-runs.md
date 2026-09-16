# Automated scene-runs

Scene-runs are the production, deadline-controlled counterpart to debugging
episodes. The original episode dashboard (`/`) and repair studio (`/repair`)
remain available for isolated testing.

## Start

```bash
./scripts/start.sh
```

Open `http://127.0.0.1:8090/scene-runs`. Before a queued run can start,
`/repair/gpu` must show a selected LRZ allocation in `ST=R` with its setup
loaded. The local scene-run manager is independent of the browser and writes
`outputs/scene-run-manager.log`. `./scripts/start.sh` always stops and
restarts that manager so queued runs pick up the current scene-run code.

Each run starts from the selected catalog scene and writes only below:

```text
outputs/scene-runs/run_YYYYMMDD_HHMMSS/
```

`scene_original.ply` is immutable. `scene_repaired.ply` is the cumulative
checkpoint used by both the GPU worker and the local viser/VLM harness.

## Loop

1. Render RGB and the bird's-eye path map at the current camera.
2. Ask the configured CliRelay VLM for one action.
3. Depending on `repair_trigger`, synchronously run image edit and GSFix3D:
   - `every_step`: the first `report_artifact`, then every observation
   - `every_artifact`: every `report_artifact`
   - `regenerate_yes`: only `report_artifact(..., regenerate="yes")`
4. Download the repaired PLY, hot-reload viser, and continue with the same VLM
   history, camera pose, path, and coverage state.
5. Stop at the earlier of the user deadline and LRZ reservation deadline.

The default path is Venetian Balcony, 960×720, RGB plus bird's-eye map,
one hour, Qwen-Image-Edit-2511, `regenerate_yes`, and the original GSFix3D
CUDA refine from [refine_gs.py](https://github.com/GSFix3D/GSFix3D/blob/main/scripts/gsfix3d/refine_gs.py):
20 photometric iterations per repaired view (0.8 L1 + 0.2 SSIM, densify every
5 steps). Repeated triggered views incrementally update `scene_repaired.ply`.
The dashboard **Repair type** dropdown can switch to `looped`, which repeats
those 20-iter chunks until **Repair Time** (default 180s) or Stop.

On a 128G / 1-GPU hold the remote worker keeps Qwen-Image-Edit-2511 on
`cuda:0` for the whole run. Each triggered view is edit → activation
`empty_cache` → GSFix3D, without reloading the ~40GB weights. Heartbeat
and `metrics.json` record host-cgroup and CUDA memory at those handoffs
(`memory.qwen_resident`). If that cgroup still OOM-kills, allocate 256G.
64G holds still spawn a disposable Qwen child before GSFix so the 62 GiB
cgroup does not OOM. Two GPUs are not required and do not overlap these
sequential phases; `image_edit.device: 1` is only a VRAM-isolation fallback
after a measured CUDA OOM.

## GPU ownership and recovery

One scene-run owns the selected allocation for its duration. Mutating repair
operations are blocked while that lease is active. Closing the browser does
not stop a run; use the dashboard Stop action or:

```bash
touch outputs/scene-runs/<run-id>/STOP
```

All phases and failures are persisted in `status.json` and `events.jsonl`.
If the GPU allocation expires, the latest complete PLY remains reviewable.
Model weights are never downloaded implicitly; preload Qwen into the remote
Hugging Face cache or explicitly enable the one-time download in configuration.
