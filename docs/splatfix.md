# Splatfix

`src/splat_explorer/splatfix/` separates view selection, GPT-image editing, and
GPU reconstruction. It is independent of the old artifact-hunting loop.
Both reconstruction modes use the authors' fresh ArtiFixer3D reconstruction
implementation, with its sparse MCMC/LPIPS recipe and 30,000 training steps.
Neither mode uses the extended pipeline's fixed-topology fitter.

## Select views and save edits

Run from the repository root with the project installed. `splatfix` is also
available as `python -m splat_explorer.splatfix.cli` or
`splat-explorer splatfix`. Global `--config` goes before the stage name.

```bash
python -m splat_explorer.splatfix.cli --config configs/splatfix.yaml select \
  --scene /path/to/scene.sog --output outputs/splatfix
```

The example config uses the existing CliRelay credentials and model settings.
`.env` is loaded without overwriting environment variables. The default project
config alone uses a scripted sweep for offline smoke tests; it does not perform
semantic VLM selection. `--renderer gsplat` selects CUDA rendering;
`cpu_splats` is the headless default. `viser` needs its existing viewer service.

The single loop selects six views by default; use `--views 8`, `10`, or `12` to
change that. Each VLM turn receives current RGB, a tiled selection image, and a
map. Finished tiles remain fixed; the active tile follows the camera. The tools
are, in order, `move_toward`, `rotate_around`, `move`, `rotate`, `finished`,
`view_map`, and `view_coverage_map`. The prompt asks for useful views, parallax,
and complementary multi-angle scene coverage. The floor coverage map is a
navigation aid, not a measurement of Gaussian reconstruction coverage.

`finished` records exactly the current RGB and camera, then starts selecting the
next view. `--max-steps` bounds actions per view (default 40); exhaustion raises
an error and retains the partial checkpoint instead of inventing missing views.
Only selected RGBs are written; intermediate RGB/map/tile images stay in memory.
`actions.jsonl` records actions and motion feedback.

Once all views are selected, the command sends each to GPT-image and saves its
repair. To select without image-edit calls, add `--select-only`. Complete the
image-edit stage separately, or retry it after an API failure:

```bash
python -m splat_explorer.splatfix.cli --config configs/splatfix.yaml edit \
  outputs/splatfix/run_<id>
```

Successful edits are reused on retry. Original RGBs are immutable inputs to the
baseline. Edits preserve the saved camera resolution; a response at another
resolution is resized only if its aspect ratio matches. A changed aspect ratio
is rejected. Metadata stores the prompt, backend, response dimensions and
completion state without saving another base64 copy of the image.

## Reconstruct on the GPU

Use a GPU environment with this project, PyTorch and gsplat for initial
trajectory rendering. The separate official ArtiFixer environment must contain
the authors' dependencies, 3DGRUT CUDA extensions, release checkpoint and cached
base model. The runtime paths are in `configs/splatfix.yaml`; they can be
overridden with `--artifixer-repo`, `--artifixer-python`,
`--artifixer-checkpoint`, `--artifixer-model-id`, and `--artifixer-hf-home`.
Provision according to the [official installation instructions](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/README.md).

The adapter requires a clean checkout at
`a392c4dfe17459ef9952407accdb9fcdcdddba98` and its matching 3DGRUT submodule.
It does not install models or modify upstream source while running. Model
downloads are disabled during reconstruction.

Copy the entire checkpoint directory to the GPU host. For its first trajectory
preparation, provide the same source scene there; `--scene` supplies its relocated
path. Its content fingerprint and recorded opacity/LOD settings protect against
mixing saved RGBs with a different source scene. After trajectory preparation,
the saved RGB/opacity/camera cache can be reused without loading the source.

```bash
# Original-method comparison: no external view-selection VLM or GPT-image calls.
python -m splat_explorer.splatfix.cli --config configs/splatfix.yaml repair \
  /gpu/runs/run_<id> --scene /gpu/assets/scene.sog \
  --mode baseline --output /gpu/reconstructions

# Same authors reconstruction, using the saved GPT-image repaired anchors.
python -m splat_explorer.splatfix.cli --config configs/splatfix.yaml repair \
  /gpu/runs/run_<id> --scene /gpu/assets/scene.sog \
  --mode edited --output /gpu/reconstructions
```

Every invocation creates a separate reconstruction directory, so either mode can
be rerun repeatedly on the same checkpoint. Reconstruction never invokes the external view-selection VLM or image editor.
First-time caption preparation runs the authors’ local Qwen model on the
original saved views, then encodes its caption with Wan UMT5. Edited mode requires all saved repairs; baseline works with
a completed `--select-only` checkpoint.

The caption stage calls the unchanged `generate_caption_hdf5` helper with
original anchors in checkpoint order, downsample factor 1, and its remaining
defaults. Six selected anchors become six submitted video frames at the helper’s
60 fps setting, producing one scene caption; the Qwen processor controls any
internal frame sampling. This is a montage of selected views, not a continuous
recorded video. One anchor uses the helper’s image-caption path. Edited images
never influence the caption, so baseline and edited runs share its exact
embedding, including the final plus pass.

`captions/<signature>/caption.h5` caches the authors’ uint16-encoded bfloat16
embedding beside a manifest. The cache key includes ordered original-image
hashes, helper defaults, upstream revision, and immutable Qwen/Wan snapshot
identities. The helper receives those exact snapshot paths. Every replay checks
the caption hash, shape, finite values, and nonempty text; corruption raises an
error rather than substituting zero conditioning. Hub blob content addresses
avoid rehashing large model weights. Copied model files without content-addressed
blobs are hashed once on generation and subsequently checked by size/mtime;
that validation limitation is recorded in the manifest. Both caption models
must already be provisioned locally. Snapshot resolution is local-only and
cache hits neither load models nor contact the Hub. Only the caption subprocess
allows normal model metadata access; the remaining worker phases retain their
configured offline policy. These local caption and encoding calls do not send
images to an external VLM/GPT service.

Selected cameras now define one continuous path through the authors' unchanged
`Renderer.interpolate_orbit_poses`: PCA orbit ordering, linear camera positions,
SLERP rotations, and the authors' default spacing `0.1 / metric_scale`, with
`loop=False`. Original anchor depth is prepared first for scale measurement;
the resulting path is then rendered and cached for both arms. At least two
distinct saved poses with shared intrinsics are required. Every anchor appears
exactly once in the full path. For reference-only inputs, the first anchor is
the start context and the remaining cameras are waypoints (two-camera inputs
use the authors' direct ordering). This input adaptation is recorded.
If the default spacing produces only anchors and no generated viewpoints,
the adapter retries the same authors' interpolator with smaller spacing.
The requested and effective spacing, and the reason for the adjustment, are
recorded in trajectory provenance.

Legacy `--frames` and `--span-fraction` are retained for CLI compatibility but
do not control the new path. Existing caches remain unchanged; the Python API
can explicitly select `trajectory_mode='legacy_local_loops'` for diagnostics.
Original anchor RGBs are referenced directly rather than copied. Saved camera
poses use OpenCV C2W. Before reconstruction, an explicit
index map removes repeated camera poses from the training catalogue, including
loop endpoints and overlaps across anchors. A saved reference takes priority
over a generated image at the exact same calibrated pose. Distinct nearby
cameras are retained without a distance tolerance; conflicting saved references
at one identical pose raise an error. `supervision.json` records every mapping.
The authors’ renderer then renders the original full trajectory so the plus
pass retains its original frame indices. This prevents duplicate supervision
in saved-view loops; it does not explain artifacts in the Bicycle benchmark,
whose source cameras are unique.
A single shared conversion supplies OpenGL C2W to both inference passes and
reconstruction, matching the authors’ preparation and reconstructed-COLMAP
loader. `--seed` defaults to 42. Scale is measured automatically before inference:
trajectory cache v2 saves original-anchor opacity and expected camera-Z depth,
then a separate measurement COLMAP samples at most 4,096 valid pixels per
anchor (opacity ≥ 0.8; at least 32 samples). Pixel-center backprojection supplies
observations to the unchanged official MoGe alignment. The depth is camera-Z,
not ray distance, as specified by [gsplat's depth implementation](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py).

The measured scale, diagnostics, original input hashes and sampling policy are
cached beside the trajectory and shared by both arms. Conditioning uses
`metric_scale * 0.01`; edited/generated images are never used for measurement.
This remains a scale estimate from rendered geometry, not photographic ground
truth. Measurement points never initialize reconstruction. Invalid geometry or
scale fails explicitly; there is no silent scale-1 fallback. Nonfinite ancillary
MoGe diagnostics are retained as null with their field names recorded.

Older saved view checkpoints remain usable: the first new repair rebuilds their
trajectory cache with depth from the original scene. Cached v2 runs can replay
without that source file. `--camera-scale VALUE` explicitly overrides measurement
and is recorded as a manual multiplier. MoGe weights must be provisioned offline;
`runtime.moge_model_path` optionally selects a local checkpoint file. Model
weights are hashed into scale-cache provenance. The photographic
Bicycle benchmark retains its separate official photographic preparation.


The baseline is the **authors' algorithm on our selected rendered inputs**.
It is not a reproduction of their photographic benchmark dataset. Where an
original COLMAP `points3D.bin` is available in the same scene coordinates,
provide `--artifixer-source-points3d /path/to/points3D.bin`. Otherwise source
Gaussian positions and RGB become a point-cloud initialization for fresh
3DGRUT training. Original Gaussian scales, opacities, and appearance parameters
are not continued. This input adaptation is recorded in each run.

Both arms call unmodified upstream diffusion inference, followed by
`data_processing.artifixer3d.train_artifixer3d` and `render_artifixer3d`—the
implementations behind the [authors' reconstruction entry point](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/data_processing/run_artifixer3d.py).
There are no exact-starter patches, periodic GPT refreshes, custom fitting
safeguards, or resumed source-splat checkpoints. The distinction between arms
is the saved anchor RGBs used as reference and reconstruction supervision.

## Outputs and verification

`checkpoint.json` stores selected views, camera calibration, source provenance
and repair metadata. `trajectories/` holds reusable camera/RGB/opacity bundles.
Each reconstruction directory contains `request.json`, phase logs, the fresh
3DGRUT checkpoint, `artifixer3d.ply`, and `result.json` (or `failure.json`).

The PLY is exported directly by the authors' exporter, retaining the trained
Gaussian parameters and spherical harmonics. It is never merged with the input
scene. ArtiFixer3D+ then applies diffusion to the fresh reconstruction's renders;
its output is the `plus_frames` directory recorded in `result.json`, not a
second PLY. This follows the paper's distinction between ArtiFixer3D and 3D+.

```bash
python -m splat_explorer.splatfix.cli inspect outputs/splatfix/run_<id>
python -m pytest tests/test_splatfix_checkpoint.py tests/test_splatfix_view_loop.py \
  tests/test_splatfix_cli.py tests/test_splatfix_repair.py
```

CPU tests cover checkpoint persistence, real-rendered command execution,
dynamic view counts, image inputs, camera validation, editor retry behavior,
trajectory reuse, and reconstruction dispatch. These do not establish visual
repair quality or substitute for a full run with CUDA and official weights.

## Reconstruction dashboard

Open **[/splatfix](http://localhost:8090/splatfix)** on the existing dashboard.
The dashboard and the existing `scene-run-manager` must both run:

```bash
splat-explorer --config configs/splatfix.yaml dashboard
splat-explorer --config configs/splatfix.yaml scene-run-manager
```

The usual `scripts/start.sh` also starts both services. Restart them after
updating the code when no jobs are active. The main workspace has four sections:

1. **Find views** selects the catalog scene and 6, 8, 10, or 12 views. It saves
   original calibrated RGBs without immediately making GPT-image edit calls.
2. **Saved checkpoint** selects reusable inputs, shows baseline and improved readiness,
   and links to original/repaired image pairs with saved camera and repair metadata.
3. **Repair images** creates missing GPT-image edits for a completed checkpoint.
   Successful saved edits are reused when retrying.
4. **Reconstruct** queues either the original-RGB baseline or the saved-image
   improved mode. Baseline can run before image editing; improved reconstruction
   requires every repaired image. Each job writes an independent isolated PLY.

Each runnable stage has **Run now** and **Schedule** controls. Scheduled times are entered
in the browser's local timezone and saved as UTC. Opening the schedule dialog
freezes the selected scene/checkpoint and options; changing another control
cannot silently change the queued inputs. They are earliest start times,
not reserved GPU allocations. Jobs survive web-server restarts and execute while
the existing queue manager is running. An offline manager and missing LRZ
configuration are shown explicitly. The new pipeline uses the same disk-backed
queue, cancellation markers, and cross-process compute lease as existing scene
runs; it does not create another scheduler or Codex automation.

Selection uses the local CPU splat renderer and the configured CliRelay VLM;
image edits use the configured GPT-image backend. Local stages can run without
an LRZ allocation. Reconstruction stages stage the checkpoint and source on the
configured LRZ/DSS workspace, launch inside the existing container/allocation,
and fetch logs, trajectory and metric-scale caches, caption caches, provenance,
PLY, and diffusion frames. Completed caption HDF5 files and manifests are
checksum-validated and atomically copied back into the canonical checkpoint,
then staged automatically for the next job in either mode. Temporary caption
source links and measurement-image links into old GPU jobs are omitted;
malformed caption caches fail explicitly. The
separate official ArtiFixer environment described above must already exist there;
the dashboard does not download models or provision that environment. Existing
GPU setup alone does not establish that the official environment is installed.

The queue is serialized. A running stage can delay other stages, including local
ones. A pending unavailable GPU does not prevent a ready local stage from being
chosen next. Cancellation stops a local stage's process group or sends the remote
worker its STOP marker. GPU cancellation is cooperative between rendering or
model-worker checks. Remote reconstruction identity is saved before launch;
a restarted manager reattaches to that job instead of launching another copy.
Unresolved remote connections stay queued for reattachment before other work.

Checkpoints under `outputs/splatfix/run_*` and dashboard jobs under
`outputs/scene-runs/run_*/checkpoints/` appear in the checkpoint selector. Job
configuration, status, logs, and events use the existing scene-run store.
Reconstruction downloads are under each job's `gpu/results/` directory.
ArtiFixer3D+ previews are the subsequent diffusion images, not a second PLY.

Dashboard scheduling, dependencies, cancellation, artifact boundaries, and worker
dispatch have offline tests in `tests/test_splatfix_dashboard.py`. These tests do
not execute paid image/VLM requests or validate a real LRZ/CUDA reconstruction.

### Published-dataset benchmark jobs

The dashboard also has an independent **Published dataset comparison** panel
inside the collapsed **Advanced** section below the main workspace.
It queues the authors' photographic/COLMAP baseline pipeline without VLM view
finding or GPT-image calls. This is a published Bicycle 3-view pipeline test;
the website's exact orbit and settings have not been confirmed.

Benchmark manifests record the pinned release inference defaults explicitly:
`kv_cache`, four denoising steps, seven latent frames per block, local attention
size 21, sink size 7, and one context-parallel process. Both diffusion passes
receive these arguments. These are verified release defaults, not recovered
settings from the unpublished website run. The 1.3B model remains the default.

For resolution diagnosis, `scripts/benchmarks/prepare_bicycle_resolution_control.py`
creates separate inference inputs from a completed 1.3B benchmark. It keeps the
full target order, poses, metric scale, caption, and initial fitted scene; it
resizes target renders, opacity maps, and references together and scales all
camera intrinsics. It verifies normalized calibration and records input/output
hashes. Its default 1088×720 is an experimental control, not a new baseline
default or a verified website resolution. Compare it separately from the
short-history control: changing both would confound their effects. Historical
random noise is unavailable, so neither comparison establishes causality alone.

A dataset appears after import registers
`outputs/benchmarks/<name>/input/benchmark.json`. The same input directory must
contain `colmap/images`, `colmap/sparse/0`, and `selected_images.txt`. Incomplete
registrations are visible but cannot be queued. Symbolic links are rejected so
staging only transfers the registered local input directory. The registry serves
only image and JSON artifacts from the benchmark output root.

The dashboard offers 1.3B (default) and 14B model choices with matched release
checkpoint/base-model pairs. **Run benchmark** and **Schedule** use the same
persistent queue, shared GPU lease, cancellation, and reattachment handling as
saved-view reconstructions. They do not provision weights. Existing scene-based
baseline/GPT-image comparisons remain independent.

API clients can queue a prepared source with `POST /api/splatfix/jobs`:

```json
{
  "stage": "benchmark",
  "source": "/absolute/project/outputs/benchmarks/bicycle/input",
  "mode": "baseline",
  "model": "1.3b"
}
```

Add a timezone-aware `scheduled_at` for delayed eligibility. Published-dataset
jobs stage inputs at `/workspace/splatfix-jobs/<job>/benchmark-input` and call
`run_benchmark` for official preparation, inference, fresh ArtiFixer3D, and the
3D+ image pass. Checkpoint-based `select`, `edit`, and `repair` contracts are
unchanged.

#### Bicycle source import

`scripts/benchmarks/prepare_bicycle.py` consumes the released ReconFusion
`mipnerf360/bicycle` directory (`images_4`, `transforms.json`, and
`train_test_split_3.json`) under `source/reconfusion/`, plus the original
Mip-NeRF360 Bicycle `sparse/0` binaries under `source/original/bicycle/`.
The public sources are [ReconFusion](https://reconfusion.github.io/) and
[Mip-NeRF360](https://jonbarron.info/mipnerf360/). The three released training
indices are 2, 28, and 110; the 25 published test indices are recorded separately.

```sh
python scripts/benchmarks/prepare_bicycle.py \
  --source outputs/benchmarks/bicycle/source \
  --output outputs/benchmarks/bicycle/input
```

The importer keeps JPEG bytes unchanged, uses the released intrinsics/poses,
verifies a single global similarity against the original COLMAP poses, and
transforms the sparse points to that coordinate system. Measured pixel
observations are scaled to the released image dimensions for MoGe alignment.
It records input SHA256 hashes and refuses to overwrite an existing input.

The default benchmark uses the authors' orbit interpolation through the three
reference cameras and the 25 published test poses, preserving exact test nodes
for scoring. Only novel targets enter generated supervision; photographic
references are appended as contexts by the official preparation entry point.
`split_trajectory.json` records this inference catalogue, while `split.json`
preserves the source-camera preparation for reuse. Runtime option
`trajectory_mode='source_cameras'` retains the earlier filename-order diagnostic.
The benchmark is an execution test with the published three reference photographs,
not proof of matching the unpublished website orbit or the paper's complete
benchmark protocol. The run manifest retains this distinction; do not report
metrics over all 191 non-reference images as the published 25-image test split.
The website model variant, caption, random state and initialization remain to
be established before matching settings can become verified splatfix defaults.

While GPU stages run, their current phase log is mirrored as a UTF-8-safe tail
of at most 64 KiB, fetched no more often than every 15 seconds. The dashboard
links that snapshot before completion and the full phase log after final
transfer. Log observation failures do not interrupt reconstruction.

A failed published-dataset benchmark can be retried as a **new job** with
`resume_from` set to its prior job ID. Source and model must match. The current
resume contract requires successful `prepare`, `reconstruct`, `render`, and
`scale` stages and a saved prepared Bicycle tree; earlier or cancelled
preparations are not reusable. The dashboard shows **Resume in new run** only
when this readiness is recorded or established from the prior manifest.
The executor resolves one supported failed result within the prior remote job,
then passes its container path to the benchmark runner. The runner copies
prepared artifacts into a new result directory and lets the official stages
choose reuse. Prior job records and outputs remain unchanged.

For model comparisons, use `preparation_from: "<prior job ID>"` instead of
`resume_from`. The prior benchmark must be completed or failed, use the same
registered source, and have completed preparation, base reconstruction,
rendering, scale alignment, and captioning. **Compare model using this
preparation** uses that prior run's source and the model currently chosen in the
benchmark panel; both are captured when clicked. The new job preserves the prior
outputs and reuses a copied preparation. The worker validates tokenizer and text
encoder identity before sharing caption embeddings across model variants.
`preparation_from` and `resume_from` cannot be combined. Running and cancelled
jobs are not preparation sources.

Benchmark downloads retain RGB renders, opacity, checkpoints, PLYs, logs and
evaluation/provenance records. Dense depth maps under the prepared benchmark
tree remain on LRZ; they are regenerable diagnostics and are not used by
inference or evaluation. `artifact-download-policy.json` records this policy.
Saved-view repair downloads keep their anchor depth because scale replay needs it.


### Source-render fidelity correction (2026-10-06)

Saved-view selection images can come from the approximate `cpu_splats` renderer.
They are navigation previews, not faithful baseline supervision. Previously,
repair preparation discarded GPU-rendered anchor RGB and reused those previews,
while keeping GPU alpha/depth and surrounding frames. Both Venetian orbit runs
therefore used six approximate CPU images as their reference supervision.

Repair now saves source-scene RGB from `BundleRenderer` at every selected camera
and uses it for baseline references, scale alignment, captioning and trajectory
frames. The same change applies to legacy local loops. Trajectory recipe version
4 prevents reuse of old mixed-renderer caches. `selection_rgb` preserves preview
provenance; existing checkpoint images and past results are not modified. Edited
mode still intentionally uses its explicit saved edits as references. Published
photographic Bicycle benchmarks use their separate preparation pipeline.

This fixes the reference-source mismatch; it does not establish pixel equivalence
between CUDA gsplat and the interactive viewer. Verify fresh source renders before
launching another full reconstruction.
