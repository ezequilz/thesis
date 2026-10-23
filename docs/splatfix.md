# Splatfix

`src/splat_explorer/splatfix/` separates view selection, GPT-image editing, and
GPU reconstruction. It is independent of the old artifact-hunting loop.
Step 4 selects a reconstruction method. ArtiFixer and G4Splat are registered as independent sibling backends.
See [G4Splat setup and output semantics](g4splat.md).

Step 3 defaults to **Supply initial repair as context** for all checkpoint types.
The first image is repaired from its original RGB. Each subsequent request sends
its original RGB first and the first repaired PNG second, with the same prompt.
For normal checkpoints, **Prior repair as context** instead supplies the immediately
preceding repaired image (image 2 supplies context for image 3, and so on).
**RGB only** preserves independent single-image repairs. Context stays within the
selected capture set. The option is saved in queued jobs as
`endpoint_repair_context` (`initial_repair`, `prior_repair`, or `rgb_only`) and is
also available through `edit --endpoint-repair-context`. Retries reuse saved
repairs; completed repairs remain skipped.

### Manual repair framing

In the saved-checkpoint popup, select a GPT image and choose **Resize for input**.
Drag the rectangle to move it or a corner to resize it. The selection stays 16:9
and cannot shrink below 1280 × 720 source pixels. **Save crop** preserves the full
GPT response and prior repair, saves `manual_resized.png` at the selected pixel
size, and creates resolution-specific derivatives. Reopening the editor always
loads the full response, so adjustments never accumulate cropping or resampling.
The opacity slider blends the GPT layer over the original RGB for alignment;
the RGB stays fixed while the GPT layer pans or zooms. After releasing a crop
handle, the preview waits 450 ms and fits the selected region into the frame.
Handles remain editable outward after zooming. **Show full repair** restores the
full-source view without changing the crop. Opacity and preview zoom are display
settings only and are never baked into the saved repair.

Step 4 uses the saved manual crop before its usual resolution transform. Without
a manual crop, it retains the existing full-response path (or saved repaired PNG
for legacy checkpoints). The existing 480p training profile is 832 × 480 and
retains its calibrated center-fit policy. Original RGBs and camera poses are not
changed by manual repair framing.

### Reconstruction module boundary

`methods.py` is an explicit, lazy-loaded registry. The dashboard lists its public
metadata, validates checkpoint readiness through the selected backend, and saves
`reconstruction_method` in the queued job. `executor.py` dispatches GPU execution;
`job_worker.py` independently resolves the same method on the GPU host. Unknown
method IDs are rejected. Jobs and worker requests predating this field default to
`artifixer`. Scheduled jobs retain the selection made when they were queued.

The implementation lives in `src/splat_explorer/splatfix/artifixer/`:

- `backend.py`: public adapter and method-specific option/readiness validation.
- `execution.py`, `worker.py`: LRZ staging, reattachment, cancellation, downloads,
  and ArtiFixer stage orchestration.
- `repair.py`, `author_trajectory.py`, `segmented_inference.py`,
  `official_worker.py`: saved-view adaptation, trajectory generation, autoregressive
  inference, fresh reconstruction and the subsequent 3D+ pass.
- `benchmark.py`, `repeat_benchmark.py`, `repeat_render.py`, `resume.py`: published
  dataset comparisons and reuse of prior preparations.
- Evaluation, trajectory diagnostics, preview/log mirroring, interrupted-result
  preservation, graceful trainer shutdown, 3D+ captures and `python_compat/` also
  belong to this backend.

Checkpoints, view finding, image editing, resolution transforms, rendering and
Viser capture transport remain shared. Existing top-level module names are thin
compatibility entry points, preserving imports and persisted executable paths;
there is no second copy of the implementation. Existing output directories and
trajectory/caption caches keep their layout and signatures. The application-wide
LRZ transport and legacy scene-run integrations outside `splatfix/` are reused.

### Adding another method

Create `splatfix/<method>/backend.py` and register its fixed module path, ID, label,
description and supported `stages` in `methods.METHODS`. Its adapter implements:

- `validate_options(raw) -> dict`: validate and return only its own settings.
- `readiness(checkpoint) -> {ready, reason}`: requirements for saved cameras.
- `execute_remote(executor, run_id, options, root, stop, update)`: stage inputs and
  runtime, persist the chosen method in `worker-request.json`, execute/reconnect,
  publish progress, and download results. The shared executor supplies `cfg` and
  `store`; the queue supplies scheduling and the exclusive GPU lease.
- `execute_worker(root)`: read the staged request, honor STOP/deadline handling,
  execute its algorithm and publish `worker-status.json` (running/completed/error/
  stopped). Evaluation and partial-result preservation belong to the backend.
- `run_repair(checkpoint_dir, output_dir, **kwargs)`: local CLI entry point.

Use `Checkpoint.load()` and `camera_from_record()` for the common input contract:
OpenCV camera-to-world poses, calibrated intrinsics, and
`checkpoint.image_path(view, repaired=mode == 'edited')` for RGBs. Source splat,
depth, opacity and generated trajectories are optional backend inputs; another
method does not have to use ArtiFixer's orbit or inference pipeline. Keep new
method caches separate from the legacy ArtiFixer trajectory/caption caches.

Return isolated outputs under `gpu/results/<run>/result.json` with `splat_path`
and `output_dir`; record the method ID and input provenance. The existing result
catalog supports this common PLY contract and retains legacy ArtiFixer artifact
fallbacks and optional inference/3D+ galleries. Add method-specific controls or
additional result galleries when the new algorithm needs them. The advanced
published-dataset benchmark UI remains ArtiFixer-specific. The local CLI accepts
`--reconstruction-method`; its current tuning flags describe ArtiFixer and should
be extended alongside a future method's runtime options.

After updating a running installation, restart the dashboard and queue manager
so their Python processes load the registry and dispatcher. Refreshing the page
alone updates HTML but cannot reload an already-imported Python backend.

### ArtiFixer reconstruction

Both reconstruction modes use the authors' fresh ArtiFixer3D reconstruction
implementation, with its sparse MCMC/LPIPS recipe and 30,000 training steps.
Neither mode uses the extended pipeline's fixed-topology fitter.

The single reconstruction default is `apps/colmap_3dgut_sparse_mcmc_lpips`
from the pinned ArtiFixer checkout and its matching 3DGRUT submodule. Saved-view
repairs, photographic benchmarks, and repeat repairs all select this recipe
explicitly for fresh 30,000-step training. New run manifests record its identity
as `reconstruction_recipe`.

At the pinned revision this uses opacity-based birth and relocation sampling,
5% growth every 100 steps after warmup up to one million Gaussians, relocation
through the upstream 25,000-step cutoff, and perturbation through the 27,500-step
cutoff. Opacity/scale regularizers remain disabled in this sparse recipe;
generated targets use its LPIPS loss settings. The authors' pinned sparse recipe
inherits `base_gs_sparse -> base_gs` and switches the strategy to MCMC. Both
`base_gs` and the separate regularized `base_mcmc` are upstream configurations;
SplatFix does not replace that inheritance with `base_mcmc`. No custom loss
adapter or error-based allocation is used. Exact recipe replication does not
remove the documented SplatFix input adaptations (rendered or edited anchors
and full source PLY initialization for saved-view repairs).

### Generated-image reconstruction losses (step 4)

ArtiFixer repair runs expose two independent controls in all three pipeline modes:

- **RGB L1/SSIM contribution for generated images** defaults to off
  (`generated_rgb_loss=false`, `lambda_reconlosses_override=0`). On sets the
  multiplier to 1 and copies the reference-image L1/SSIM weights: 0.8 / 0.2.
- **LPIPS loss downscale for generated images** defaults to on
  (`generated_lpips_downscale=true`, `lambda_lpips_override=0.1`). Off uses 1.0,
  retaining LPIPS without downscaling.

Thus generated-image loss is `RGB_enabled × (0.8 L1 + 0.2 (1 − SSIM)) +
LPIPS_weight × LPIPS`, plus any separately enabled Gaussian regularization.
Reference-image losses are unchanged. These choices survive the authors pipeline
preset enforcement, are saved in job/runtime settings and reconstruction recipes,
and affect the fitting configuration recorded by the trainer. Historical jobs
without these fields retain the original ArtiFixer weights. Benchmark controls
are unchanged.

The [original MCMC training code](https://github.com/ubc-vision/3dgs-mcmc/blob/main/train.py)
uses RGB L1/SSIM and Gaussian regularization, with default SSIM weight 0.2; it
has no LPIPS training term. Full-weight LPIPS is therefore an ArtiFixer extension,
not a reproduction of the complete original MCMC objective.

### Reconstruction regularization comparison

The dashboard's **Reconstruction regularization** selector applies to new repair
and benchmark runs (including scheduled runs). **ArtiFixer sparse — off** is the
default described above. **MCMC regularization — on** inherits the authors'
`base_mcmc` opacity and scale penalties, both weighted 0.01 at the pinned revision.
Only those four enable/weight settings are inherited; initialization, opacity-based
allocation, image losses, and training schedules remain fixed for comparison.

The on option is a comparison variant, not the exact sparse ArtiFixer recipe.
The pinned generated-view LPIPS branch omits regularization from its total loss;
a scoped adapter adds the already-computed weighted terms for generated views.
Anchors already include them, so they are not added twice. Upstream files remain
unchanged. The penalties encourage lower opacity and smaller scales, which may
reduce unnecessary or oversized Gaussians but can also suppress useful detail.

The queue, staged worker request, and run recipe record `regularization_profile`
(`artifixer` or `base_mcmc`). The trainer's `parsed.yaml` records effective weights.
Historical runs without a selection retain the off behavior. To reuse benchmark
preparation, choose the option and use **Compare settings using this preparation**;
resume preserves the original run's option. This comparison regenerates inference
and is not a strict identical-target or identical-randomness ablation.

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
semantic VLM selection. Selection uses the harness Viser capture service; keep
the capture visor visible for selection. Step 4 defaults to **Authors 3DGUT baseline**:
the pinned upstream renderer supplies paired source RGB and opacity and retains
the native reconstructed RGB/opacity for ArtiFixer3D+. It does not need a Viser
capture connection during reconstruction. No VLM is involved during rendering.
**Legacy Viser / gsplat** restores Viser RGB with gsplat opacity/depth and the
Viser RGB replacement before the ArtiFixer3D+ pass.

Editing and edited reconstruction reject checkpoints whose selection renderer is
not Viser or is unrecorded; regenerate those selections and edits. Baseline repair
can reuse old camera poses, but renders fresh anchors with the selected backend.
Renderer provenance versions invalidate old trajectory, scale and caption caches.
Published photographic benchmarks retain their separate authors' pipeline.

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

## Resolution policy

**v0 · Early original resolution** is preserved as `early_original` in the
resolution setting selector and `splatfix.resolution_profile`. It restores the
previous 960 × 720 camera-capture default, leaves saved checkpoints unchanged,
and copies photographic/COLMAP inputs byte-for-byte without a resolution cap.
For the original bicycle data, saved images remain 1237 × 822; the authors'
loader works internally at 1232 × 816 and saves back at the source dimensions.
This is the early Splatfix resolution setting, not an authors-endorsed preset.
Only resolution behavior is versioned; other current pipeline settings remain
in effect. Historical unversioned runs are labelled separately in run details.

**v1 1280 x 720 (default)** (`720p`) is the default.
**v1 832 x 480** (`training`) remains available as the smaller alternative.

New Splatfix camera captures render directly at **1280 × 720** by default, or
**832 × 480** with `splatfix.resolution_profile: training`. Both dimensions in each
profile are divisible by 16, so upstream alignment does not resize these frames.
832 × 480 is 26:15, not exact 16:9; 1280 × 720 is exact 16:9.
The smaller size follows the [official Wan2.1 T2V-1.3B size configuration](https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/__init__.py),
not a demonstrated optimal ArtiFixer training resolution or required aspect ratio.

The former 960 × 544 default was inspired by rounding typical 960 × 540 DL3DV
frames to multiples of 16. Existing saved captures remain at their recorded size.
The internal resolution-policy revision is now 2 to prevent reuse of prepared
inputs under the old bounds; the user-facing profile labels remain v1.
Dashboard job requests can select `training`, `720p` or `early_original` through
`resolution_profile`; direct Python repair/benchmark calls use that runtime key.
Other explorer pipelines retain their own resolution settings.

Step 4 creates an isolated checkpoint at the exact selected profile size, even
when the saved selection used a smaller resolution. Original RGBs are recaptured
through the selected renderer at the saved poses with a uniformly scaled, center-cropped camera
frustum. Subsequent trajectory captures use these same dimensions and intrinsics.
Edited references are generated from the saved full GPT response using uniform
scaling and center cropping, without padding or stretching. Legacy checkpoints
without a saved response use their repaired PNG as a fallback. Source checkpoints
and full responses remain unchanged; no additional GPT call is needed.

ArtiFixer stores reusable source captures under the saved checkpoint's
`artifixer-inputs/<signature>/` directory, including a capture manifest and camera
records. The signature includes the source fingerprint, calibrated cameras,
resolution profile and scene-loading settings. Completed trajectory RGBs,
opacity/depth, camera transforms, scale alignment and caption caches are published
there as preparation proceeds and restored into subsequent run-local checkpoints.
LRZ downloads return these caches to the original saved checkpoint for the next
run. Edited targets are read fresh, so changing an edit does not recapture the
source. Different interpolation recipes retain separate trajectory signatures.
Checksums reject damaged caches. Captures of the newly reconstructed ArtiFixer3D+
scene still run per reconstruction; source/checkpoint uploads to LRZ also remain.
Existing checkpoints populate this cache on their next run.

Photographic benchmark inputs use a separate bounds-based policy: fit within
the profile without upscaling, uniformly scale, then symmetrically crop fewer
than 16 output pixels per dimension for alignment. Intrinsics and COLMAP image
observations receive the same transform; world poses and points stay unchanged.
Bicycle's 1237 × 822 photographs therefore become **720 × 480** (`training`) or
**1072 × 720** (`720p`). Forcing those photographs into a widescreen aspect would
change their framing substantially. Derived COLMAP images use lossless PNG
payloads under the published filenames to retain split identity.

Preparation writes derived inputs inside the new result directory. Original
photographs, checkpoints and historical results remain untouched. Benchmark
preparation from an older or different policy is rejected; run fresh preparation.
Evaluation applies the recorded transform to photographic ground truth and
requires all four stages to have exactly those dimensions. These scores should
not be compared directly with historical scores at a different resolution.
This preprocessing adaptation is recorded explicitly; upstream source and
inference/history settings are unchanged. A GPU quality comparison is still
needed to establish whether resolution reduces autoregressive drift.

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
Step 4 exposes **ArtiFixer prompt source** independently of reconstruction input images:

- **authors Qwen3-VL/UMT5 on original anchor captures (baseline)** preserves the existing caption recipe and cache. Original Viser anchors alone determine the caption, including in edited reconstruction mode.
- **Authors Qwen3-VL/UMT5 on original anchors + repaired GPT input references** passes the ordered original anchors followed by the repaired references used by this run to the same authors’ `generate_caption_hdf5` helper. This requires edited inputs; endpoint runs include only their selected repaired endpoint references.
- **Original WorldExplorer scene prompt · UMT5 encoding (default)** directly encodes the exact saved scene prompt using the authors’ UMT5 tokenizer and embedding helper, without Qwen caption generation. A missing prompt raises an error rather than falling back.

WorldExplorer already archives the submitted prompt in `request.json`. New checkpoints also store it in `metadata.worldexplorer_prompt`; older checkpoints recover it from the scene’s archived request when selected for this mode. This metadata travels with the checkpoint to remote workers. CLI runs accept `--prompt-source original_anchors|original_and_repaired|worldexplorer`.

Visual captioning retains the authors’ single-image or 60 fps ordered-video preparation, with frame sampling controlled by the Qwen processor. The additional modes have separate content-addressed caption recipes, including repaired-image checksums or the exact scene prompt. Prompt selection does not change MoGe alignment against original rendered geometry; depth is not a direct reconstruction loss.

`captions/<signature>/caption.h5` caches the authors’ uint16-encoded bfloat16
embedding beside a manifest. The cache key includes ordered original-image
hashes, helper defaults, upstream revision, and immutable Qwen/Wan snapshot
identities. The helper receives those exact snapshot paths. Every replay checks
the caption hash, shape, finite values, and nonempty text; corruption raises an
error rather than substituting zero conditioning. Hub blob content addresses
avoid rehashing large model weights. Copied model files without content-addressed
blobs are hashed once on generation and subsequently checked by size/mtime;
that validation limitation is recorded in the manifest. Required models must already be provisioned locally (UMT5 only for WorldExplorer prompt mode; Qwen and UMT5 for visual captioning). Snapshot resolution is local-only and
cache hits neither load models nor contact the Hub. Only the caption subprocess
allows normal model metadata access; the remaining worker phases retain their
configured offline policy. These local caption and encoding calls do not send
images to an external VLM/GPT service.

Selected cameras define camera legs through the authors' unchanged
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

Both diffusion passes support `split_mode` in the runtime/job settings and
`--split-mode` on `splatfix repair` and the trajectory adapter. The dashboard's
**Trajectory splitting** selector exposes both modes:

- **single-split** preserves the previous behavior: one forward series per
  adjacent pair in the authors' ordered input waypoints.
- **double-split** is the default: each reference-to-reference leg's generated
  positions are divided into two halves. The first half runs forward from the
  first reference; the second runs in reverse from the next reference toward
  the midpoint. With five target positions, generation orders are
  `reference A → 0 → 1 → 2` and `reference B → 4 → 3`.

An odd target count assigns the extra position to the first half. Empty halves
are skipped (a one-target leg needs only one series). There is no duplicated
midpoint. Every nonempty half begins with its trusted endpoint RGB as rendered
conditioning, opacity 1, and the matching camera. This frame is excluded from
PNG export and generated supervision: reconstruction uses the trusted anchor
image directly. With **Image cache insertion** off (the default), the upstream denoiser is unchanged;
this supplies a clean input image rather than clamping the first generated latent.
Turn it on in the repair dashboard, or pass `--image-cache-insertion`
(`--no-image-cache-insertion` disables it; runtime key: `image_cache_insertion`).
Both repair diffusion passes then reuse `starter_inference`: preserve the VAE-encoded
endpoint as the first latent, insert it into KV memory at timestep zero, and
generate subsequent blocks with clean timestep-zero cache refreshes, using
a 21-latent rolling window and one persistent starter sink, configured on
both the pipeline and transformer at construction. Each cut clears the caches
and uses its own resolution-prepared endpoint image. The encoded starter is
preserved exactly; decoding it still has the VAE's normal reconstruction loss.
Each new insertion run also checks exact latent equality after sampling and writes
`inference-insertion-audit.json` (or `plus-insertion-audit.json`) plus
`SPLATFIX_STARTER_VERIFIED` log records. These record the repaired input hash,
seed camera, generation/export indices, and first generated latent/RGB offset.
Missing runtime confirmation fails inference. The starter is not an exported
prediction: subsequent generated views and the fitted splat are not constrained
to reproduce its pixels exactly.
This works with either split mode; single-split also prepends its trusted starting endpoint
when enabled. References and direct anchor supervision are retained. The setting
applies to baseline as well as GPT-repaired inputs; published benchmarks are unchanged.

The benchmark's held-out test cameras are never trusted seed images. Double-split
groups the existing path between its photographic reference cameras; test
waypoints remain targets inside those spans. A non-looping path's open tail
beyond its last reference is one forward series from that reference. Its
geometry is preserved rather than inventing a reference at the final test view.

The adapter obtains camera ordering from the authors' helper without
interpolation and does not re-sort individual pairs. The full camera catalogue
and global PNG indices stay fixed for reconstruction and evaluation, even when
inference order runs backward. Provenance records mode, directions, target
indices and seed cameras; these also distinguish trajectory caches. Historical
repeat-benchmark jobs retain their recorded inference mode. Geometry and the
authors' PCA ordering are unchanged, so this change does not flatten a
zigzagging camera path.

There is no fixed intermediate-frame-count option in this helper. For each leg,
the number of intervals is `max(1, ceil(hypot(translation_distance,
rotation_distance) / (spacing / metric_scale)))`, with default spacing 0.1.
Rotations use the authors' default weight of 1. Endpoint ownership and reference
exclusion determine the final generated-frame count. Short series still use
upstream temporal padding, which is trimmed before saving PNGs.

The saved-view worker explicitly selects `--render_trajectory trajectory` and
clears caches before each leg. Every leg receives **all saved GPT-repaired
images** in edited mode (six for a six-view run); the saved-view baseline uses
all corresponding original renders. The photographic benchmark supplies its
three original photographs to every leg through `segmented_inference.py` and
the authors' dataset-factory entry point. Model weights load once per diffusion
pass; each leg gets a separate upstream inference call and fresh caches. Both
the first inference and the ArtiFixer3D+ pass use this policy. Trajectory cache
recipe v5 prevents reuse of the old single-series manifest.

History was already bounded in these SplatFix paths. The pinned authors'
[KV pipeline](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/pipeline/kv_cache_pipeline.py)
and [attention processor](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/net/transformer.py)
use 7 latent frames per block and 21 attention-cache positions: a permanent
initial 7-frame sink plus 14 rolling positions, including the current block.
These are latent frames, not RGB frames. The retained KV entries come from the
last transformer call of each block; this release does not perform an additional
clean-latent timestep-zero cache write after the final scheduler step. Reference images are separate neighbor
conditioning, not the sink. New items initialize KV/text/neighbor caches;
decoding clears them. Splitting limits autoregressive error propagation even
though the previous long series did not attend to its entire generated history.

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


The baseline uses the **authors' reconstruction machinery on our selected rendered inputs**.
It is not a reproduction of their photographic benchmark dataset. Saved-view
runs from an original PLY now snapshot its exact bytes into `initialization.ply`
and use the pinned 3DGRUT `import_ply` path. Positions, scales, rotations, opacity,
and degree-3 SH are retained; DC-only files are accepted with zero higher-order
SH. Unsupported partial SH layouts fail explicitly instead of dropping appearance.
The source stays in the same coordinates as the saved cameras, without normalization.

The importer creates a new optimizer and starts at `import_ply.init_global_step=0`.
The existing 30,000-step sparse MCMC/LPIPS schedule, including Gaussian birth and
relocation, remains unchanged. `base_checkpoint=None` is intentional: this loads
full Gaussian parameters without resuming the old optimizer or iteration count.
This is an upstream-supported initialization variant, not the authors' default
fresh-from-points experiment. It does not freeze or guarantee preservation of
unobserved surfaces. A CUDA render comparison is needed to verify import fidelity.

Set `splatfix.runtime.reconstruction_initialization: point_cloud` to restore the
previous initialization path; the default is `full_splat`. In the fallback,
`source_points3d` selects original COLMAP points when supplied; otherwise source
Gaussian positions/RGB seed fresh training. Non-PLY assets retain that fallback.
Photographic benchmarks keep their original COLMAP initialization. Both paths
retain `base_checkpoint=None`. Requests/results record initialization, fresh
training state, and the copied PLY hash; changed PLY snapshots are rejected before
GPU reconstruction. The PLY snapshot bypasses our viewer loader, which drops SH.

By default, both arms call unmodified upstream diffusion inference, followed by
`data_processing.artifixer3d.train_artifixer3d` and `render_artifixer3d`—the
implementations behind the [authors' reconstruction entry point](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/data_processing/run_artifixer3d.py).
By default there are no exact-starter patches. Repair's optional image cache
insertion enables the existing exact-starter adaptation on the inference instance.
There are no periodic GPT refreshes, custom fitting safeguards, or resumed
source-splat optimizer checkpoints. The distinction between arms
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

The usual `scripts/start.sh` also starts both services, and reloads them
when `src/` or `configs/` change. That restart interrupts an active job, so
avoid a code-changing start while a reconstruct is running. `--force` restarts
them even when those files are unchanged. The main workspace has four sections:

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

### Consecutive benchmark repair

A benchmark job can set `repeat_from` to a completed benchmark job ID (same
source and model, mutually exclusive with `resume_from`/`preparation_from`).
The worker copies the original orbit and conditioning. The prior ArtiFixer3D
checkpoint replaces the simple reconstruction input. The requested resolution
profile calibrates references and intrinsics and freshly renders RGB/opacity with
the pinned upstream renderer, preserving every original camera pose. The
`early_original` profile can reuse historical RGB/opacity without resizing. Original photographic references,
trajectory, caption, camera scale and COLMAP point initialization stay fixed.
The native checkpoint represents the same splat as its exported PLY; avoiding a
PLY roundtrip preserves all trained parameters and the authors' renderer output.
Inference, fresh 30,000-step reconstruction, export and the plus image pass run
again. Old diffusion predictions are not used as the new predictions. The output
manifest records parent hashes and repair pass number. Authors' stochastic
inference/training behavior is retained; this is not a deterministic paired trial.
Legacy manifests without `inference_settings` replay the recorded completed
inference/plus commands, preserving defaults from the pinned upstream revision.
The `training` profile uses v1 sizing (720 × 480 for Bicycle); it does not reuse
old full-resolution RGB/opacity as inference inputs.

### Per-run trajectory and quality chart

Every completed Splatfix dashboard reconstruction now runs an evaluation phase.
Benchmark jobs attempt the pinned photographic PSNR/SSIM/LPIPS evaluation in the
configured ArtiFixer environment. Each run writes `trajectory-quality.png` and
`trajectory-quality.json` from its actual inference cameras and available
per-image metrics. Saved-view jobs without photographic ground truth show only
camera motion. Independent inference segments are not connected across cuts.
Height uses the mean source-capture camera up vector where available, otherwise
reference camera up; the vector and its source are recorded in the JSON.

The run detail page's **Stage comparison** gallery has previous/next arrows,
keyboard navigation, a position label, and full-size image links. It includes
both the saved stage image and the new evaluation chart when present. The chart
has no hypothetical ellipse overlay. `evaluation-status.json` records partial
failures, which the page reports without discarding a completed reconstruction.
Missing metric weights must be provisioned before photographic evaluation can
succeed; geometric charts require no GPU or extra plotting libraries.

Existing downloaded results can be backfilled without rerunning inference:

```sh
python -m splat_explorer.splatfix.evaluation_chart /absolute/path/to/result
```

### Graceful stop before GPU expiry

Splatfix reconstruction workers save the current Gaussian model when stopped,
without adding periodic checkpoints. A process-local wrapper around the pinned
trainer checks a stop marker after each completed optimizer iteration, invokes
the authors' checkpoint writer and PLY exporter, flushes diagnostics, then exits.
The upstream checkout, optimization recipe and normal checkpoint schedule remain
unchanged. `*.lifecycle.json` records this runtime adaptation.

The queue manager passes its effective deadline to the remote worker. Both the
manager and remote worker request a stop **300 seconds before GPU expiry** by
default, so a disconnected desktop does not disable the worker's deadline.
Manual dashboard Stop uses the same save path. The defaults in
`configs/default.yaml` are `splatfix.stop_before_gpu_end_seconds: 300` and
`splatfix.save_stop_timeout_seconds: 240`; the save timeout must be shorter than
the deadline buffer. Slurm end times without an offset use Europe/Berlin.

During training, cancellation waits for the iteration-boundary save instead of
immediately sending SIGTERM. A hung save is terminated after the bounded grace
period, with an explicit timeout diagnostic. Sudden GPU loss, OOM or forced
external termination still cannot guarantee recovery. Other phases preserve
available files but have no trainable splat to save.

`<phase>.interrupted/stop-state.json` records the actual saved iteration,
checkpoint path, PLY path and any save/export error. The model checkpoint is the
authors' format; exact optimizer/RNG continuation is not guaranteed. The
`partial-result.json` record exposes saved intermediate geometry in Results as
**Interrupted · incomplete**, alongside available inference images and logs.
A checkpoint can remain recoverable even if PLY export fails. No completed
`result.json` is fabricated for an interrupted reconstruction.

### Cache the connected allocation deadline at setup

**Load GPU setup** on `/gpu` reads Slurm's end time in its existing setup check
and, after successful setup, atomically saves `outputs/gpu-allocation.json`.
This local record is bound to the job ID, host, user and workspace. It includes
an absolute UTC deadline and survives dashboard/manager restarts. The manager
and launch validation use this record instead of polling Slurm for readiness
or deadlines. No deadline countdown is extended by rereading the cache.

For each newly selected GPU job, load setup once. Missing, mismatched or expired
records leave runs waiting for setup; they never reuse another allocation's
remaining time. Reloading setup explicitly refreshes the saved deadline if an
allocation was extended. The existing five-minute save-and-stop buffer is
applied to this deadline and passed to the remote worker. Explicit dashboard
GPU probes remain available, but do not replace the setup-owned deadline.

The results dashboard lists benchmark and repair jobs immediately, including queued
and failed jobs without outputs. Run visibility is independent of artifact detection
or file arrival. The initial details link continues to work after outputs arrive.
Live run details offer **Stop gracefully & download**, using the existing cooperative
stop contract to save and download available reconstruction outputs.

After first-pass autoregressive inference finishes, benchmark, repeat and saved-view
repair workers publish `inference-preview.json` with the exact generated
reconstruction-input indices. The manager downloads those images once in a
background transfer (capped at 10 MiB/s), while reconstruction continues on the GPU.
Small camera JSON records accompany the images so the desktop can draw the trajectory
chart immediately. No early image-quality scoring is performed; those scores remain
part of the existing end-of-run evaluation.
The gallery becomes visible only after that transfer succeeds. Its result URL stays
stable when the final PLY, evaluation and other artifacts arrive through the normal
end-of-run download. Both result pages refresh every 15 seconds; detail refreshes
pause while the image lightbox is open. Early-transfer failures are recorded in
`inference-preview-transfer.log` and retried after 30 seconds without stopping
reconstruction; the final download
still retrieves the outputs. A successful preview is reused after manager restart.
Workers launched before this change do not publish the early-preview marker.


### Manual view finding

On `/splatfix`, choose the source scene, capture resolution, and view count, then
click **Custom** beside **Schedule**. The shared Viser loads that scene and opens
an ephemeral **Custom view finding** panel. Navigate in the viewer and click
**Save current position** for each view. Each click saves the rendered RGB and
its calibrated OpenCV camera pose to a normal checkpoint; no VLM job is queued.
The clicked Viser tab supplies both the pose and capture, even with other tabs open.

Click **Save selection** to finish and remove the panel. You can finish early;
the checkpoint then uses the number of views actually saved. The dashboard
selects this checkpoint and refreshes its images and downstream readiness.
Duplicate positions are rejected. If the shared viewer switches scenes, captures
are refused until the requested scene is ready again. **Close (keep saved views)**
removes the controls without deleting an incomplete selection. This feature
requires both the dashboard and Viser processes to run the updated code.

## OracleGS

Step 4 also offers OracleGS with SEVA two-pass trajectory generation and the authors' VGGT-attention-guided reconstruction. See [OracleGS integration](oraclegs.md) for the separate dataset-adapter choices, 576p default, 720p experiment with automatic diffusion stride padding, runtime setup, and validation limits.

### Per-view SEVA preset experiment

In OracleGS settings, select **Dataset integration → Per-view SEVA preset**,
then choose **Orbit**, **Spiral**, **Lemniscate**, or **Shuffle orbit**. The existing
pairwise and saved-pose modes remain available. Each selected view starts a
separate SEVA `img2trajvid` job conditioned on that view's RGB alone. With six
views and 80 target frames this produces six disconnected clips and 480 synthetic
images. All saved references remain GT for reconstruction; targets, including
those coincident with a reference pose, remain synthetic.

The paths follow Stability AI's Basic-demo presets at pinned revision
`fe19948e9b7bea261ab2db780a59656131404a83`:

- Orbit: full 360° clockwise orbit around local `[0, 0, 10]`, radius 10.
- Spiral: Gradio radii `[1, 1, 0.5]`, two turns, `zrate=0.5`, bounds `[1, 5]`,
  rebased so its first pose is exactly the input camera.
- Lemniscate: the author's 60° figure-eight construction, focus distance 10.
- Shuffle orbit: balanced counts of those three presets, shuffled using the run
  seed (23 by default). Six views get exactly two of each. For other counts the
  preset counts differ by at most one. Assignments are recorded per segment.

The size control specifies **scene units per demo unit**, default 1. All presets
share it. It scales world-space translation, preserving rotations, while every
clip retains its saved camera's calibrated intrinsics and orientation. Scene
units are not necessarily metres. SEVA subsequently normalizes camera positions;
this control is not a guarantee of proportional visible motion. Orbit can sweep
far outside the observed scene. No collision avoidance or scene-depth adaptation
is applied, so this is a reproducible diagnostic, not a guaranteed valid path.

Sampling includes the initial camera and excludes the repeated loop endpoint.
Generation uses camera scale 2, CFG `[4, 2]` for Orbit and `[3, 2]` for Spiral and
Lemniscate, as in the Basic-demo preset settings. Geometry guidance, when enabled,
uses the clip's own reference. Reconstruction depth and initialization settings
remain independent. Saved metadata records the source, scale, seed, assignment,
poses and reference ownership; results label each independent clip by preset.

The source of the trajectory formulas is
[SEVA geometry.py](https://github.com/Stability-AI/stable-virtual-camera/blob/fe19948e9b7bea261ab2db780a59656131404a83/seva/geometry.py),
with Gradio defaults from
[demo_gr.py](https://github.com/Stability-AI/stable-virtual-camera/blob/fe19948e9b7bea261ab2db780a59656131404a83/demo_gr.py).
The authors recommend orbit/spiral/lemniscate to show scene depth in their
[usage guide](https://github.com/Stability-AI/stable-virtual-camera/blob/fe19948e9b7bea261ab2db780a59656131404a83/docs/GR_USAGE.md).
We preserve saved calibration instead of the demo's assumed 54° field of view;
this integration is not a claim to reproduce a particular highlight video.

### WorldExplorer source scenes

Exported WorldExplorer runs with saved camera metadata appear automatically in
scene selection as `WorldExplorer · <run id> · <export timestamp>`. The base PLY
and every image in that run's `img2trajvid/transforms.json` stay linked through
the checkpoint's source scene path. Stage 4 stages a checksummed copy of these
original RGBs and maps their OpenGL poses through the exact export's Nerfstudio
`dataparser_transforms.json` transform and scale.

ArtiFixer and OracleGS add these images as original RGB supervision alongside
new generated views and the existing GPT-repaired references, for both source
splat and fresh initialization. Original images retain their resolution and
intrinsics. They do not enter GPT editing or diffusion conditioning. OracleGS places these originals in the same
synthetic sampling pool as its new SEVA outputs, leaving only selected repaired
references in the GT pool. WorldExplorer images receive VGGT attention by default; the dashboard also
offers the previous unweighted synthetic RGB loss (L1, SSIM, LPIPS). Global
attention shuffling mixes WorldExplorer and new SEVA targets. New SEVA outputs
always require their attention maps. Depth supervision remains attached to the selected
reference views. Missing images or normalization metadata stop staging instead
of silently omitting source views. G4Splat does not use this integration.

### Reduce OracleGS run storage

All inactive OracleGS runs (completed, stopped, errored or GPU-expired) offer **Reduce** and **Backup & reduce** on the results
page. Reduction retains the repaired reference images, full `samples-rgb.mp4`
trajectory videos, original/source and final repaired PLY, logs and essential
lifecycle/provenance metadata. It removes input/generated image datasets, pose
exports, sparse videos, attention visualizations and intermediate model checkpoints.
The retained run remains viewable but cannot be used to resume training.

Backup & reduce first copies the entire run to `outputs/run_history/<run_id>` and
verifies file contents with SHA-256 before modifying the active copy. Existing
backups are never overwritten. The backup requires additional local space and can
be moved manually to an external drive. Reduction refreshes the saved size and
marks the result Reduced. Shared source checkpoints outside the run are untouched.

Reduction preserves available artifacts on incomplete and older runs even when no
final or original PLY was produced. A forced disk-size rescan runs after cleanup,
and the returned size immediately updates every dashboard card for that run.

### Reduce ArtiFixer run storage

ArtiFixer result cards also offer Reduce and Backup & reduce for inactive runs.
Each saved inference or plus-pass prediction batch becomes a separate H.264 MP4
at 12 fps, in numeric filename order (CRF 18, browser-compatible YUV420, odd image
sizes padded to even dimensions). The reduction record saves the source filename
order. Every video must pass a decoded frame-count check and a complete FFmpeg
decode before any source images are removed. A failed encode leaves all images
and checkpoints intact for retrying.

Reduction keeps original and final repaired PLYs (including native final PLYs),
logs, inference videos, and essential lifecycle/result/evaluation metadata. It
removes degraded RGBs, prediction images, reference images, pose datasets,
intermediate training checkpoints. Stage-comparison images and the trajectory-quality
chart (with their metadata) are retained for compact visual debugging. Run details expose playable
videos and downloads. Videos are lossy visual-debugging previews, not substitutes
for original images in training or pixel-level evaluation. Backup & reduce copies
and verifies the full run before encoding or cleanup; external shared source
checkpoints are never deleted.

Use **Save inference videos** in the ArtiFixer run-details header to export the
same verified MP4s without reducing the run. It is available for inactive,
unreduced runs and processes inference/plus batches across the whole run. Source
images, PLYs, checkpoints, logs and previews remain intact. Videos appear in run
details after export, and the stored run size is recalculated. Repeating the action
rebuilds the same video files rather than adding duplicates.

Interrupted ArtiFixer runs retain their partial PLY at its recorded path, the
partial-result manifest, stop-state diagnostics and failure/lifecycle records.
When the registered original is a SOG bundle or directory, reduction exports the
unfiltered LOD-0 review scene as `original.ply` before removing any inputs. If
that export fails, all source images remain. Video-only export does not require
copying or converting the original scene.


### Step 4 ArtiFixer reconstruction controls

**Reconstruction images** and **Reconstruction initialization** are independent:

- **All images reconstruction** (`reconstruction_images=all`) preserves
  the existing behavior: saved references, ArtiFixer-generated views, and all
  archived WorldExplorer images supervise reconstruction.
- **ArtiFixer3D images only** (`reconstruction_images=distilled`, default) keeps the saved
  references and generated views, without staging, uploading, or adding the
  WorldExplorer source-image archive. Repaired anchors remain direct targets;
  this does not discard the references used by the authors' distillation recipe.
- **Original PLY snapshot** (`reconstruction_initialization=full_splat`)
  imports all source Gaussian parameters, including SH, with a fresh optimizer
  and training schedule.
- **Fresh reconstruction** (`reconstruction_initialization=point_cloud`, default)
  creates new Gaussians using the existing point-cloud initialization path.
  By default the seed points are source-splat positions and RGB; no original
  Gaussian scales, rotations, opacity, or SH are imported. This is not a literal
  zero-point initialization, nor new SfM from the generated images.

Both initializations produce separate output files. Source-splat RGB/opacity
still condition diffusion in every combination. The 30,000-step author fitting
recipe and selected regularization profile remain unchanged. Run requests,
worker runtime, supervision manifests, and completed distillation results record
these choices. New runs default to repaired references plus generated views and
fresh point-cloud initialization; explicitly saved choices remain supported.

These custom controls apply to **Legacy Viser / gsplat**. CLI equivalents:
`--artifixer-pipeline legacy_gsplat --reconstruction-images all|distilled` and
`--reconstruction-initialization full_splat|point_cloud`.

### Default authors rendering and reconstruction

Step 4's **ArtiFixer rendering and reconstruction** setting labels the default
**Authors 3DGUT baseline** and selects
`authors_3dgut`, `authors_3dgut_viser_rgb`, or `legacy_gsplat` (default). It is saved with the job, propagated
to the GPU worker, and recorded in run provenance. CLI: `--artifixer-pipeline
authors_3dgut|authors_3dgut_viser_rgb|legacy_gsplat`; config: `splatfix.runtime.artifixer_pipeline`.

Authors mode imports the original Gaussian PLY with all SH coefficients into the
pinned 3DGUT implementation. It uses the authors' rendering configuration,
native `pred_rgb`/`pred_opacity`, RGB PNG rounding and 8-bit opacity export/read
convention. Both ordinary target channels come from the same renderer. The
existing inserted starting reference policy still replaces that frame's opacity
with one. Native ray-distance accumulation is converted to expected camera-Z
only for the application's auxiliary scale estimation; opacity is not changed
by that conversion.

Reconstruction uses the upstream fresh 30,000-step sparse MCMC/LPIPS
ArtiFixer3D recipe with its original regularization settings, references and
generated views. Native mode fixes initialization to `point_cloud` and images to
`distilled`; custom initialization, extra images and regularization remain in
legacy mode. The two generated-image loss controls above remain available in
native modes as explicit recipe overrides. The “+” pass retains the authors' paired reconstruction renders.
Renderer-specific cache keys prevent reuse of legacy RGB/opacity as native
inputs. Errors never silently fall back to gsplat.

This switches the rendering and fitting implementation, not the application's
source-data protocol: selected/GPT-edited references, camera selection, prompt
options and source-splat point initialization remain application inputs. It does
not rerun the paper's initial sparse photographic reconstruction. Native source
rendering currently requires binary Gaussian PLY with DC-only or degree-3 SH;
other formats can explicitly select legacy mode. Existing saved worker requests
without a pipeline field retain their legacy renderer validation.

The third option, **Viser RGB + authors 3DGUT geometry / fitting**
(`authors_3dgut_viser_rgb`), captures conditioning RGB with Viser while retaining
the native 3DGUT opacity and auxiliary depth, including opacity quantization and
camera-Z conversion described above. It uses the same fixed fresh ArtiFixer3D
fitting recipe as the authors 3DGUT option. The “+” pass likewise uses Viser RGB with native
reconstruction opacity. Its trajectory cache is separate from both other modes;
RGB-only captures can share the Viser capture cache. This mixed-renderer variant
is an application option, not the authors' original paired-rendering method.
The default is `legacy_gsplat`; selecting native geometry still requires a supported PLY.

### Endpoint-only ArtiFixer checkpoints

In Step 1 choose **Endpoint repair · first and last only (2+ views)**. Save at least two
positions in the intended travel order. The horizontal FOV controls now include
**77°**, including Custom capture. Original RGB and calibrated cameras remain
saved for every position. Step 3 edits only the first and last positions, resuming missing
endpoint edits without repeating successful requests.

These checkpoints require edited ArtiFixer reconstruction and automatically use
**No split**. Consecutive saved poses use the authors' interpolation helper;
legs are joined in selection order into a single continuous trajectory. Intermediate
saved RGBs never become reference images. Degraded source renders provide the
inference inputs along the complete interpolated path.

**Endpoint checkpoint references** selects either both repaired endpoints
(starting endpoint first) or only the starting repaired endpoint. Initial inference
runs twice, with fresh inference caches: front to back and the exact reverse.
`inference-directions.json` records frame indices, reference order and output
ownership. Forward predictions retain the normal `inference/` location and enter
reconstruction with the two repaired references. Reverse predictions are saved
under `inference-back-to-front/` and never enter reconstruction. The existing
post-reconstruction 3D+ pass remains forward-only.

CLI selection accepts `--repair-policy endpoints --views 6 --horizontal-fov 77`;
repair accepts `--endpoint-reference-mode both` (default) or `initial`.
`--split-mode no-split` is also available independently; endpoint checkpoints
force this mode regardless of a previously selected split setting.

Existing checkpoints with at least two views can use the same workflow: in Step 4 select
**Image trajectory → Endpoint image trajectory**. This run-only override uses
just the first and last repaired images even when all views were repaired. It
preserves the source checkpoint, its images, and its saved camera calibration;
it does not require recapture or new GPT edits. The run automatically uses
No split and supports both endpoint reference modes. CLI equivalent:
`--image-trajectory endpoints`.

For normal ArtiFixer runs, leave **Image trajectory → Use checkpoint type** and
choose **ArtiFixer trajectory splitting → No split** in Settings. All six saved
reference images and their poses remain conditioning and reconstruction
supervision. The usual authored path is processed as one continuous sequence,
with no per-leg cache resets and no reverse inference run. CLI equivalent:
`--split-mode no-split` without the endpoint trajectory override.

Endpoint checkpoints accept any integer view count of two or more, without the
normal selection's 24-view limit. Custom selection may finish once at least two
positions have been saved. Only the first and last selected views receive GPT
repairs; all intermediate cameras remain trajectory waypoints and never become
reference or repaired-image supervision. This also applies to the run-local
endpoint override on prior checkpoints.

### ArtiFixer input renderer comparison

The experimental `splat_explorer.splatfix.artifixer.renderer_comparison` command
compares an existing prepared run with the authors' pinned 3DGUT renderer without
training a new scene or changing the production renderer. Run it on the GPU host
where the saved request's paths and authors environment are available:

```bash
python -m splat_explorer.splatfix.artifixer.renderer_comparison \
  --request /absolute/path/to/run/request.json \
  --source-ply /absolute/path/to/original-source.ply \
  --output /absolute/path/to/new-renderer-comparison \
  --infer
```

The source PLY must match the checkpoint fingerprint. The command requires the
existing caption, references, trajectory and source points; it does not relocate
downloaded requests automatically. `--infer` runs the same seeded ArtiFixer
inference for each arm; omit it for rendering and input measurements only.
The GPU environment needs `gsplat` as well as the pinned 3DGRUT dependencies.

The four arms are `current` (saved Viser RGB/gsplat alpha), `gsplat` (paired RGB
and alpha), `3dgut` (paired RGB and alpha from the authors' renderer), and
`opacity_only` (saved RGB with 3DGUT alpha, intentionally mixed for diagnosis).
Select a subset with `--arms`. The paired arms share the full raw PLY parameters,
full SH appearance, black background, exact camera calibrations and the authors'
8-bit opacity export/read convention. Raw float alpha is also retained. References,
camera scale, prompt, seed and sequence/starter policy remain fixed. Starting
reference opacity is still overridden by the existing inference worker.

The output contains isolated requests, render pairs, `experiment.json`,
`input-differences.json`, and, with inference and the current arm enabled,
`output-differences.json`. MAE and PSNR here measure disagreement, not quality:
use held-out real images and multi-view/temporal inspection to judge improvement.
The current arm can additionally differ in SH, filtering and background. Compare
the two matched arms to isolate rendering more closely. This imports the existing
source PLY; it does not reproduce the authors' initial sparse reconstruction.

This adapter has local isolation/calibration tests but still requires a CUDA
smoke test and empirical output comparison before production adoption.

### Initial reference seeding

New ArtiFixer repair runs default to **Image cache insertion → On · seed initial
image**. Each series starts with its selected reference RGB and opacity 1,
including forward and reverse endpoint runs and the forward “+” pass. This
matches the earlier double-split input-frame seeding; it does not enable the
stronger exact-latent cache adaptation. **On · seed and preserve initial image in
cache** additionally enables that adaptation. **Off · reference conditioning
only** explicitly disables seeding, including for double-split runs, but retains
reference conditioning and direct reconstruction supervision.

`initial_image_seeding` records this choice separately from `image_cache_insertion`
(the exact cache-preservation extension). CLI: `--initial-image-seeding` /
`--no-initial-image-seeding`. Historical worker requests without the new field
retain their per-segment behavior. Per-series `inference-seeding.json` and
`plus-seeding.json` record effective seeding and cache preservation. Existing
queued jobs are not rewritten.

Historical compatibility check (2026-10-22): the double-split planner and
`scene_runs_ext/starter_inference.py` are unchanged from pre-chat commit
`564e8cc`; the starter sampler also matches `3204718`, preceding
`run_20261021_204353`. The strongest option retains the original standalone
clean starter latent, timestep-zero starter KV insertion, seven-latent generated
blocks, clean timestep-zero refresh after each generated block, one-frame
attention sink, 21-latent local attention window, and cache reset per series.
Double-split's explicit `seed_index` alone never enabled this sampler: the
historical dispatcher installed it only when `image_cache_insertion=true`.
The new seeding flag cannot disable seeding when cache preservation is enabled.

### Repeat a saved ArtiFixer reconstruction

On an ArtiFixer result detail page, **Queue next ArtiFixer pass** appears above
OracleGS reconstruction. It snapshots the saved output PLY as the next
checkpoint's `original.ply`, the calibrated reference images, and the recorded
orbit and effective runtime. The job uses the normal manager/GPU queue. Repeat
inputs are independent of the parent run; second, third and later passes record
the parent result and pass number.

Repeat workers retain the previous camera scale and exact orbit instead of
running alignment/interpolation again. They recapture anchor and trajectory RGB
with the recorded renderer (Viser for legacy/hybrid pipelines), recompute opacity,
depth and seed points, and run captioning, inference, distillation and plus again.
Initial-image seeding and cache insertion retain their recorded settings. Saved
GPT references remain the same, as in an ordinary reconstruction from that
checkpoint. Old image, geometry and caption caches are not copied.

The action accepts completed outputs and saved intermediate PLYs from interrupted
runs; repeat provenance records whether the parent was incomplete and which PLY
was used. It requires a saved result PLY, its prepared checkpoint, effective
request settings and saved authored orbit. Missing/reduced inputs disable the
button with a reason rather than silently using current defaults or new cameras.
Bicycle benchmarks retain their existing benchmark repeat workflow. No GPU job
is launched until the action is clicked. Restart the web/manager process after
updating Python code.

### ArtiFixer Gaussian limit

Step 04 exposes **Gaussian limit**, initially **1,000,000**. Click the value to
enter a positive integer, then press Enter or click away to save. This applies
to ArtiFixer checkpoint reconstructions in all three pipeline modes. The queued
options, effective runtime and reconstruction recipe record `max_gaussians`;
repeat passes inherit the recorded value. Historical runs default to one million.

The value overrides the authors' `strategy.add.max_n_gaussians` densification
cap. It does not prune an imported source PLY that already exceeds the cap.

Reduced ArtiFixer runs can repeat from the original registered checkpoint when
reduction removed the prepared checkpoint. Exact retained reference images take
precedence over subsequent edits to that checkpoint. Reference hashes and saved
anchor poses/calibration are checked before inference. A retained trajectory or
full-orbit export is reused first. If those poses are missing, the worker rebuilds
the orbit using the pinned authors implementation, saved camera scale, endpoint
policy, split mode and interpolation settings; it checks the regenerated segment
and reference mappings against the parent provenance. Missing checkpoint inputs
or incompatible trajectory provenance remain explicit errors.

Reduction now retains the small anchor/full-orbit/provenance JSON exports even
when reconstruction images are not retained, to support future repeats.

Repeat snapshots recover the WorldExplorer prompt before replacing the scene
path: parent request, prepared checkpoint, original checkpoint/scene, then the
registered original scene. The recovered prompt and its source are embedded in
the repeat checkpoint. If WorldExplorer captioning is selected and no prompt is
recoverable, repeat validation rejects the request before it enters the GPU queue.

### Restart failed or cancelled queue jobs

Failed, cancelled and GPU-expired Splatfix jobs have a **Restart** button in the
main queue. It creates a new run using the stored effective configuration and
existing inputs, retaining the old run, logs and settings snapshot. The old
schedule is cleared so the new job is immediately eligible for the normal queue;
old STOP flags, GPU allocation details and worker execution state are not copied.
This starts the stage again, rather than resuming optimizer state. Repeated clicks
reuse an already queued/active restart of that job. Missing saved checkpoints or
input snapshots produce an error. The new settings snapshot retains the original
dashboard controls/submission and records `restart_from` for traceability.
