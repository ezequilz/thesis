# OracleGS in Splatfix Step 4

OracleGS is an independent reconstruction method under `src/splat_explorer/splatfix/oraclegs/`. Select **OracleGS** and **GPT-image improved** in Step 4. The repaired RGBs are the ground-truth/reference dataset at their saved, calibrated camera poses. They are not replaced by SEVA outputs. Baseline input mode remains available for a comparison using original RGBs.

## Pipeline

1. Stage saved references, validate OpenCV camera-to-world rotations/calibration, preserve the source files, and downscale RGBs with Lanczos. Center-crop by default (`seva_fit: crop`), uniformly resizing to cover the output and updating focal lengths and principal points by the same image-space affine. Non-16:9 checkpoints are supported. Repaired-image aspect ratio must still match its saved camera: a previously unknown crop cannot be inferred safely. Padding (`seva_fit: pad`) is an explicit fallback to preserve framing. Inputs that require upscaling are rejected. Saved originals remain unchanged.
2. Use the calibrated reference cameras with COLMAP feature extraction, exhaustive matching and fixed-pose triangulation. Match the COLMAP database IDs explicitly; permit two-view tracks for sparse inputs. Before depth or SEVA inference, verify that SfM retained every staged reference identity, pose and calibration. Save pixel-reprojection checks in `camera-alignment.json`; reject drift rather than continuing with misaligned supervision. No source-splat geometry initializes this reconstruction. Estimate reference pseudo inverse depth with Depth Anything V2 Large, then run the authors' `utils/make_depth_scale.py` to align depth to SfM. Empty triangulation or entirely unreliable depth fails the run.
3. Stage SEVA `transforms.json` in OpenGL camera-to-world coordinates (`c2w_cv @ diag(1,-1,-1,1)`) and `train_test_split_N.json`. Split the selected path into adjacent reference pairs **before** interpolation. Each pair uses SEVA GUI natural Kochanek–Bartels interpolation with only its two endpoints, producing 80 strictly interior targets by default. Six references yield five independent paths and 400 synthetic targets. Every pair receives all six repaired reference images and cameras, regardless of path order.
4. Run each pair sequentially in a separate SEVA process using ordered `img2vid`, `interp-gt` with the ordered-pair chunk adapter, `gt-nearest` first-pass conditioning, and CFG `[3,2]`. Each invocation performs SEVA's two passes: sparse generated anchors, then dense target synthesis. Generated anchors do not carry across pairs. Validate every exported pose and calibration before merging targets into the canonical reconstruction dataset; retain original repaired references for reconstruction. Per-pair videos, requests, logs, and camera/conditioning audits remain available. The `saved_poses` diagnostic retains its single invocation.
5. Run the pinned authors' VGGT attention operations through a reference-grounded batching adapter. Each chunk includes every saved reference and shuffled targets from one sub-trajectory by default (`oracle_attention_batching: within_trajectory`). The alternative `global` mode shuffles targets across all trajectories. Both are available in the reconstruction UI and CLI (`--oracle-attention-batching`). For 80 interior targets, steps 1–40 use the starting reference and 41–80 use the ending reference; midpoint ties use the start. For each context chunk, each required endpoint reference is placed first for a separate query; only maps for targets assigned to that source are published. Chunks are capped at 96 total images, including references. Shuffling is deterministic under the run seed. With six references and 80 targets per segment, the default has five 86-image contexts and ten source queries. Layers 0/22, weights 1/4 and 3/4, per-layer normalization, padded preprocessing, raw map export and training unpadding remain upstream operations. `attention-chunks.json` records source identities and target membership; per-chunk diagnostics retain repeated reference maps separately. This is a disclosed method variant, not the released random batching policy.
6. Run the original OracleGS optimizer and its pinned inverse-depth CUDA rasterizer. Ground-truth views use photometric supervision plus reliable SfM-aligned inverse-depth supervision. Synthetic views retain `is_synthetic=True` and use attention-weighted L1, weighted SSIM and globally weighted LPIPS. The original MCMC Gaussian updates, regularizers and GT/synthetic sampling code are unchanged. Defaults: 22,000 iterations, LPIPS 0.3, synthetic SSIM 0.3, scale/opacity regularization 0.01, noise LR 500000, `cap_max=1800000`, `init_type=sfm`. Inputs are already resized, so training uses `-r 1`.
7. Save native 3D Gaussian PLY, recipe, command requests/logs, depth products, sparse model, SEVA outputs and attention maps. The queue downloads results and exposes the PLY through the existing results UI. Cancellation terminates the active subprocess group; completed intermediate products are retained. A stopped run is not advertised as a completed reconstruction.

## Separate adapter options

For WorldExplorer source scenes, original generated images join the SEVA
synthetic sampling pool. Only saved repair references remain GT. The dashboard's
**WorldExplorer image attention** defaults to **Include in VGGT attention**
(`oracle_worldexplorer_attention: weighted`); **Unweighted synthetic supervision**
(`unweighted`) retains the previous behavior. CLI: `--oracle-worldexplorer-attention`.

Weighted mode stages both sources before VGGT extraction. **Shuffle across all
generated images (SEVA + WorldExplorer)** (`oracle_attention_batching: global`)
shuffles their union deterministically using the run seed. Each context contains
all repaired references and at most 96 total images. Every target receives one
published attention map. SEVA query assignment stays unchanged; WorldExplorer
uses the nearest repaired camera center in normalized scene coordinates, breaking
ties by viewing direction and then reference index. This is a geometric heuristic,
not an overlap or quality guarantee. With within-trajectory grouping, SEVA keeps
its trajectory groups and WorldExplorer is grouped by assigned repaired reference.

WorldExplorer images retain their calibrated cameras and resolution and receive
no depth loss. Weighted mode requires their VGGT maps for the original synthetic
losses; unweighted mode uses the upstream unweighted synthetic loss. New SEVA
outputs always require attention maps. Dataset folder counts continue to reflect
the repaired GT count. The mode and target/source assignments are recorded in the
recipe, WorldExplorer supervision report, and attention plan. Use a fresh run
with the updated dashboard/worker code; archived results are not recomputed.

`oracle_adapter: trajectory` is the requested six-reference-plus-intermediates configuration. The official COLMAP reader expects a dataset RGB at every target pose and would skip a target without one. Dataset assembly supplies generated RGBs at intermediate target entries, then a narrowly scoped reader adapter removes these entries from the **GT camera list before training selection**. The original synthetic camera objects and all loss/attention code remain untouched. At saved poses, repaired reference RGBs stay in `images/`, and generated RGBs go in `synthetic_N/`. The adapter never promotes generated RGBs to GT.

`oracle_adapter: saved_poses` generates only at the saved path poses. All target poses have real repaired reference images, so the original dataset loader runs without the reader adapter. This is an unmodified-loader diagnostic option, not a reproduction of the paper's benchmark splits or a substitute for the requested full trajectory.

The two options are shown separately in Step 4 and recorded as `checkpoint-trajectory-adapter` or `saved-poses-original-loader` in results. The default is `trajectory`.

## Resolution

- **576p (default):** 1024 × 576, calibrated center crop, `L_short=576`. Both dimensions satisfy SEVA's 64-pixel spatial stride. For 16:9 inputs this is simply downscaling; other aspect ratios are cropped with matching intrinsics.
- **720p experiment:** 1280 × 720 references and reconstruction for both fitting modes. SEVA internally receives 1280 × 768 with 24 pixels above/below and `cy + 24`; generated outputs are cropped back to 720 with the original calibration. This preserves 16:9 framing without stretching.
- **Image fitting:** `crop` and `pad` control how non-16:9 sources fit the requested frame. Stride padding is independent of that choice and automatic for 720p.

The 720-pixel dimension is incompatible with the released SEVA U-Net's 64-pixel stride. Native 1280 × 720 diffusion is not claimed. `inputs.json` distinguishes the requested preset, actual reconstruction resolution, internal SEVA resolution, fitting mode, and per-reference image affine/intrinsics. SEVA output pose and calibration are checked before images are admitted to reconstruction.

## Provenance and differences from paper benchmarks

`upstream/` contains byte-for-byte snapshots of the implementation source and licenses from OracleGS, SEVA, the authors' VGGT attention fork, and Depth Anything V2. The original CUDA rasterizer, simple-knn, and required GLM headers are also included. Large example media, publication figures, Git internals and optional SIBR viewers are excluded from the source snapshot. `upstream-lock.json` records repository URLs, exact revisions, every included file's SHA-256, and submodule gitlinks. Provisioning fetches the complete pinned repositories and required OracleGS CUDA submodules. Runtime validation checks the source hashes and submodule revisions.

The paper's prose gives a Beta-distribution synthetic sampling formula. The released `train.py` instead computes `prob_gt = 0.9 - progress * (1-progress)**3` when `--gt_synth_schedule` is enabled. This integration deliberately preserves the released implementation; it does not silently change the schedule to the paper formula.

The OracleGS README's generation example uses `img2img`, CFG 3 and `T=80`. Our pairwise trajectory setup uses SEVA's `img2vid` / `interp-gt` recipe with an explicit ordered-pair chunk adapter, CFG `[3,2]` and adaptive context length. The number of target images is distinct from the SEVA context window `T`. Consequently this is an original-optimizer Splatfix baseline with an explicitly recorded input/generation configuration, not a claim to reproduce the paper's benchmark metrics. The Gaussian cap is configurable because the paper derives it per scene from another baseline. No held-out GT exists along the custom trajectory; generated images must never be reported as held-out ground truth for PSNR/SSIM/LPIPS evaluation.

Sources: [OracleGS authors' implementation](https://github.com/atakan-topaloglu/OracleGS), [VGGT attention extractor](https://github.com/atakan-topaloglu/vggt_aten), [SEVA CLI documentation](https://github.com/Stability-AI/stable-virtual-camera/blob/main/docs/CLI_USAGE.md), [COLMAP known-pose reconstruction](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

## Runtime and use

The method uses separate Linux NVIDIA environments for the original OracleGS trainer and the newer SEVA/VGGT/depth models. Existing ArtiFixer/G4Splat environments are not reused. `auto_setup: true` provisions missing runtimes on the CUDA worker; incompatible revisions or modified sources fail rather than being overwritten. Initial installation and model downloads require network access, disk space, CUDA build support and any required Hugging Face model access. For explicit setup on the worker:

```sh
bash src/splat_explorer/splatfix/oraclegs/provision.sh /workspace
```

Settings live in `splatfix.oraclegs_runtime`. Defaults use `/workspace/third_party/OracleGS`, `/workspace/third_party/oraclegs-{seva,vggt,depth}`, `/workspace/oraclegs-env/bin/python`, `/workspace/oraclegs-model-env/bin/python`, and `/workspace/models/oraclegs/depth_anything_v2_vitl.pth`. Set `auto_setup: false` for a manually installed custom layout. All runtime paths can be overridden there.

The trainer's compiler is installed separately in `/workspace/oraclegs-cuda-11.7`
from NVIDIA's pinned CUDA 11.7.1 packages. Setup verifies `nvcc --version`, CUDA
headers and the runtime library before compiling the original extensions, even
when the Python environment already exists. This avoids the `cudatoolkit-dev`
runfile post-install hook, which failed in the LRZ container while environment
creation still reported success. Incomplete compiler installations are repaired
on retry; the system CUDA and other methods' environments are left unchanged.
The setup also discards NGC's inherited CUDA architecture list and isolates the
Python library paths. SEVA uses Diffusers 0.32.2 and Hugging Face Hub 0.28.1 with
PyTorch 2.5.1: newer unpinned Diffusers releases require unavailable PyTorch APIs.
Setup checks the SEVA module imports as well as both trainer CUDA extensions. Every run repeats isolated import/version probes before SfM; existing Python executables alone no longer allow an incomplete installation to bypass automatic repair.


SEVA's pinned loader requests the unavailable `stabilityai/stable-diffusion-2-1-base`
VAE repository (observed HTTP 401; also reported in
[SEVA issue 92](https://github.com/Stability-AI/stable-virtual-camera/issues/92)).
The wrapper redirects only that VAE lookup to a local, checksum-verified snapshot
of `sd2-community/stable-diffusion-2-1-base`, revision
`4e63672c03103b6c636b8fb4119ba982469b2955`. Both configuration and safetensors
hashes are pinned in `model_assets.py` and recorded in `vae-provenance.json`.
The weights match the SD 2.1 public clones and the blob published by Stability AI
in `sd-vae-ft-mse`; direct comparison with the unavailable original endpoint is
not possible. This is a disclosed model-source recovery, with the upstream
SEVA codec, sampling, and network code unchanged. No arbitrary VAE replacement
or silent fallback is permitted. SEVA and CLIP weights still use their upstream
sources and access requirements. The `model_access` stage checks the official
SEVA configuration before SfM or large CLIP downloads. A 401/403 produces a concise
actionable error; accept the model's access conditions and run
`/workspace/oraclegs-model-env/bin/huggingface-cli login` inside the worker
container as the same user. Credentials belong in the Hugging Face credential
store or protected environment, never in job JSON.

For users without a worker shell, run the local helper from the repository:

```sh
.venv/bin/python scripts/lrz/setup_oraclegs_access.py
```

Connect to LRZ through Splatfix first and accept the official model access
conditions. The helper reads a token with a hidden prompt, sends it only through
SSH stdin, authenticates the same worker account, and caches `config.yaml` and
`modelv1.1.safetensors`. The optional `--download-local` also copies both files
back to the local machine; omit it to avoid the additional multi-GB transfer.
The helper and every OracleGS model subprocess explicitly share the LRZ
account's `~/.cache/huggingface` (resolved from the account database, not `HOME`).
This includes `HF_HOME`, both Hub-cache variables, and `HF_TOKEN_PATH`.
Splatfix redirects `HOME` and `XDG_CACHE_HOME` for its general container runtime;
those redirects must not hide the login or the weights installed by this helper.
By default no model files are copied to the local machine. Use `--check-only`
to verify the worker cache without entering a token again, or `--copy-only
--download-local DIRECTORY` to retry an explicitly requested local copy. The
copy command supports Apple's bundled rsync 2.6.9.
No token is placed in shell arguments, job JSON, or the repository. The helper
does not start reconstruction. A live configured GPU allocation is required for
the existing container transport.

Remote validation on 2026-10-16 used six repaired references at **1024 × 576**.
Both CUDA extensions and SEVA modules imported on the allocated A100.
Run `run_20261016_180646` passed known-pose SfM, camera-alignment and depth gates
(51 triangulated points; two positive depth scales out of six), then exposed the
retired VAE dependency. Its replacement snapshot was downloaded and both hashes
verified on the worker. The subsequent retry `run_20261016_180953` was stopped
after a separate check confirmed HTTP 401 for the gated official SEVA model.
Generation, attention extraction and final training have **not** been validated
end to end; authorized model access is required to finish that validation.


Local CUDA CLI example:

```sh
splatfix --config configs/splatfix.yaml repair \
  /path/to/saved/checkpoint \
  --output /path/to/fresh/run \
  --reconstruction-method oraclegs --mode edited \
  --seva-resolution 576p --oracle-adapter trajectory --oracle-target-frames 80
```

The dashboard uses the existing queue, GPU lease, LRZ transport, status/deadline handling and result download flow. The selected per-job generation and adapter options are frozen when queued. High-resolution source reference files are preserved, and no new GPT-image request or source recapture occurs in Step 4.

## Validation boundary

CPU tests cover camera projection and conventions, exact saved-pose inclusion, repaired-image preservation, separate GT/synthetic membership using the actual upstream COLMAP reader, database camera IDs, non-widescreen crop/pad projection, SfM pose-drift rejection, 720p padding/cropping, output identity and calibration checks, source hashes, option validation, worker cancellation, and full stage dispatch with model subprocess test doubles. These tests do not establish CUDA kernel compatibility, successful model downloads, generated-image quality, or end-to-end GPU training. A CUDA smoke run and a full six-reference reconstruction remain necessary before claiming a validated reproduction.

### Live research diagnostics and interruption

OracleGS run details now read locally cached artifacts: calibrated ground-truth
references and their image affine/intrinsics, camera-alignment and SfM/depth gates,
SEVA sparse first-pass anchors and full trajectory, a sampled generation contact
sheet, exact-filename synthetic/attention pairs, source revisions, commands and
logs. Missing reports are **unverified**, not successful checks. Alignment validates
camera encoding consistency; it does not establish scene geometry accuracy.
First-pass filenames are export indices, not an inferred target-camera mapping.

The queue mirrors research outputs at most once per minute during generation and
attention, then once on entry to reconstruction. It excludes model checkpoints and
combined-image duplicates. Browser refresh only reads the local cache. Cache status,
phase and last-transfer time are shown. Logs refresh at most once per minute;
worker status checks are spaced ten seconds apart, with local cancellation checks
between them. Final transfer and compute-lease handling retain the existing path.
For jobs launched before this change, a bounded read-only observer is available:

```sh
python -m splat_explorer.splatfix.oraclegs.preview_cache \
  --run-dir /local/run --remote-dir /remote/workspace/splatfix-jobs/run \
  --max-seconds 43200
```

It reads existing local worker status, skips setup, exits on terminal status, and
shares a file lock and freshness record with the queue to avoid duplicate transfers.
It does not launch, cancel or resume GPU work. Existing workers require updated
worker code on a subsequent launch for the snapshot behavior below.

The original trainer supports intermediate Gaussian PLY snapshots. Runs now retain
its 3000/7000/15000 save points (when below the configured final iteration), plus the
final iteration. A wrapper records a snapshot only after upstream `Scene.save`
returns successfully. On interruption or failure, the latest confirmed snapshot
is copied to `oraclegs-partial.ply` and labeled incomplete in `partial-result.json`,
with its actual saved iteration. A newer interrupted write is never selected.
This is recovery of the **latest saved iteration**, not serialization at the exact
stop instant or a resumable optimizer checkpoint. Before the first save, only
available generation/geometry/attention artifacts can be downloaded. No upstream
source, losses, optimizer or training schedule is modified by the stop hook. The reference-grounded attention batching adapter is described above.


## Camera-path debugging, 2026-10-16

The inspected run `run_20261016_183036` contains six reference cameras and 80
synthetic targets. Its reference centers span approximately
`[12.283559, 1.122240, 4.441781]` in source-scene coordinates. The generated
`transforms.json` retains translation. The OpenCV → OpenGL export and SEVA's
OpenGL → OpenCV parser conversion preserve camera centers; COLMAP inversion
also retains them. This evidence does **not** establish that generated image
content obeyed those cameras. Output metadata alone cannot prove that.

Two integration issues were corrected:

- `oracle_target_frames` previously allocated one total budget over all pairs.
  It now means images per pair. Its default 80 yields 400 targets for six views.
  `inputs.json` records versioned count semantics, segment endpoints, sample
  indices, reference identities and actual target positions. Existing staged
  runs keep their original trajectories and require a fresh run to use this fix.
- The previous custom path used straight-line position interpolation plus
  independent SLERP. It now uses SEVA's GUI quaternion and position splines,
  with exact saved endpoints, natural ends and no implicit loop. The pinned
  upstream model and optimizer files remain unchanged. This is the GUI path
  construction, not the separate smoothed B-spline prior used by img2img.

The selected path itself has difficult transitions: reference 3 → 4 (zero-based)
rotates approximately 175.31 degrees across 5.38 scene units. The old straight
segment approaches within about 0.65 units of the least-squares viewing center.
This is a plausible contributor to the visual failure, **not proof of a camera
encoding defect**. A camera spline does not infer obstacle clearance or ensure
multi-view consistency of independently repaired images. Selection order remains
explicit and is never silently reordered.

Before diffusion, `seva-camera-audit.json` now verifies the actual upstream parser
outputs against staged images, OpenCV poses, pixel intrinsics, split membership,
and sampled first-pass anchors. The existing reconstruction gate still checks
SEVA's exported poses and calibration. Neither gate measures whether the generated
pixels depict the requested geometry. CPU regression tests cover moving/rotating
keyframes, exact endpoints, 400 distinct outputs and parser mismatch rejection;
a fresh GPU generation/reconstruction is needed to assess visual improvement.


## Manual stop and partial PLY

New OracleGS reconstructions support **Stop & save PLY** on the result details
page. The worker lets the current iteration finish, synchronizes CUDA, and
atomically publishes the current Gaussian model before exiting. This works even
before iteration 3000. The downloaded `oraclegs-partial.ply` is explicitly marked
incomplete and records its saved iteration. It is a model export, not a resumable
optimizer checkpoint.

The stop-only hook is inserted in memory after the pinned trainer's iteration
updates; source hashes are verified and the shared upstream checkout is unchanged.
The parent waits up to 240 seconds for a cooperative save before terminating a
stuck process. A failed or timed-out save falls back to the last confirmed
scheduled snapshot, when one exists. Before reconstruction begins, stopping
retains generated artifacts but cannot create a PLY. Processes already launched
with the old worker cannot acquire this hook; they retain scheduled-snapshot
recovery until a new run is launched.

### Diffusion-boundary audit and stale-run protection

The additional review of the supplied video and cached `run_20261016_183036`
confirms that this run predates the per-transition trajectory manifest. Its 80
exported target cameras span `[12.283559, 1.122240, 4.441781]` scene units;
translation was not absent from those poses. The five selected transitions rotate
approximately 86, 102, 45, 175 and 109 degrees. The video shows strong turning and
geometry changes, but does not identify a camera-encoding error by itself.
Its geometry gate reports only 52 triangulated points and positive depth scales
for two of six references. These are weak geometric constraints, not evidence of
successful multi-view reconstruction. The exact cause of the generated pixel
inconsistency remains unverified without a fresh GPU run.

New workers reject legacy total-budget staging before provisioning or inference;
start a fresh run to stage 80 targets for each adjacent pair (400 for six views).
The result page now distinguishes that count from historical total-budget runs.
`seva-conditioning-audit.json` checks every call to the upstream diffusion
conditioning function in both passes. It permits SEVA's centering and uniform
positive scale, verifies preserved rotations and relative translations, and checks
that Plucker ray moments encode the normalized camera centers. This closes the
previous audit gap between parsed poses and diffusion ray conditioning. It does
not modify the pinned upstream sources, model, chunking, or normalization, and
does not establish that generated pixels obey the supplied cameras. Both camera
audit reports are linked from run details. CPU regression tests inject erased ray
moments and collapsed camera centers to verify rejection; real CUDA execution of
this new audit and visual improvement remain to be validated.


### Ordered reference keyframes (trajectory manifest v3)

No OracleGS-specific orbit override was found. The active wrapper calls SEVA
before attention extraction or OracleGS training; `img2trajvid` reads supplied
poses and does not enter SEVA's preset-orbit branches. The previous integration
nevertheless discarded known reference positions in the sequence by exporting
references as a prefix and generating separate targets at those same poses.
SEVA's `img2trajvid` interpolation deliberately excludes these unordered GT
references from its temporal anchor ordering. A passing pose/ray audit cannot
prove that the resulting generated pixels respect translation.

Version 3 uses the native ordered-reference path described above. The change is
an explicit SEVA input/task adaptation, not an alteration to OracleGS training
or diffusion weights. Version 2 staging must be regenerated. Reload the persistent queue manager
(`scripts/start.sh` detects changed code) before queuing a fresh run; creating a
new job alone does not reload Python modules. Existing running processes retain
their original request and cannot acquire this change in place.
The saved-pose and off-path-reference modes remain available and are disclosed
in `seva_task` and `trajectory.reference_conditioning`.

Regression checks execute the pinned SEVA parser, prior selection, interpolation
chunking and final reference inclusion on CPU. They check 400 target queries,
all six reference anchors at indices 0/81/162/243/324/405, conditioning within
segment boundaries, and unchanged reference pixels. Dataset tests verify native
output reordering before pose/calibration checks and synthetic assembly at both
resolutions. This verifies the implementation contract; a new GPU generation is
still required to establish that it corrects the stationary-looking output.


### Independent pair generation (trajectory manifest v4)

Version 4 supersedes the version 3 whole-path invocation. Each pair is staged in
`inputs/segments/NNN/scene` with all original references followed by only that
pair's targets. Generation proceeds sequentially into `seva-segments/NNN`; the
validated merge records each target's source in `seva-segments.json`. Attention
extraction and OracleGS reconstruction run once on the combined dataset.

Fresh staging is required; workers reject earlier manifest versions. Reload the
persistent queue manager before starting a fresh run. Existing generation
processes keep their original inputs. CPU tests validate staging, sequential
invocations, native SEVA chunking, and merge contracts; GPU image quality must
be assessed on a new run.

All saved references also remain present in every independent SEVA segment and in both generation passes (`gt-nearest` first pass, `interp-gt` second pass). Endpoint selection for VGGT uses the saved interpolation alpha, not Euclidean nearest-camera distance. Improved reconstruction quality requires a fresh GPU comparison; existing outputs are unchanged.

The run details dashboard also exposes the authors’ original RGB/plasma attention grids (`*_grid.png`), including legacy flat exports and per-query exports under `attention-raw/chunk_*`. These are padded VGGT-space overlays, not unpadded training weights. New query grids display their source reference, segment membership and assigned-target count; `attention-chunks.json` is linked for exact membership.

## Optional source-geometry experiments (Step 4)

Two independent OracleGS selectors keep the existing defaults:

- **Depth supervision:** `oracle_depth_source: predicted` (default) runs Depth Anything V2 and SfM scale/offset alignment. `source_splat` renders the original checkpoint scene at the reference cameras, fits depth to the staged RGB calibration, converts positive camera-Z depth to `1/z`, and stores float32 inverse depth in scene units. There is no min/max normalization, 16-bit quantization, monocular fallback, or SfM rescaling in this mode. Invalid coverage, a one-pixel boundary, and local depth discontinuities above 10% are masked out. The regular reference-only depth loss and weight schedule remain unchanged; synthetic views still have no depth loss. Front-surface depth is a source-geometry prior, not the trainer's exact alpha-composited inverse-depth rendering.
- **Reconstruction initialization:** `oracle_initialization: sfm` (default) retains the current sparse-point initialization. `source_splat` loads the staged source Gaussians with positions, scales, rotations, opacities, and available SH appearance before ordinary OracleGS optimizer setup. It starts at iteration 1 with fresh optimizer state and the normal schedule, rather than resuming an optimizer checkpoint. The scene-dependent position learning-rate scale and radius buffers are initialized. Existing source files are never modified or merged into the output. The configured Gaussian cap limits additions; an existing larger source is not truncated.

Standard binary PLY initialization preserves the original file and all supported SH coefficients (degrees 0–3). SOG initialization uses the existing decoded viewer working set, including DC colors; higher-order SH is not decoded by that loader. The chosen representation, source fingerprint, Gaussian count, SH degree, asset hashes, and both settings are recorded in the input manifest and recipe. Source assets are staged locally and transferred with the job. SEVA generation, VGGT attention, RGB losses, MCMC updates, and graceful stopping continue normally. Fixed-pose COLMAP preparation still runs; it supplies depth alignment when predicted depth is selected and initialization when SfM is selected.

CLI equivalents are `--oracle-depth-source source_splat` and `--oracle-initialization source_splat`. Either option can be enabled alone. Use a fresh run after reloading the persistent server/queue process; existing jobs retain their staged settings. These adapters do not modify pinned upstream files. CPU regression tests cover the four combinations and both dataset modes; GPU training quality must be compared on a new run.

## SEVA source-depth guidance (experimental)

`seva_geometry_guidance: off` is the default, independent of
`oracle_depth_source` and `oracle_initialization`. Step 4 exposes **SEVA geometry
guidance → Source-splat depth**, and the CLI accepts
`--seva-geometry-guidance source_splat`. The choice is frozen into each queued
run, its staged manifest, model requests and recipe. A fresh run is required.

SEVA has no native predicted-depth conditioning input. This option is an
inference adapter, not replacement of a predicted depth tensor: it reprojects
saved reference RGBs using source-splat camera-Z depth and the unnormalized
OpenCV cameras. It selects one reference per target using translation and
rotation distance, resolves projected sample collisions with a target-Z buffer,
and excludes invalid depth, uncovered pixels and coverage boundaries. At 720p,
target intrinsics include SEVA's vertical padding; the reference depth remains
in its original calibrated 720p grid.

Both sparse-anchor and dense generation use these RGB reprojections as soft
constraints on denoised latent estimates (strength 0.35). The same affine blend
is applied to both classifier-free guidance branches; conditioned input frames
remain untouched. Conservative pooling and erosion restrict latent guidance to
covered areas. Uncovered regions remain generative. This does not introduce new
model channels, change weights, or modify pinned upstream source files. The
per-invocation `seva-geometry-guidance.json` records source-reference indices,
projected/latent coverage and completed calls and is linked in run diagnostics.

The source-only Z buffer cannot detect occluders absent from the selected source
view. Edited RGB geometry can disagree with source depth. Guidance is therefore
an experimental prior, not a guarantee of geometry consistency. CPU projection,
configuration and dispatch checks do not validate its CUDA inference behavior or
visual quality; compare a fresh guided run against the same seed with guidance
off before relying on it. Enabling this setting alone retains predicted depth
for reconstruction and SfM initialization.

### Investigation of the supplied anchor pair

Pixel hashes matched `file-7.png` and `file-8.png` to cached run
`run_20261016_214940`, pair 000, first-pass anchors `013.png` and `014.png`.
The parser audit maps these to target indices 73 and 79 (zero-based), at
trajectory alpha 74/81 and 80/81. Their camera delta is about 0.530 scene units
and 6.385 degrees; the latter is still 0.0883 units and 1.064 degrees from the
ending reference. Thus these are consecutive sparse anchors, not consecutive
dense frames. The native CPU schedule generates all 15 anchors jointly with
all six reference RGBs in one 21-frame call. Dense generation then uses seven
calls, each retaining all six references. A cross-call seam or dropped reference
does not explain this particular anchor pair.

That run had no source-depth guidance during generation. VGGT source-reference
assignment occurs after SEVA and affects training attention, not anchor creation.
The geometry gate passed with only 52 triangulated points and positive depth
scales for two of six references; the ending reference `ref_000001` had scale
zero. This gate checks minimum availability, not adequate geometric coverage.
It cannot establish generation quality. The cached pose/ray audits passed and
show consistent normalization within this segment, so they do not support a
translation-loss or camera-reset explanation for this pair. Pixel-level
multi-view consistency of independently edited references remains unverified.


### Ordered independent pairs (trajectory manifest v5)

Version 5 retains independent pair generation and restores ordered saved endpoints.
Each segment is staged as `[off-path references, A, interior targets, B]` using
`img2vid`. All saved references remain RGB conditioning in both passes. The
first pass uses native `gt-nearest` joint generation; the dense-pass adapter
retains every reference exactly once per chunk but uses only A, B and generated
on-path anchors as temporal bounds. Off-path references cannot bound intervals.
The context budget includes references, bounds and targets. No upstream source
or diffusion weights change; anchor count and CFG settings remain unchanged.

Every generated anchor and target has an explicit source assignment: A through
the midpoint, B afterward, matching the existing VGGT endpoint policy. The
`seva-ordered-pair.json` report records actual anchor indices, target assignments,
and dense chunk bounds. Source-depth reprojection, when enabled, consumes that
assignment rather than independently choosing the nearest camera. With guidance
off, source assignment is provenance; it does not give one reference exclusive
control over SEVA's joint RGB conditioning.

Per-pair videos include A, targets and B, excluding off-path context images.
PNG/export indices retain every reference for validated canonical merging. The
diagnostics gallery labels assigned sources and links the per-pair report.
Generated views remain synthetic; saved reference pixels remain reconstruction
GT. Both resolutions retain their existing calibration/padding handling.

Version 4 inputs are rejected. Reload the persistent queue/server and stage a
fresh run to use version 5; running jobs cannot acquire it in place. CPU tests
exercise native SEVA task parsing and first-pass chunking, the ordered dense
adapter, source assignments, video slicing, merge contracts and source-depth
selection. A fresh GPU comparison is still required to assess visual quality.

### Reference images and connected-path splitting

On `/splatfix`, OracleGS exposes **Reference all images** (default) and
**Reference single image** below Input images. All-image mode supplies every
saved RGB to each SEVA run; single-image mode supplies only that run's source
RGB. The selected baseline/repaired input mode still determines the RGBs used.
All saved references remain ground truth for reconstruction.

Connected trajectories default to **Single-split**, preserving whole pairwise
legs. **Double-split** divides each leg's existing target samples into disjoint
inward halves and reverses the second half. Six saved views therefore produce
ten SEVA runs instead of five, preserving the total target count (400 by default).
Single-image mode enforces double-split, including for API and CLI requests.
With all-image references, either split is selectable.

Preset trajectories always remain one intact clip per saved view; splitting
does not apply to them. They support both reference-image settings. The legacy
saved-poses-only adapter requires all-image references. CLI equivalents are
`--oracle-reference-mode all|single` and
`--oracle-split-mode single-split|double-split`.

## Reusable 54° SEVA captures

Step 2's camera icon queues `seva_capture`, a local, queue-managed stage. It
renders each missing view at its saved OpenCV pose, with **1024 × 576** pixels
and **54° horizontal FOV** (`fx = fy ≈ 1004.86`). RGB comes from Viser; depth
comes from the existing CPU front-surface rasterizer at the identical camera.
Keep the capture visor connected. This does not select new poses or edit images.

Each view gains a `seva_capture` record and `views/<id>/seva/` assets, including
calibration, source fingerprint, RGB/depth hashes and separate repair slots.
A view is published only after both captures validate. Retries skip completed
views, including after a partially completed capture job. The per-view button
captures only that view; the Step 2 icon captures all remaining views. Original
images, original repairs and saved poses are retained.

Step 3's **Images to repair → SEVA captures** sends those new RGBs to the existing
GPT-image backend. Repairs and original responses are stored in the SEVA image
set. The original GPT-image repair prompt and API request remain unchanged.
The full-resolution response bytes and a full-resolution PNG preview are retained,
and a separate 1024 × 576 PNG is produced locally by full-frame Lanczos resizing.
No center crop is applied to SEVA repairs. Raw dimensions, aspect-ratio comparison
and the x/y resize factors are recorded in `image_repair`. OracleGS reads the
normalized repair and its matching capture calibration, not the raw response.
Original-capture editing retains its existing fitting behavior.

New dashboard OracleGS jobs default to `oracle_capture_source: seva`; the
**Original captures · legacy calibration** selection retains the old path.
New direct CLI reconstructions likewise default to SEVA. Historical jobs with
no capture-source field retain their saved-capture behavior. Missing captures
or missing SEVA repairs fail before staging; original repairs are never silently
substituted. Run staging only reads these references. When source-depth guidance
or supervision is selected, it reuses the cached depth rather than rendering it
again. Depth-guidance, reconstruction, trajectory and image-fitting settings
remain independent. The optional 720p setting resamples these 576p captures; it
does not recapture at higher resolution.

CLI:

```sh
python -m splat_explorer.splatfix.cli seva-capture CHECKPOINT
python -m splat_explorer.splatfix.cli seva-capture CHECKPOINT --view-id 000
python -m splat_explorer.splatfix.cli edit CHECKPOINT --image-set seva
```

For new manual references, choose **54° · SEVA** in the dropdown beside
**Custom** in Step 1. Viser switches to the 54° horizontal camera profile and
captures each saved position at 1024 × 576 with matching depth. These new
checkpoints contain only the SEVA image set; original-FOV RGB and repair slots
remain empty. The gallery defaults to 54° for these checkpoints, and Step 3
can repair the captured SEVA references directly without a Step 2 recapture.

### Source depth without triangulation

Select **Reconstruction depth supervision → Source-splat depth** to replace
monocular depth and its SfM alignment with staged camera-Z inverse depth. This
is independent of **SEVA geometry guidance**, which only affects generation.
Select **Reconstruction initialization → Existing source splat** as well to
skip COLMAP feature extraction, matching, triangulation and export entirely.
The adapter writes the saved cameras in COLMAP format with an empty point set;
the trainer loads the staged source Gaussians. With SfM initialization selected,
triangulation is still required to initialize the reconstruction.

The source-depth runtime imports `scene` before camera utilities to avoid the
upstream circular import. `source-geometry-runtime.json`, linked in diagnostics,
records reference names, dimensions, valid fractions and loaded Gaussian count
as the trainer actually consumes them. It is distinct from the staging gate.
The upstream `model/input.ply` remains a dataset placeholder, not proof of the
actual initialization. Source depth retains scene units, calibrated image fitting
and invalid-pixel masks; no predicted-depth fallback is used. Its front-surface
estimate is a geometry prior, not exactly the alpha-composited inverse depth
rendered by the training rasterizer.

Reload the queue/server and create a fresh run to apply these changes. Existing
staged runs keep their saved settings. CPU regression tests cover the real
upstream camera import order and all depth/initialization combinations; GPU
loss behavior and visual quality still require a fresh source-guided run.

### Scene-memory references (new-job default)

`oracle_reference_mode=scene_memory` conditions sparse SEVA generation on eight
fixed WorldExplorer scaffold images, five prior generated views, and the six
saved references. `mode=edited` is the default: all six saved references are
GPT-repaired images. Explicit `mode=baseline` substitutes six original captures;
the two sets are never combined. Scaffold and memory images are generated scene
context, not additional original RGB captures or repaired ground truth.

The five memory views compete in one pool of original WorldExplorer outputs and
completed SEVA segments. Selection votes over nearest SO(3) rotation angles, using
WorldExplorer's trace/arccos distance. Opposite trajectory origins (more than 90°
away) and each trajectory's first two generated frames are excluded. Selection is
performed per segment over its target cameras; arbitrary saved trajectories use
their starting reference orientation instead of WorldExplorer's discrete sector
IDs. This adapts the paper's memory idea rather than reproducing its entire sampler.

Sparse batches use 27 slots (19 references plus up to eight anchors); the dense
pass keeps the existing 21-slot interpolation and six-reference handling. Memory
is conditioning only and does not change exported frame identities or GT labels.
Per-segment `scene-memory/selection.json` records sources; `conditioning.json`
records actual conditioning slots. Longer paths can require multiple sparse batches.

This setting requires six calibrated saved views, all eight posed scaffold
images, and at least five eligible prior generated views. Missing context causes
an explicit error. Select `oracle_reference_mode=all` (**Repaired images only**)
to restore the original behavior, or `single` for the existing one-image option.
Historical staged inputs without this field retain legacy validation semantics.

The default `oracle_preset=spiral_expanding` option (default size: 0.25 scene units per demo unit) keeps the original two-turn
spiral available as `spiral`. It scales forward/backward translation in the saved
camera's local Z axis by 1.5, and smoothly grows the canonical spiral radius from
1.0 to 1.25 during the second turn (smoothstep, reaching 1.25 at the last sampled
frame). The first turn's radius is unchanged. Cameras continue looking toward
the original focus; the saved starting camera, intrinsics, frame count, and
scene-unit size setting are preserved. This expanding path is not closed.

### Optional fixed initial source for preset clips

`oracle_source_reference=nearest` is the default and preserves prior behavior,
including queued jobs whose settings or staged manifests omit this field. Native
SEVA pools the global image embedding across conditioned frames; optional source
geometry guidance chooses a reference by pose/rotation proximity. There is no
native exclusive RGB “source” in this mode.

The opt-in `fixed_initial` fallback uses each preset clip's starting saved image
as the sole input to SEVA's global image embedding in **both** generation passes.
All other reference RGB latents, masks and cameras remain spatial conditioning;
this does not add an image or remove the scene-memory references. If source-depth
guidance is enabled, its source is fixed to the same saved image. This option does
not enable depth guidance itself. GPT-repaired mode stays the default; explicit
baseline mode uses the original capture instead, never both. Connected pairwise
paths retain their existing behavior; this option applies to per-view presets.

This is an explicit conditioning extension, not a guarantee that the model will
ignore conflicting context. Each clip writes `seva-source-reference.json` with the
source index, image checksum, input mode and sampling-batch counts. Old manifests
without the setting are never upgraded implicitly.
