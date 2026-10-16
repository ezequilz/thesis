# OracleGS in Splatfix Step 4

OracleGS is an independent reconstruction method under `src/splat_explorer/splatfix/oraclegs/`. Select **OracleGS** and **GPT-image improved** in Step 4. The repaired RGBs are the ground-truth/reference dataset at their saved, calibrated camera poses. They are not replaced by SEVA outputs. Baseline input mode remains available for a comparison using original RGBs.

## Pipeline

1. Stage saved references, validate OpenCV camera-to-world rotations/calibration, preserve the source files, and downscale RGBs with Lanczos. Center-crop by default (`seva_fit: crop`), uniformly resizing to cover the output and updating focal lengths and principal points by the same image-space affine. Non-16:9 checkpoints are supported. Repaired-image aspect ratio must still match its saved camera: a previously unknown crop cannot be inferred safely. Padding (`seva_fit: pad`) is an explicit fallback to preserve framing. Inputs that require upscaling are rejected. Saved originals remain unchanged.
2. Use the calibrated reference cameras with COLMAP feature extraction, exhaustive matching and fixed-pose triangulation. Match the COLMAP database IDs explicitly; permit two-view tracks for sparse inputs. Before depth or SEVA inference, verify that SfM retained every staged reference identity, pose and calibration. Save pixel-reprojection checks in `camera-alignment.json`; reject drift rather than continuing with misaligned supervision. No source-splat geometry initializes this reconstruction. Estimate reference pseudo inverse depth with Depth Anything V2 Large, then run the authors' `utils/make_depth_scale.py` to align depth to SfM. Empty triangulation or entirely unreliable depth fails the run.
3. Stage SEVA `transforms.json` in OpenGL camera-to-world coordinates (`c2w_cv @ diag(1,-1,-1,1)`) and `train_test_split_N.json`. References retain their independent input order. The ordered target path includes each saved path pose exactly, plus intermediate poses. By default the path follows checkpoint selection order; `oracle_path_order` can supply a separate list of checkpoint indices through configuration or the job API. Camera translations and orientations use the natural Kochanek–Bartels keyframe splines from SEVA’s GUI (tension 0, open path); saved endpoint poses remain exact. Intrinsics interpolate between calibrated endpoints. No image interpolation is implemented here.
4. Call the pinned SEVA `demo.main` with `task=img2trajvid`, `use_traj_prior=True`, `chunk_strategy=interp-gt`, `chunk_strategy_first_pass=gt-nearest`, and per-pass CFG `[3.0, 2.0]`. SEVA itself selects and generates sparse anchors in the first pass, then generates the full ordered trajectory in its second pass. The context window/anchor count remains adaptive as in its published implementation. Default target count: 80 per pair **per consecutive viewpoint pair**, including both endpoints. Six views produce five segments and 400 outputs. Junction poses occur twice with distinct image identities. Both saved poses and intermediate poses receive synthetic outputs.
5. Run the authors' unmodified `vggt_aten/visualize_attn.py` implementation on the combined reference and synthetic images. The released global-attention extractor uses layers 0 and 22 with weights 1/4 and 3/4, per-layer normalization, source-frame index 0, and its existing batching behavior. Reference and synthetic images at the same pose have distinct filenames in this combined set. A fixed NumPy seed makes the authors' shuffle reproducible. All synthetic views must have matching attention maps.
6. Run the original OracleGS optimizer and its pinned inverse-depth CUDA rasterizer. Ground-truth views use photometric supervision plus reliable SfM-aligned inverse-depth supervision. Synthetic views retain `is_synthetic=True` and use attention-weighted L1, weighted SSIM and globally weighted LPIPS. The original MCMC Gaussian updates, regularizers and GT/synthetic sampling code are unchanged. Defaults: 22,000 iterations, LPIPS 0.3, synthetic SSIM 0.3, scale/opacity regularization 0.01, noise LR 500000, `cap_max=1800000`, `init_type=sfm`. Inputs are already resized, so training uses `-r 1`.
7. Save native 3D Gaussian PLY, recipe, command requests/logs, depth products, sparse model, SEVA outputs and attention maps. The queue downloads results and exposes the PLY through the existing results UI. Cancellation terminates the active subprocess group; completed intermediate products are retained. A stopped run is not advertised as a completed reconstruction.

## Separate adapter options

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

The OracleGS README's generation example uses `img2img`, CFG 3 and `T=80`. Our user-requested trajectory setup uses SEVA's documented sparse-reference `img2trajvid` two-pass recipe, including CFG `[3,2]` and adaptive context length. The number of target images is distinct from the SEVA context window `T`. Consequently this is an original-optimizer Splatfix baseline with an explicitly recorded input/generation configuration, not a claim to reproduce the paper's benchmark metrics. The Gaussian cap is configurable because the paper derives it per scene from another baseline. No held-out GT exists along the custom trajectory; generated images must never be reported as held-out ground truth for PSNR/SSIM/LPIPS evaluation.

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
source, attention method, losses, optimizer or training schedule is modified.


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
