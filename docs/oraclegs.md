# OracleGS in Splatfix Step 4

OracleGS is an independent reconstruction method under `src/splat_explorer/splatfix/oraclegs/`. Select **OracleGS** and **GPT-image improved** in Step 4. The repaired RGBs are the ground-truth/reference dataset at their saved, calibrated camera poses. They are not replaced by SEVA outputs. Baseline input mode remains available for a comparison using original RGBs.

## Pipeline

1. Stage saved references, validate OpenCV camera-to-world rotations/calibration, preserve the source files, and downscale RGBs with Lanczos. Rescale intrinsics to match. The presets require 16:9 cameras and images at least as large as the requested output; incompatible images fail instead of being stretched, cropped or silently upscaled. Choose 720p captures in Steps 1–2 if creating new compatible checkpoints.
2. Use the calibrated reference cameras with COLMAP feature extraction, exhaustive matching and fixed-pose triangulation. Match the COLMAP database IDs explicitly; permit two-view tracks for sparse inputs. No source-splat geometry initializes this reconstruction. Estimate reference pseudo inverse depth with Depth Anything V2 Large, then run the authors' `utils/make_depth_scale.py` to align depth to SfM. Empty triangulation or entirely unreliable depth fails the run.
3. Stage SEVA `transforms.json` in OpenGL camera-to-world coordinates (`c2w_cv @ diag(1,-1,-1,1)`) and `train_test_split_N.json`. References retain their independent input order. The ordered target path includes each saved path pose exactly, plus intermediate poses. By default the path follows checkpoint selection order; `oracle_path_order` can supply a separate list of checkpoint indices through configuration or the job API. Camera translations and orientations are interpolated to define query poses; no image interpolation is implemented here.
4. Call the pinned SEVA `demo.main` with `task=img2trajvid`, `use_traj_prior=True`, `chunk_strategy=interp-gt`, `chunk_strategy_first_pass=gt-nearest`, and per-pass CFG `[3.0, 2.0]`. SEVA itself selects and generates sparse anchors in the first pass, then generates the full ordered trajectory in its second pass. The context window/anchor count remains adaptive as in its published implementation. Default target count: 80. Both saved poses and intermediate poses receive synthetic outputs.
5. Run the authors' unmodified `vggt_aten/visualize_attn.py` implementation on the combined reference and synthetic images. The released global-attention extractor uses layers 0 and 22 with weights 1/4 and 3/4, per-layer normalization, source-frame index 0, and its existing batching behavior. Reference and synthetic images at the same pose have distinct filenames in this combined set. A fixed NumPy seed makes the authors' shuffle reproducible. All synthetic views must have matching attention maps.
6. Run the original OracleGS optimizer and its pinned inverse-depth CUDA rasterizer. Ground-truth views use photometric supervision plus reliable SfM-aligned inverse-depth supervision. Synthetic views retain `is_synthetic=True` and use attention-weighted L1, weighted SSIM and globally weighted LPIPS. The original MCMC Gaussian updates, regularizers and GT/synthetic sampling code are unchanged. Defaults: 22,000 iterations, LPIPS 0.3, synthetic SSIM 0.3, scale/opacity regularization 0.01, noise LR 500000, `cap_max=1800000`, `init_type=sfm`. Inputs are already resized, so training uses `-r 1`.
7. Save native 3D Gaussian PLY, recipe, command requests/logs, depth products, sparse model, SEVA outputs and attention maps. The queue downloads results and exposes the PLY through the existing results UI. Cancellation terminates the active subprocess group; completed intermediate products are retained. A stopped run is not advertised as a completed reconstruction.

## Separate adapter options

`oracle_adapter: trajectory` is the requested six-reference-plus-intermediates configuration. The official COLMAP reader expects a dataset RGB at every target pose and would skip a target without one. Dataset assembly supplies generated RGBs at intermediate target entries, then a narrowly scoped reader adapter removes these entries from the **GT camera list before training selection**. The original synthetic camera objects and all loss/attention code remain untouched. At saved poses, repaired reference RGBs stay in `images/`, and generated RGBs go in `synthetic_N/`. The adapter never promotes generated RGBs to GT.

`oracle_adapter: saved_poses` generates only at the saved path poses. All target poses have real repaired reference images, so the original dataset loader runs without the reader adapter. This is an unmodified-loader diagnostic option, not a reproduction of the paper's benchmark splits or a substitute for the requested full trajectory.

The two options are shown separately in Step 4 and recorded as `checkpoint-trajectory-adapter` or `saved-poses-original-loader` in results. The default is `trajectory`.

## Resolution

- **576p (default):** 1024 × 576, `L_short=576`. Both dimensions satisfy SEVA's 64-pixel spatial stride. Reference images are downscaled before inference.
- **720p experiment:** reference/output images are 1280 × 720. The released SEVA U-Net cannot directly process 720 pixels in height: its three latent downsamplings and skip concatenations require multiples of 64. A separate, recorded resolution adapter pads 24 pixels above and below to 1280 × 768, shifts `cy` by 24, runs the original model with explicit dimensions, and crops the generated outputs back to 1280 × 720. No stretching, model changes, or replacement generation algorithm is used. Native 1280 × 720 diffusion is **not** claimed. Attention and reconstruction consume the cropped 720p images with the original scaled calibration.

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

CPU tests cover camera projection and conventions, exact saved-pose inclusion, repaired-image preservation, separate GT/synthetic membership using the actual upstream COLMAP reader, database camera IDs, 720p padding/cropping, output identity checks, source hashes, option validation, worker cancellation, and full stage dispatch with model subprocess test doubles. These tests do not establish CUDA kernel compatibility, successful model downloads, generated-image quality, or end-to-end GPU training. A CUDA smoke run and a full six-reference reconstruction remain necessary before claiming a validated reproduction.
