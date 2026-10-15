# G4Splat reconstruction in SplatFix

Step 4 offers **G4Splat** for completed saved selections with at least two
camera positions. Choose original RGBs or saved GPT-image repairs, then Run now
or Schedule. ArtiFixer remains the default for old jobs and benchmarks.

The adapter in `src/splat_explorer/splatfix/g4splat/` runs the official
[DaLi-Jack/G4Splat](https://github.com/DaLi-Jack/G4Splat) checkout at
`ec0736126707a42bb2c26ed8ba2c314909edc7a9`. It invokes `train.py`
with the README's posed recipe: `--sfm_config posed --use_view_config
--config_view_num N --select_inpaint_num 10 --tetra_downsample_ratio 0.25`.
`N` is the actual saved selection count, not the paper's five-view example.
All selected images are training inputs. Geometry is inferred by MASt3R, with
plane-aware depth refinement and all three See3D rounds retained. Source splat
points, opacity, ArtiFixer trajectories and ArtiFixer weights are not used.
The original optimization configurations, mesh extraction and rendering run
unchanged. Only the final held-out benchmark evaluator is skipped because saved
selections contain no held-out photographs or ground-truth mesh. Commands,
revision, input hashes, cameras and adaptations are recorded in the run folder.
New runs apply the explicit `camera-consistency-v1` correction described below;
they must not be reported as an unmodified-upstream baseline.

## Camera and image contract

Camera positions alone cannot determine focal length or principal point.
SplatFix checkpoints already save the horizontal field of view and dimensions:
`fx = fy = width / (2 tan(horizontal_fov / 2))`, `cx = width / 2`,
`cy = height / 2`. The shared resolution preparation recaptures originals in
Viser at the selected dimensions and adjusts focal length for the uniform
cover/center crop. Saved full GPT responses use the corresponding center crop.
No GPT call is made. Source checkpoints remain unchanged.

The input exporter writes one COLMAP PINHOLE camera per image. Saved OpenCV
camera-to-world transforms are inverted to world-to-camera rotations and
translations, with COLMAP's scalar-first quaternion order. There are no
fabricated feature tracks or initial points. The upstream `posed.yaml` requests
fixed calibration and poses, but upstream still optimizes translations internally
before restoring the supplied poses.

### Camera-consistency correction and required geometry check

The shared upstream checkout remains pristine. `corrected_runtime.py` verifies
the exact SHA-256 of the pinned MASt3R entrypoint and creates a run-local
`corrected-runtime/` overlay. A source mismatch aborts the run. Its correction:

- Applies a real centered crop to RGB and subtracts the crop offset from the
  principal point. The saved full-resolution image has exactly the aspect ratio
  of its MASt3R point map, avoiding a hidden crop followed by stretching depth
  over a different field of view. For 832x480 inputs this saves 832x468 pixels
  (six rows removed from each edge) with `cy=234`. The original staged image
  remains 832x480 with `cy=240`. The 1280x720 and 960x720 profiles need no crop.
- When upstream restores calibrated poses, transforms every view's point map
  from its previous camera frame to the restored frame, preserving camera-local
  coordinates and depth. A non-positive or non-finite global alignment scale
  aborts the run. This restores coordinate consistency; it does not guarantee
  that predicted scene depth is accurate.
- Records exact pointmap intrinsics and crop calibration. Immediately before
  `scripts/align_charts.py`, checks every point against its originating pixel.
  All points must be finite and have positive camera z; maximum reprojection
  error must be at most 0.5 pixels. Image dimensions, scaled pointmap intrinsics,
  centered principal points and full-resolution COLMAP intrinsics must agree.
  Invalid samples are not hidden by confidence masks or percentile thresholds,
  because upstream depth alignment consumes all of them.

`geometry-correction.json` records the patch/helper hashes;
`reprojection-check.json` records per-view errors and the pass/fail outcome.
Failure stops depth alignment and prevents publishing a completed result. The
recipe and result identify the correction version. Existing outputs are not
rewritten; rerun reconstruction to apply the correction. Strict checks may reject
an upstream point map with invalid depth even after its poses are consistent.

## Native output and viewer compatibility

The official implementation uses **2D Gaussian surfels**. Its native two-scale
PLY remains at `official/free_gaussians/point_cloud/iteration_7000/point_cloud.ply`.
It is not a volumetric 3D Gaussian reconstruction. `g4splat-viewer.ply` adds a
normal-axis radius of 1% of the smaller tangent radius so the existing 3DGS viewer
can load it. This is an explicitly recorded visualization approximation;
positions, rotations, tangent scales, opacity and SH coefficients are preserved.
Use the native PLY and official renderer for method-quality comparisons.

## GPU environment

The default runtime and dashboard configuration enable `splatfix.g4splat_runtime.auto_setup`.
On the CUDA worker, missing installations are provisioned by the packaged
`g4splat/provision.sh` script under a filesystem lock. It clones the pinned
upstream source, creates a separate Python 3.9 / PyTorch 2.0.1 / CUDA 11.8 prefix,
pins the CUDA compiler to 11.8 and CMake below 4 for the upstream build files,
selects the allocated GPU's architecture for PyTorch extension builds and installs
the OpenGL/GLib libraries required by upstream OpenCV,
pins the otherwise unconstrained `kiui` dependency to Python 3.9-compatible 0.2.20,
builds the official rasterizer, KNN, tetrahedralization and MASt3R extensions,
and downloads DepthAnythingV2, MASt3R, SAM and the official See3D sparse-view
weights. Provisioning does not modify shared source or model weights. The run-local
calibration correction is applied after runtime preflight; depth models, losses
and the three generative rounds are retained. The See3D model snapshot is pinned to
`cbf13b6f813137134907408e40d3f2a17d6f0a80` in `bruiiii/See3D`.

Installation requires Linux x86_64, access to the public package/model hosts,
and an NVIDIA GPU. The authors report A100 80GB testing. Downloads and build
products persist on the mounted workspace across allocations. Installation
uses its own environment; application and ArtiFixer packages are not changed.
The first run can take substantially longer while dependencies and weights
are installed. STOP and the job deadline also apply during automatic setup.
Concurrent runs wait for the active installer to release its filesystem lock.
The lock file is created automatically and may remain after setup; its existence
does not mean installation is still running.

For manual provisioning inside the GPU container:

```sh
bash /workspace/code/src/splat_explorer/splatfix/g4splat/provision.sh \
  /workspace/third_party/G4Splat /workspace/g4splat-env \
  ec0736126707a42bb2c26ed8ba2c314909edc7a9
```

Configure `splatfix.g4splat_runtime.repo` and `.python` to use an existing
installation; set `auto_setup: false` to prohibit automatic installation.
Runtime validation rejects a different revision, modified tracked source,
missing or empty primary weights, and incomplete See3D checkpoints. Executable-bit
changes introduced by the DSS filesystem are ignored; tracked file content is
still checked. Before
training, `runtime_probe.py` checks CUDA execution and compiled dependencies;
its output is saved in `preflight.log`. Child `python` commands use the official
interpreter with application `PYTHONPATH` and user-site packages disabled.
The container's PyTorch/TensorRT library paths are excluded from
`LD_LIBRARY_PATH`, preventing the NGC image's libraries from loading into the
separate G4Splat environment. Temporary dependency builds use node-local `/tmp`
to avoid DSS executable-bit changes during Git checkouts.
The upstream scripts concatenate shell commands, so runtime and run paths must
contain only letters, digits, underscores, slashes, dots and hyphens.

```sh
splatfix --config configs/splatfix.yaml repair /path/to/checkpoint \
  --reconstruction-method g4splat --mode edited --output /path/to/results \
  --g4splat-repo /workspace/third_party/G4Splat \
  --g4splat-python /workspace/g4splat-env/bin/python
```

The dashboard uses the existing LRZ allocation, durable queue and lease. Local
Viser capture occurs before staging; only calibrated images and COLMAP inputs
are transferred. Worker status, restart reattachment, STOP/deadline handling and
artifact download use the existing job contract. Use **Stop gracefully & download** to retain an incomplete reconstruction.
During surfel training, STOP/deadline requests save the current native PLY after
optimizer updates and attempt an authors-format checkpoint before exiting. The
run-local trainer hook verifies the pinned source hash; scheduled saves and the
shared checkout are unchanged. A thin-Gaussian viewer PLY is also exported.
During image generation or other non-training stages, cancellation retains the
last saved surfels. Before any surfels exist, only inputs and diagnostics can be
retained. A hung trainer is terminated after 240 seconds (configurable through
`SPLATFIX_SAVE_STOP_TIMEOUT_SECONDS`); timeout/save failures remain in diagnostics.
`partial-result.json` identifies incomplete outputs and whether they were saved
on stop or recovered from an earlier stage. Native/viewer files are downloaded
before the GPU lease is released and appear in run details. Exact optimizer/RNG
resume is not guaranteed. This hook applies to newly launched workers; already
running workers must finish or stop using their original cancellation behavior.
No completed result is published after a failed or cancelled pipeline.

Restart the dashboard and queue manager after updating code. Integration tests
exercise geometry export and orchestration without CUDA. The provisioned LRZ
A100 80GB environment passed CUDA/import/KNN preflight with PyTorch 2.0.1+cu118;
run `run_20261009_004858` advanced through MASt3R into official chart alignment.
A completed reconstruction is still required to assess end-to-end output quality.

### Optional source-splat depth extension

Set `splatfix.g4splat_runtime.source_depth: true` to render input-view camera-z
maps locally using the same `CpuSplatRenderer.render_depth()` used by episodes.
The default is `false`. This is recorded as `source-splat-depth-extension`, even
when `mode=baseline` selects unedited RGB; it is **not** an authors-only baseline.
Both local and LRZ staging send lossless float32 NPZ maps alongside RGBs.

Each map records its pose/intrinsics, checksum, validity and a binary reliability
mask. The existing renderer accepts front-layer coverage of at least 0.15;
the extension additionally erodes validity by one pixel and rejects local depth
spreads above 10%. Confidence is this binary reliability indicator, not a learned
probability. Source splats can still contain floaters or wrong geometry: the
extension transfers their trusted depths rather than guaranteeing ground truth.

The runtime verifies source maps against the saved cameras, applies the same
actual center crop as RGB (832×480 → 832×468 removes six rows at each edge),
and uses nearest-neighbor sampling rather than blending depth across surfaces.
Trusted depths replace MASt3R point-map samples before the mandatory reprojection
gate and replace input-view depth prediction. Uncovered pixels retain predicted
depth, which must be finite and positive where used. Trusted metric depth and
world points are restored after chart optimization, with the chart scale factor,
and after input-view plane refinement. Novel See3D views retain upstream
inference. The original splat is neither overwritten nor uploaded to the worker.

`inputs.json` records depth coverage and provenance; `source-depth-check.json`
records trusted coverage on the point-map grid; `source-depth-extension.json`
records hashes of the additional run-local upstream patches. Runs fail closed on
mismatched cameras/checksums, no surviving trusted coverage, or invalid fallback
geometry. CPU contract tests cover masks, exact cropping, pixel reprojection,
metric scaling, and nonmutation of the upstream checkout. End-to-end CUDA
training quality still requires a new reconstruction and native-render review.
