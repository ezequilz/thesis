# G4Splat reconstruction in SplatFix

Step 4 offers **G4Splat** for completed saved selections with at least two
camera positions. Choose original RGBs or saved GPT-image repairs, then Run now
or Schedule. ArtiFixer remains the default for old jobs and benchmarks.

The adapter in `src/splat_explorer/splatfix/g4splat/` runs the official
[DaLi-Jack/G4Splat](https://github.com/DaLi-Jack/G4Splat) checkout at
`ec0736126707a42bb2c26ed8ba2c314909edc7a9`. It invokes the unchanged `train.py`
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
fabricated feature tracks or initial points. The upstream `posed.yaml` fixes
focal length, principal point, rotation and translation, while retaining the
upstream internal image scaling and camera-location alignment.

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
builds the official rasterizer, KNN, tetrahedralization and MASt3R extensions,
and downloads DepthAnythingV2, MASt3R, SAM and the official See3D sparse-view
weights. It does not replace any depth estimates, losses, training stages or
model code. The See3D model snapshot is pinned to
`cbf13b6f813137134907408e40d3f2a17d6f0a80` in `bruiiii/See3D`.

Installation requires Linux x86_64, access to the public package/model hosts,
and an NVIDIA GPU. The authors report A100 80GB testing. Downloads and build
products persist on the mounted workspace across allocations. Installation
uses its own environment; application and ArtiFixer packages are not changed.
The first run can take substantially longer while dependencies and weights
are installed. STOP and the job deadline also apply during automatic setup.

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
artifact download use the existing job contract. Cancellation terminates the
training process group and retains files already written by upstream; it does
**not** promise an iteration-boundary checkpoint like the ArtiFixer adapter.
No completed result is published after a failed or cancelled pipeline.

Restart the dashboard and queue manager after updating code. Integration tests
exercise geometry export and orchestration without CUDA. A successful GPU run
is still required to establish runtime compatibility and reconstruction quality.
