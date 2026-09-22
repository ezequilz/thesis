# ArtiFixer repair baseline

The extended scene-run path uses one fixed recipe:

1. The existing explorer selects an artifact and edits its rendered anchor.
2. Render a calibrated local translated trajectory from the current scene,
   retaining its RGB and actual opacity as ArtiFixer conditions.
3. Run the pretrained ArtiFixer once over that trajectory. The edited anchor
   is a generation reference only.
4. Jointly optimize Gaussian positions, scales, rotations, opacity and RGB
   against all generated frames, sampled equally with a shared Adam optimizer.
5. Publish the completed candidate atomically. Cancellation or failure leaves
   the incumbent untouched.

There is no appearance/structure decision. Legacy proposals containing that
field cannot freeze geometry. The edited anchor is not appended as a separate
training target: that previously introduced a conflicting image at the same
camera as the generated trajectory endpoints.

## Defaults and diagnostics

Defaults are 25 frames, trajectory radius 4% of central median depth, four
inference steps and 1,000 total fitting updates. The fitting loss is
`0.8 L1 + 0.2 (1 - SSIM)`. Existing explicit iteration settings are respected;
start a new run to get the new default. The iteration budget is a local
continuation choice, not a published ArtiFixer hyperparameter.

Image editing defaults to 1920×1440. Propagation and fitting use the calibrated
exploration camera resolution (normally 640×480); the reference is downsampled.
Increasing image-edit resolution alone does not increase fitting detail.

Each request retains the reference, input renders, opacity, camera bundle,
generated targets, and fitting metrics. Metrics identify
`artifixer-generated-views-v1`, the trainable parameters, fitting resolution,
loss, per-view before/after L1 and update counts. These measure agreement with
synthetic targets, not scene correctness. There is no automatic quality gate.

## Relationship to the original method

The [paper's 3D distillation section](https://arxiv.org/html/2603.00492v2#S4.SS2)
generates the desired views before standard reconstruction. We follow that
ordering within each repair bundle and keep its pretrained generator and
RGB/opacity/camera conditioning.

This remains an **asset-only adaptation**, not a reproduction of the released
ArtiFixer3D benchmark. The [official reconstruction entry point](https://github.com/nv-tlabs/ArtiFixer/blob/main/data_processing/run_artifixer3d.py)
defaults to a fresh 30,000-step 3DGRUT reconstruction from prepared scene data.
Here we continue an existing scene using gsplat, a short local trajectory,
RGB/DC appearance and a synthetic edited reference. Original support photographs
and higher-order spherical harmonics are unavailable in this scene interface.
No densification, pruning, confidence weighting, broader view-selection agent,
or generator retraining is added. Fixed topology limits missing-surface repair.

## Validation

Run `python -m pytest tests/test_scene_runs_ext.py tests/test_scene_run_gpu_worker.py
tests/test_scene_run_runner.py tests/test_scene_run_studio.py` in the project
environment. The optimizer wiring test requires PyTorch; transaction tests do
not require CUDA or model downloads.

For visual validation, start a new extended run on the same door view. Compare
the saved generated target with native renders at the anchor and translated
nearby cameras, using identical cameras for original and repaired scenes.
Check both target transfer and unaffected surfaces. A lower fitting loss alone
does not establish improvement; this baseline still needs a CUDA scene run.

This baseline supersedes the experimental intervention routing in the earlier
research plans for the active extended implementation.
