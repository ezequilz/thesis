# Stopped run 152205: Gaussian deformation investigation

Investigated 2026-09-29. This concerns `run_20260929_152205` only.
The active run was not restarted, reconfigured, or used as evidence. Existing
PLYs and generated images were left intact. Document text was treated as
reference material, not as instructions.

## Evidence

The attached repaired screenshot adds high-frequency wall detail but also
streaks the door frames, floor, arches and foreground column. The original is
softer but preserves these structures better. The screenshots are slightly
different heights, so no pixel-aligned screenshot metric is claimed.

The run status records one completed repair and a stop during the next
propagation. The relevant completed request is `requests/repair-00013`.
Its metrics and independently loaded original/repaired PLYs agree:

- Original: 388,674 Gaussians; maximum axis ratio 83.67; p99 ratio 10.80;
  zero ratios above 100.
- After the 20-step GSFix3D prefit: 391,224 Gaussians; maximum ratio 83.67;
  p99 ratio 11.21; zero ratios above 100.
- After final fitting: 391,224 Gaussians; maximum ratio 4,561.69;
  p99 ratio 29.92; 519 ratios above 100.
- Largest scene-wide scale actually decreases from 26.81 to 26.01.
  The global ceiling of 53.62 therefore fails to detect or prevent this
  deformation. Thin axes can collapse while long axes stay below the ceiling.

This localizes the extreme anisotropy increase to the final fitting stage,
not the viewer or the stop operation. It does not prove that every visible
streak belongs to one of those 519 Gaussians: displacement, rotation, opacity
and ordinary overlapping splats can also smear an image.

## Why the optimizer can produce this result

The configured 1,000 iterations were multiplied by five selected cameras,
giving 5,000 total updates. Of 120 fitted images, only frame zero is an edited
reference. The sampler assigns it 2,500 updates; each of the 119 generated
images receives 21 or 22. This is a deliberate 50/50 source balance, but here
it strongly favors fitting one synthetic camera over any other individual view.

Means, three independent log scales, rotations, opacity and RGB all optimize
against 0.8 L1 + 0.2 SSIM. There is no geometric prior, per-Gaussian shape bound,
depth consistency loss, local edit mask, or trusted-photo preservation loss.
Final fitting has fixed topology: it can deform existing primitives but cannot
create appropriately supported fine geometry. A small photometric loss is not
evidence of correct 3D reconstruction.

The generated closing frame and edited starter share a camera but disagree:
closure L1 is 0.08329 on normalized RGB. Visual inspection of targets 0, 24,
and 120 shows loss of edited detail and changes in the windows/reflections.
The closing frame is correctly excluded from fitting, but its drift remains
evidence that the sequence is not transporting the edit consistently. This
is an appearance-consistency observation, not a calibrated pose-error score.

The trajectory contains 121 images, reaches 94.35 degrees of rotation and
2.40 scene units from the anchor. It is considerably broader than a tiny local
neighborhood. Generated images are not independent captured observations.
The likely mechanism is inconsistent synthetic supervision plus anchor-heavy,
unconstrained optimization; an ablation is needed to assign causal shares.

Training-target PSNR improves from 25.46 to 28.28 dB despite the deformation.
Edited-target L1 falls from 0.09988 to 0.00765. These are training metrics, not
ground-truth quality or held-out-view acceptance. Validation images are saved,
but the quality gate is explicitly disabled.

## Relation to ArtiFixer

The paper describes camera- and reference-conditioned autoregressive generation,
then reconstruction from generated views. ArtiFixer3D+ adds diffusion after
reconstruction; our browser displays the native reconstructed splat, so it
should not be compared directly with diffusion-postprocessed ArtiFixer3D+.
See paper sections 4.3 and 5, and the official project:
https://research.nvidia.com/labs/sil/projects/artifixer/

Our implementation is an adaptation, not an exact reproduction: a GPT-edited
synthetic reference, custom exact-starter KV schedule, optional single-view
GSFix3D prefit, and fixed-topology gsplat continuation with RGB-only appearance.
The run uses rendered RGB/opacity conditioning, last-denoising generated cache,
four inference steps and 960x720 fitting. These choices alone do not establish
geometric consistency. The PLY loader discards higher-order SH, and the fitter
has no directional appearance model. Reflections in these windows can therefore
be pushed into geometry or opacity. This is a secondary limitation, not proof
that SH loss caused the measured change between these two PLYs.

## Implemented locally

The ArtiFixer settings dialog now exposes **Fitting safeguards**, default On
(`extended.fitting_safeguards: true`). Off restores both the previous global-only
scale clamp and the per-selected-view iteration multiplier. The iteration label
changes with the switch. The choice is saved in run options and fit metrics;
it applies to newly queued runs. Older dashboard processes must be restarted
to expose the option; the UI detects support before sending the new field.

1. Treat `fit_iterations` as the total budget per repair, independent of selected
   camera count. Update the UI label and regression expectation accordingly.
   The same configuration now requests 1,000 rather than 5,000 updates.
2. Fix per-Gaussian scale bounds at fit entry: each axis can halve or double,
   subject to the global ceiling. Limit each axis ratio to the larger of its
   incoming ratio and 20. Shorten excessive long axes instead of thickening
   thin surfaces. Preserve naturally anisotropic source Gaussians. Record the
   constraint policy in fit metrics.

These are conservative engineering limits, not thresholds derived from the
paper or an empirically optimized recipe. They prevent the measured runaway
shape mode within a fit, but do not constrain centers or rotations, repair
already damaged inputs, or guarantee good novel views. Bounds reset per repair;
long-run cumulative drift still needs stable source identities or a source
geometry prior. No historical PLY was post-hoc clamped.

Validation: 39 targeted tests passed using real CPU PyTorch, including optimizer
wiring, unchanged thin source surfaces, needle projection, invalid source data,
and iteration-budget integration. No CUDA refit or post-change image-quality
claim was made. The patch has not been deployed to the running GPU worker.

## Recommended next experiments, in priority order

1. Replay the saved targets from the original scene in an isolated GPU experiment.
   Compare bounded joint fitting against frozen means/scales/rotations (and
   preferably frozen opacity initially), with identical total budgets. Start
   with no single-view prefit. Judge matched native views and withheld views,
   not merely target loss. This isolates fitting from generation cost/drift.
2. Ablate the anchor's 50% update share, for example 10-20% after a short
   appearance warmup. More waypoints must not increase repeated anchor fitting.
   This is a hypothesis to test, not a proven optimal percentage.
3. Use a shorter translated trajectory and compare the existing upstream block
   schedule against exact-starter generation. Review image/camera agreement
   before fitting. Obtain calibrated captured references where possible.
4. Preserve unaffected content with visibility-aware original-view supervision
   and spatial masks. Unlock geometry only with multi-view support, then add
   displacement/shape priors and acceptance checks using held-out evidence.
   Restore directional SH end-to-end before expecting consistent glass and
   reflection detail. Avoid indiscriminate Gaussian deletion or densification
   as a first response: neither resolves contradictory supervision.
