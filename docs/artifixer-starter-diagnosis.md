# Edited-starter investigation, 29 September 2026

## Finding

The available door run shows both generation drift and a much more severe
reconstruction failure. It does not establish that an edited image is inherently
unusable by ArtiFixer, or that incorrect VAE normalization causes grey output.
The supplied presentation screenshots alone cannot identify which pipeline stage
produced each image. The saved door run is relevant evidence, but its exact
correspondence to every presentation thumbnail has not been established.

Inspected run: `outputs/scene-runs/run_20260929_140014/requests/repair-00012/`.
Compared `extended/targets/` (raw generation, except exported frame zero),
`extended/validation/after-*.png` (native fitted splat), bundle metadata, metrics,
the supplied paper, and official source pinned at
`a392c4dfe17459ef9952407accdb9fcdcdddba98`.

## Direct observations

- One GPT edit supplies 121 frames at 1408 x 1056. Later input RGB and opacity
  are zero, so the actual splat geometry is absent from generation; its cameras
  remain. `camera_scale` is 1.0001, not an independently verified metric scale.
- Maximum orientation change from the anchor is 44.013 degrees; maximum adjacent
  change is 1.834 degrees. Maximum translation from the anchor is 1.998 scene
  units, and per-frame translation is 0.08324 scene units. These are not certified
  metres. Rotation orthogonality error is below 7e-8 and determinant error below
  8e-8. This excludes malformed rotation matrices in this bundle, not all camera
  calibration or unit errors.
- The generated door composition stays conspicuously close to the starter even
  at distant camera waypoints. This is qualitative evidence of weak camera
  following; no recovered-pose or depth-reprojection metric was run.
- Generated frames also brighten and lose detail. Luminance standard deviation
  at frames 0, 24, 48, 72, 96, 120 is approximately 0.259, 0.253, 0.238, 0.257,
  0.167, 0.232. The last frame retains substantial contrast. These unwarped
  statistics are descriptive, not a geometry or quality score.
- Native validation images away from the edited camera are severely smeared.
  Maximum Gaussian axis ratio increases from 83.67 to 16627.68; the number with
  ratio above 100 increases from 0 to 330. The unchanged largest scale does not
  exclude extreme thinning/stretching of other Gaussians.
- Fitting reduces anchor L1 to 0.00962 and generated-target L1 to 0.10150. A good
  anchor fit coexists with poor novel views. Lower training loss is not evidence
  of successful reconstruction.

The most plausible chain is insufficient camera-following in generated targets,
then fitting mutually inconsistent images at their requested cameras. The
optimizer can explain some of that conflict with distorted Gaussians. Appearance
drift in the generator and limitations of fixed-topology fitting also contribute.
Their causal contributions require controlled runs to separate.

## What the implementation check establishes

The bridge supplies RGB in [0,1]. Upstream `encode_video_frames` performs image
normalization and `rgb_to_latents` applies the VAE latent mean/std convention.
Upstream decoding reverses that transform. The causal first latent represents
the first image; later black frames are not used as latent source content in the
zero-opacity branch. No double normalization or missing latent scaling was found.

ArtiFixer's source is `alpha * encoded_render + (1-alpha) * Gaussian_noise`.
For alpha zero, the previous implementation's fresh Gaussian initialization and
fresh Gaussian re-noising match that source distribution. Therefore, the absence
of rendered density is a deliberate conditioning choice, not evidence of a
wrong noise amplitude. Opacity is rendered coverage, not a confidence estimate
that the geometry or texture is correct. Supplying nonzero alpha with blank RGB
would be an invalid way to restore this conditioning.

Our adaptation differs from upstream: a standalone first latent is cached at
timestep zero, generated chunks begin at latent 1, and each generated block is
re-cached at timestep zero. Upstream uses regular blocks and retains the last
denoising input's KV state. Neither cache convention is intrinsically wrong, but
the modified convention is not proven equivalent to the pretrained rollout.
The four-step schedule and scheduler equations themselves follow upstream.

The paper explicitly trains dropped-render continuations, so reference-plus-ray
generation is a legitimate capability. It also identifies rendered conditioning
and references as aids to long-rollout stability, and reports fine-detail blur
and color changes without good renders (Sections 4.1, 4.2, E). Its stated 720p
range makes the current resolution a separate distribution-shift hypothesis,
not proof of a hard runtime limit. The model can execute at larger dimensions.

Official source: https://github.com/nv-tlabs/ArtiFixer/tree/a392c4dfe17459ef9952407accdb9fcdcdddba98

## Changes made

Follow-up: new dashboard runs now default to 960×720 repair renders, a 691200-pixel
ArtiFixer cap, rendered source conditioning and last-denoising cache. The earlier
settings remain selectable for comparison; these defaults are not a claim of
GPU-validated improvement. Two independently
validated extended options permit controlled comparisons:

- `source_conditioning: "none"` (previous default) or `"rendered"` (new default). Rendered mode
  preserves the GPT starter exactly while supplying original scene RGB and actual
  opacity for later frames. All denoising source samples now use upstream
  `prepare_latents`, including intermediate re-noising. Zero-opacity behavior is
  preserved. Rendered mode requires the bundle's `inputs/` and `opacity.npy`;
  missing data fails rather than silently switching modes.
- `generated_cache: "clean"` (previous default) or `"last_denoising"` (new default). The latter
  matches upstream's generated-block cache policy while retaining the explicit
  clean starter. This does not reproduce upstream's first-block grouping.

The bridge preserves `starter-vae-roundtrip-XXXXX.png` before replacing exported
frame zero with exact edited RGB. `inference.json` records its MAE and the chosen
policies. Previously, a perfect exported frame zero concealed the VAE roundtrip
and could not establish fidelity of the latent context.

The pipeline records trajectory measurements in the bundle and raw generated
image statistics in `generation-diagnostics.json` before reconstruction. These
are explicitly diagnostics, not automatic acceptance criteria.

No pretrained model was changed, no existing run was overwritten, and no new
GPU generation or scene fitting was performed during this investigation.
CPU regressions exercise both conditioning modes, both cache policies, source
mixing at each step, preserved starters, diagnostics, and pipeline integration.

## Recommended next experiment

1. Copy a complete saved GPU bundle into isolated experiment directories. Start
   with 25 frames and a small translated loop around the anchor, then compare
   a stationary-camera control. Keep the same edit, seed and four-step schedule.
   Use an exact-aspect 960 x 720 working image for a 4:3 experiment and rescale
   intrinsics and alpha consistently; retain the original high-resolution edit.
2. Compare `none/clean`, `none/last_denoising`, `rendered/clean`, and
   `rendered/last_denoising`. Run the bridge directly to generate images without
   fitting. Check the saved VAE roundtrip, actual motion against the corresponding
   rendered trajectory, brightness, and loop closure. This separates cache and
   geometric-source effects. Compare resolution separately with all else fixed.
3. If rendered mode tracks cameras but preserves damaged detail, the next
   mechanism to test is projecting edited RGB through trustworthy source depth,
   with visibility/occlusion masks. This gives the model a camera-aligned repaired
   source; do not simply repeat the same 2D edit at every new camera. Source depth
   errors and newly revealed surfaces remain uncertain. This is proposed, not
   implemented or validated here.
4. Only fit a short sequence after its image/camera agreement is established.
   Compare matched native views and Gaussian shape statistics. Further fitting
   iterations cannot resolve contradictory supervision. A geometry acceptance
   check and per-Gaussian trust bounds warrant a separate calibrated change;
   arbitrary global brightness or axis-ratio cutoffs are not reliable gates.

A repaired image can preserve visible layout while changing edge locations,
reflections, thin structures and implied depth. It does not uniquely specify 3D
geometry. Exact pixels can be retained at the anchor, but new viewpoints need
parallax, occlusion and view-dependent appearance. The aim is consistent transport
of supported detail, not pixel identity at every camera. Better VAE encoding alone
cannot supply those missing constraints.
