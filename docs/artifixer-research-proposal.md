# Building on ArtiFixer: evidence-separated scene refinement

Research proposal · 22 September 2026 · no experimental results claimed

> Updated strategy: [Agent-guided hybrid research plan](agent-guided-hybrid-research-plan.md) includes the image editor and existing VLM from the first hybrid experiment. Its staged comparisons supersede the ordering below.

## Decision and updated setting

Make original photographs and calibrated poses the main setting now. Give the same permitted input photographs, camera information and initial reconstruction to every compared method. Keep exported-assets-only repair as an ablation, not the main justification for the method.

The first contribution to test is **evidence-separated distillation of a frozen ArtiFixer into a native Gaussian scene**: distinguish what real photographs and generated targets are allowed to change in geometry, base appearance and directional appearance. Keep the existing VLM controller and include image editing from the first hybrid experiment. Replay agent decisions and cache generated images to isolate the distillation mechanism.

The hypothesis is that part of the remaining quality loss occurs when fitting generated images into 3D, rather than in image generation itself. We can potentially improve this transfer without retraining a large generator. This is a defensible research hypothesis, not an established optimum or a verified novelty claim.

The intended result is better native novel-view quality and fewer regressions in already good regions, at a measured inference and reconstruction budget. A prettier generated video alone does not establish success.

## What the relevant methods already contribute

**ArtiFixer:** opacity-dependent latent mixing balances preservation and completion; camera and reference conditioning support scene-specific generation; causal distillation makes long trajectories practical. Its native-scene variant already distills generated views into 3D, so the generation–reconstruction loop is not our contribution. The paper also identifies fine-detail/text blur and color shifts as limitations. We reuse its pretrained generator and compare against its reconstruction endpoint. [Paper](https://arxiv.org/html/2603.00492v2), [official implementation](https://github.com/nv-tlabs/ArtiFixer).

**ArtifactWorld:** its important lesson is selective intervention. Artifact prediction, artifact-aware fusion and boundary anchoring guide restoration; its data construction covers multiple degradation types. A heatmap estimates where corruption appears, but does not by itself determine which Gaussian parameters should absorb a correction. We should test localization as an inexpensive baseline before claiming that our more detailed update rule is needed. Reusing its predictor would require checking the released interface and profiling its inference cost; it is not assumed to be a free plug-in. [Paper](https://arxiv.org/html/2604.12251v1).

**FixAnything:** representation-independent video cleanup and camera-recovery-based preference training target geometric consistency. The useful lesson is to judge whether outputs support reconstruction, not just image quality. However, recovering a camera trajectory does not uniquely establish the correct local surface or reflectance. We retain known calibrated poses and inspect local evidence. We do not claim geometry scoring, seed selection or preference alignment as new ideas. [Paper](https://arxiv.org/html/2608.23549v1).

**ConFixGS is an especially important closer comparison.** It already uses support-image reprojection to score generated targets and confidence-weighted refinement/densification. “Use photographs to filter hallucinations” is therefore insufficient novelty. Our candidate distinction is parameter-specific treatment of evidence and angular appearance, rather than one confidence scalar multiplying every parameter gradient. Its use of initial-scene depth also motivates testing sensitivity to a wrong geometric scaffold. [Paper](https://arxiv.org/html/2605.09688).

Other boundaries matter: Carve3D already measures reconstruction consistency; VistaDream already couples diffusion sampling and Gaussian reconstruction; GSFixer already uses geometric and semantic reference features. VLM planning, multiview generation, foundation geometry, masks and a consistency score are individually established ingredients. [Carve3D](https://arxiv.org/html/2312.13980), [VistaDream](https://arxiv.org/html/2410.16892), [GSFixer](https://arxiv.org/html/2508.09667v1).

## The technical reason to separate evidence

A rendered error can be reduced by moving a Gaussian, changing its scale or opacity, changing base color, or changing directional color. These explanations are coupled. A pixel confidence value says how much to trust a target; it does not say which explanation to trust.

For one Gaussian, directional color has the form `c(d) = sum_k a_k Y_k(d)`. Degree-three SH has 16 coefficients per channel. One camera direction constrains a combination of them; many nearly identical directions remain poorly informative. Full rendered identifiability is harder because visibility, opacity and overlapping Gaussians are also unknown. [Original 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

Consider a glossy cabinet. An inconsistent generated highlight can be fitted by distorted geometry or by spurious directional color. Conversely, forcing a real moving highlight to become view-independent can also distort geometry. Thus neither unrestricted SH nor globally disabling SH solves the problem.

A related issue is resolution: photographs can contain genuine text and texture that a generated frame smooths or changes. Giving both targets equal authority can discard recorded detail. The proposal separates three questions: where a change is needed, what evidence supports it, and which parameter family may change.

## Proposed first implementation

### 1. Establish a faithful, shared baseline

Preserve all source SH coefficients and verify renderer/camera parity. The current project drops higher-order SH during loading and uses a single-view repair objective; fixing those defects is infrastructure work, not a research gain over ArtiFixer.

First optimize with real photographs alone using the same total fitting budget. Otherwise, an apparent AI improvement could simply come from giving the scene more optimization or restoring missing source data.

For benchmark scenes, construct the starting scene exclusively from the allowed support images. An exported splat originally trained on the evaluation images leaks information even if those images are hidden from the repair code.

### 2. Build evidence from photographs

Use calibrated cameras as fixed reference coordinates. Extract cross-image matches, verify them geometrically, and triangulate where baselines and visibility permit. A frozen matching/geometry foundation model is useful for proposing correspondences or depth, but its confidence is not a probability of correctness. Align inferred geometry to calibrated coordinates and reject inconsistent estimates. [MASt3R](https://github.com/naver/mast3r), [VGGT](https://github.com/facebookresearch/vggt).

Maintain separate flags for observed structural support, directional coverage, suspected non-Lambertian appearance, and unobserved content. Missing matches are uncertainty, not proof of free space. Thin structures, transparency, moving objects and occlusion boundaries need conservative handling.

The VLM identifies semantic regions and repair intent—for example, preserve the cabinet and its lettering while fixing its edge. Numerical geometry determines camera support and parameter permissions. Semantic plausibility cannot certify that an invented object is present.

### 3. Generate once, cache, and compare fitters

Start with the official pretrained ArtiFixer input contract and unchanged photographic references. Render a fixed camera bundle from the same initial scene for every fitter. Cache outputs so the first experiment changes only distillation, not prompts, cameras or random seeds.

Use two intersecting translated paths where practical, retaining common coverage and testing off-path viewpoints. Adjacent video frames are correlated; count independent spatial/angular coverage rather than raw frame count. Do not assume more generated frames add more independent evidence.

### 4. Route optimization signals by evidence and parameter family

Use a joint objective with separate real-image, generated-image and geometric-evidence terms. The important addition is separate gradient permissions, not merely another scalar loss weight:

- **Real images:** supervise native-resolution appearance and supported geometry. Permit directional appearance updates, regularized where angular support is weak. Calibrate exposure on support images only.
- **Generated targets, base appearance:** permit masked updates with scale-dependent confidence. In well-observed areas, preserve high-frequency detail from photographs; in unsupported areas, allow generated detail but retain its synthetic provenance.
- **Generated targets, geometry/opacity:** permit updates only where independent cross-view tracks, silhouettes or other structural support agree. A practical first version uses conservative region masks and displacement limits. Do not let uncertain highlights drive geometry merely because higher SH is frozen.
- **Generated targets, higher-order SH:** first test blocking these gradients while real-image gradients remain enabled. This is an ablation, not a universal rule. A later soft policy can admit directional updates when angular support and validation justify them.
- **New geometry:** require multiview support for insertion and initialize appearance conservatively. In genuinely unobserved areas, geometry is a generative hypothesis; consistency can rank hypotheses but cannot certify their truth.

Synthetic losses can still change directional behavior indirectly through visibility and geometry. Therefore, gradient routing must be paired with real-view rendering checks; it is not a hard preservation guarantee.

A useful directional-support diagnostic is `H_g = sum_v w_gv y(d_gv)y(d_gv)^T`, computed from real views with visibility weights. Its spectrum measures the angular conditioning of a local SH fit with geometry held fixed. It is not a posterior covariance for the entire scene. Start with simple coverage bins; only add this diagnostic if it improves decisions beyond cheaper counts and masks.

Use trust-region updates and per-region real-image degradation limits relative to the incumbent. A global average can conceal the destruction of a small salient object. Source renders are supplementary guards; original photographs take precedence where the source reconstruction is wrong.

### 5. Refine one scene and validate the native result

Default to updating an isolated copy in the existing coordinate frame. Keep a reliable overlap shell around the region. For a topology failure, remove/replace only the implicated local primitives, then optimize jointly with neighboring geometry and real images. Do not concatenate independent splat scenes: duplicate overlapping density changes compositing.

Accept or reject the candidate using permitted input evidence and development-calibrated rules. Publish the candidate atomically only after checking regional and global preservation. Keep final test photographs completely outside references, matching, optimization, candidate selection, stopping and VLM inspection.

With only two or three support views, do not silently borrow benchmark test images for acceptance. A separate per-scene validation set changes the input protocol and must be supplied to all baselines. Cross-validation that claims unseen prediction must also exclude that image from the initial reconstruction and generator conditioning; merely withholding it from the last fitting stage is insufficient.

## The agent remains the controller

The agent should choose a region, an intervention and a camera bundle—not just find the ugliest screenshot. Rank regions by defect severity, available real evidence, salience and estimated repair cost. Select translated cameras that reveal occlusions or increase angular diversity.

For example, a narrow front-facing trajectory cannot distinguish a damaged chair leg from an appearance artifact. A side translation may make alternative explanations diverge. Render and generate there only if it is likely to change the repair decision. This is active synthesis; it does not acquire a new physical observation.

Compare this planner against fixed paths and simple coverage heuristics at equal generated-frame and GPU-time budgets. Foundation-model orchestration is useful only if this comparison shows value. VLM-based aesthetics should not be the main acceptance metric.

## A second, more speculative experiment

**Temporarily remove contradicted structure before generation.** High opacity indicates occupied rendered content, not correctness. If real-image evidence identifies an opaque floater or duplicated surface, form an isolated alternative scaffold with those primitives suppressed and re-render RGB and opacity consistently. Compare generation conditioned on the original scaffold against generation conditioned on this alternative.

This tests whether a wrong scaffold prevents the generator from producing a useful correction. It exploits the generator's existing conditioning interface without retraining. It is not valid to assert that high opacity makes correction impossible, and changing the input distribution may make results worse. If removal exposes an opaque background, the new opacity may remain high; the intervention is still a changed geometric scaffold, not necessarily a hole.

Always include removal plus ordinary real-image refinement without a generator. That cheaper arm may solve the problem. Do not manipulate alpha alone while leaving incompatible RGB/rays. Keep the incumbent untouched until a complete candidate passes evaluation.

Removal/inpainting and prior removal are existing ideas, including IMFine and RoMaP. Any contribution would have to be the evidence-driven choice of scaffold intervention for pretrained restoration, not “delete and inpaint.” This is higher-risk follow-up work, not a second mandatory module in the first experiment. [IMFine](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_IMFine_3D_Inpainting_via_Geometry-guided_Multi-view_Refinement_CVPR_2025_paper.pdf), [RoMaP](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Robust_3D-Masked_Part-level_Editing_in_3D_Gaussian_Splatting_with_Regularized_ICCV_2025_paper.pdf).

## Experiments that can disprove the proposal

Begin on a small development collection covering diffuse surfaces, glossy surfaces, thin geometry, readable details and incomplete regions. Use multiple scenes per category before drawing conclusions. Keep a separate evaluation collection and untouched test views, preferably including a second physical camera trajectory; Nerfbusters motivates this harder evaluation. [Dataset and protocol](https://arxiv.org/abs/2304.10532).

Run these arms in order:

1. Initial scene and real-image-only continued optimization.
2. Official ArtiFixer3D reproduction, retaining its documented reconstruction protocol.
3. ArtiFixer outputs with a common fitter, original-image anchors and scalar confidence weighting.
4. The same cached outputs and fitter with evidence-separated gradient routing.
5. Compare replayed agent proposals against live adaptive selection at the same generation budget; the agent remains in the operating pipeline throughout.

Arm 3 versus 4 is the clean causal comparison. Arm 2 establishes the end-to-end reference; substituting our fitter into ArtiFixer does not reproduce the official method. Also compare simple globally frozen higher SH and uniform multiscale losses: if those explain the gain, report the simpler mechanism.

Measure native-render PSNR, SSIM and LPIPS on untouched images, per-region detail preservation, independent geometry where available, regression rate on good regions, and stability along translated paths. Report scene-level variability, not thousands of correlated frames as independent samples. Generated-target fitting loss is diagnostic only. Include runtime, peak memory, generated frames, candidate retries and Gaussian count; report both matched-budget results and quality–cost curves.

Use a fixed evaluation mask/region set defined independently of method outputs. Include rejected repairs and no-ops in scene-level reporting; a method cannot win by accepting only easy patches. Evaluate several repair rounds to expose accumulated drift, but do not use the final test set to decide when to stop.

**Go:** arm 4 improves native quality or preservation at comparable cost across held-out scenes, and parameter/frequency ablations explain the improvement.

**Revise:** gains occur only on synthetic corruption, only at training cameras, or only because real references/resolution/compute differ. If generated errors are mostly wrong content rather than harmful assimilation, prioritize scaffold intervention or reference selection instead.

**Stop adding complexity:** if scalar confidence or real-only refinement matches the proposed routing, retain that simpler solution. If a geometry foundation model adds no benefit over calibrated feature matching, remove it.

## Role of an additional image foundation model

Include image editing from the first hybrid experiment: agent-selected rendered anchor → image edit → ArtiFixer propagation → native reconstruction. Compare with no-editor and no-video arms. Original photographs remain unchanged observations; edited anchors are separately tagged hypotheses. Also test final image postprocessing as a distinct output endpoint.

The [incremental hybrid research plan](agent-guided-hybrid-research-plan.md) supersedes this document's earlier experimental ordering. It keeps the VLM agent central and tests selective restoration, geometric validation, confidence weighting, parameter-specific fitting and speculative scaffold interventions incrementally.

## Scope of the claim

The proposed thesis is: **with a fixed pretrained restorer, the route by which synthetic supervision updates a radiance field matters; evidence-specific geometry and appearance updates can improve native scene reconstruction while preserving photographed content.**

The inspected literature establishes substantial overlap with the components. It does not establish that this precise method is new or that it will beat ArtiFixer. The first experiment should determine whether the proposed failure mechanism is real before adding unnecessary controller complexity or a new training program.
