# Agent-guided hybrid 3DGS repair: incremental research plan

22 September 2026 · Proposed experiments, not measured improvements

## Updated research commitment

Keep the existing VLM exploration loop as the orchestrator. Include GPT image editing through CliRelay in the first hybrid experiment, with a matched no-editor arm. Reuse pretrained ArtiFixer for multiview propagation. Investigate selective restoration, geometric checks, evidence weighting and parameter-specific distillation as independently switchable additions. Test temporary suppression of incorrect splats early on an isolated branch.

This supersedes the earlier recommendation to postpone the image editor and introduce agent planning only after the fitter. The agent remains present from the start; replaying its decisions is an experimental control, not a replacement architecture.

The candidate contribution is **an agent-guided choice of repair intervention, followed by evidence-specific assimilation of image and video priors into a native radiance field**. Novelty and quality remain hypotheses. Combining named models alone is not sufficient; the experiments must explain which decisions and interactions produce a benefit.

Main setting: original support photographs, calibrated poses and an initial scene built only from those support photographs. All compared methods receive the same permitted evidence. Asset-only operation remains a secondary ablation.

## Intended loop

1. The existing VLM explores the accepted scene and selects a region and novel anchor view.
2. A numerical planner adds overlapping translated views, reference photographs and boundary views.
3. The agent chooses a repair action: appearance edit, structural repair, temporary suppression and completion, or defer.
4. The image model edits the chosen rendered anchor, using relevant photographic context and a bounded repair instruction.
5. A frozen video restoration model propagates the candidate through calibrated views.
6. Selective masks and geometric checks identify usable evidence; real photographs remain separately tagged.
7. A common fitter builds an isolated candidate with configurable geometry, opacity, base-color and SH permissions.
8. Native-render checks accept or reject the candidate. The agent receives the result and continues from the accepted revision.

The normal path initially uses one edited anchor. A second anchor is a later experiment because independently attractive edits can impose incompatible geometry. Original photographs are never overwritten by edited copies.

## What we reuse, and what constitutes an adaptation

- **ArtiFixer:** reuse pretrained weights, camera/opacity/reference conditioning and generated-view distillation. Passing an edited render as an additional reference is our adaptation and must be tested for compatibility. Its 3D+ variant reapplies its autoregressive restorer after native rendering; GPT postprocessing is an analogous experimental endpoint, not that exact method. [Paper](https://arxiv.org/html/2603.00492v2).
- **ArtifactWorld:** adapt selective intervention and preservation of reliable boundaries. First implement external masks, reference selection and masked fitting. Its learned triplet fusion is architecture-specific; reproducing it inside ArtiFixer would require additional implementation and potentially training. Test its released restoration model separately before attempting architectural transfer. [Paper](https://arxiv.org/html/2604.12251v1).
- **FixAnything:** adapt the principle of assessing whether outputs support known camera geometry. First use inference-time checks and candidate ranking. Its Flow-DPO contribution is learned preference optimization, not an inference filter we can transplant. Use the released checkpoint as an alternative video backend when testing that trained prior. [Paper](https://arxiv.org/html/2608.23549v1).
- **ConFixGS:** implement a reference baseline using observation-reprojection confidence for synthetic supervision. Extend it only after measuring its failure cases. Inaccurate source depth can invalidate correspondences; low confidence must distinguish contradiction from absent evidence. [Paper](https://arxiv.org/html/2605.09688).

Do not run every video model sequentially by default. That multiplies compute and makes it hard to attribute changes. ArtiFixer is the initial backend; other checkpoints are controlled substitutions.

## Stage 0 — Make experiments faithful and replayable

Preserve all SH coefficients, verify camera conventions and full-SH renderer parity, load calibrated support images, and separate accepted from candidate GPU/host state. Restore shared image-backend selection: scene-runs currently directly constructs Qwen while the GPT wrapper uses the existing gpt-image-2 constant. Record the exact requested and returned model identity and retain the current CliRelay route.

Add an immutable repair record containing parent scene hash; agent proposal; cameras/intrinsics; masks; source photograph IDs; edited-anchor lineage; generated clip lineage; seeds where supported; inference parameters; fitting configuration; costs; and acceptance measurements. Pin prompts and cache external model outputs. API reproducibility cannot be assumed from a seed alone.

Keep the agent's navigation and exploration behavior. Extend the selected-view action into a bundle request rather than redesigning the harness. Replay the same logged proposal when isolating backend or optimizer changes.

Exit criteria: no-op roundtrip fidelity, oracle multiview fitting succeeds, rejected candidates do not alter the accepted scene, and a recorded experiment can be replayed without regenerating images.

## Stage 1 — Test the image/video combination first

Use the same agent-selected regions and camera bundles for four arms:

- Real photographs plus ordinary continued reconstruction, without generated targets.
- ArtiFixer propagation and reconstruction without an image edit.
- Image-edited anchors fitted directly with real references and the common fitter, without video propagation.
- Image-edited anchor followed by ArtiFixer propagation and the same fitter.

The fourth arm tests the user's central hypothesis: a strong image prior improves fuzzy content, while a video prior makes that improvement usable across views. Compare the fourth arm to both the second and third; neither comparison alone measures the full interaction.

First try an edited anchor through ArtiFixer's existing reference interface, with the known camera pose and unchanged real references. Treat the pose as intended, not automatically preserved: check static landmarks, silhouettes, crop and intrinsics after editing. Keep RGB, opacity and camera conditioning mutually consistent. An anchor-reference slot does not guarantee exact reproduction or propagation of the edit.

If the edit is ignored, measure that before choosing a different adapter. Do not silently replace every degraded input frame or invent matching depth for the edited anchor. Test alternative conditioning placement as a separate arm. Prefer rejecting geometry-incompatible edits over dense warping that conceals the mismatch; any registration is logged and evaluated.

Record two distinct failures: propagation loses the image improvement, or propagation retains it but native reconstruction cannot represent it. These lead to different next steps.

In parallel, apply the image editor as final render postprocessing to selected native outputs. Report those as enhanced-render results with their own latency and temporal-consistency measurements. Do not fold them into native-scene scores. A later video→image→video sequence is justified only if the first video pass consistently loses useful detail.

## Stage 2 — Selective restoration and boundaries

Add region masks to the most promising Stage 1 recipe. Compare whole-view editing against local editing with unchanged boundary context, holding the selected region fixed. Project masks across cameras with occlusion handling and refine them using image evidence; uncertain source depth cannot define a perfect mask.

Use masks independently for image editing, generated target losses and Gaussian update permissions. These are distinct controls. Preserve healthy content using real views and a reliable overlap shell. Avoid naïve per-frame paste-back after structural edits: old occlusions can contradict the new geometry, and seams can become reconstruction targets.

Start with the agent plus ordinary segmentation/geometry checks. Then test whether ArtifactWorld's released predictor or restorer adds enough value to justify its cost. External selective fitting is labelled ArtifactWorld-inspired, not a reproduction of its fusion mechanism.

Exit criterion: a measured reduction in collateral changes at comparable repair strength, not simply smaller edits.

## Stage 3 — Geometric checks and confidence weighting

Implement these as two switches and evaluate all four combinations: neither, geometry checks only, confidence weighting only, both.

Geometry checks use calibrated-pose epipolar/reprojection residuals, track coverage and triangulation support. Require adequate spatial coverage so a tiny set of easy matches cannot certify an entire repair. Include pose recovery where practical, but distinguish globally recoverable cameras from correct local surfaces. Specular and transparent regions need different confidence handling from diffuse surfaces.

Confidence weighting applies to synthetic targets by region/pixel. Real support images remain a separate observed-data term; quality masks may exclude blur, exposure faults or moving content using the same rules in every arm. Do not downweight an inconvenient photograph merely because a generated view disagrees.

Distinguish three cases: supported agreement, supported contradiction, and unobserved/uncertain. The third is not evidence of correctness and is not automatically a reason to ban completion. Keep it as a conservative, explicitly synthetic branch. When the scaffold is poor, test verified real-image tracks or an aligned geometry-model estimate instead of blindly trusting splat depth.

Exit criterion: geometry checks and weights improve native quality or reduce harmful updates beyond an equal-cost extra reconstruction budget. Log false rejections as well as accepted repairs.

## Stage 4 — Evidence-specific geometry and SH updates

Compare scalar confidence with parameter-specific routing using identical cached targets. Split parameters into positions/covariances, opacity, base color and higher-order SH. Treat opacity as a visibility-changing parameter, not merely appearance.

Real photographs supervise all supported parameter groups. Synthetic evidence receives separate geometry and appearance permissions. Compare unrestricted fitting, synthetic higher-SH blocking, conservative geometry limits, and the combined rule. Only then add angular-support-dependent soft permissions.

The mechanism being tested is that reliable appearance correction does not imply reliable geometry or directional reflectance. Blocking synthetic SH alone can force errors into geometry; geometry and appearance restrictions must be evaluated together. Existing full SH is preserved, and real-image fitting can retain legitimate view dependence.

Native high-frequency detail is evaluated separately from sharp generated pixels. A spatial-frequency cutoff is an ablation, not a substitute for structural confidence.

## Stage 5 — Speculative interventions, in parallel on selected defects

Start these branches after Stage 1 produces a working candidate, rather than waiting for all other stages.

**Temporary suppression:** for a floater contradicted by original observations, suppress implicated primitives in a clone, re-render consistent RGB/alpha/depth, then edit and propagate. Compare preserve, partial attenuation and removal. Include suppression plus real-only fitting to establish whether generation adds anything.

**Local reinitialization for mush:** fuzzy geometry is not necessarily empty or entirely wrong. Use real-image tracks, covariance/scale patterns and visibility to identify a limited region. Compare preserving its scaffold against reinitializing only that region from verified geometry, with a fixed overlap shell. Avoid deleting all fuzzy content or genuine foliage, glass and thin surfaces.

**Competing repair hypotheses:** on an ambiguous region, let the agent request appearance-only versus structural repair, then compare their native candidates against the same observed evidence. Initially limit the branch count to two as a proposed budget control, not a proven optimum.

**Trajectory-order sensitivity:** compare forward and reverse propagation through the same poses. Disagreement can expose dependence on causal context; agreement does not establish truth. Keep complete hypotheses separate rather than averaging conflicting geometry.

For each speculative branch, count extra generation and fitting cost. Promote only mechanisms that improve the quality–cost tradeoff or solve a documented failure class.

## Stage 6 — Measure the agent's contribution

The VLM remains the controller throughout implementation. For research attribution, compare replayed fixed paths, numerical coverage/artifact heuristics and live VLM selection using identical downstream modules and budgets.

Test its decisions separately: view choice, intervention choice, and reference choice. Then test the combined policy over several repair rounds. Its memory should retain accepted/rejected interventions, evidence coverage and synthetic lineage, so repeated agreement among descendants of one edit is not mistaken for independent evidence.

Give the agent diagnostics that explain why a repair failed: edit changed landmarks, propagation lost detail, geometry unsupported, boundary regressed, or native fitting failed. It may change strategy, request another view, or stop. This makes the existing exploration loop a testable decision policy.

A credible agent contribution is better scene improvement per cost or fewer destructive updates because of its choices. Merely having a VLM in the pipeline does not establish novelty.

## Experiment discipline and stopping rules

Run development screening on diffuse detail, glossy appearance, floaters, fuzzy surfaces, thin structures and missing content. Select evaluation regions independently of method outputs. Tune masks, thresholds and stopping on development scenes; final test photographs remain inaccessible to every module, including the agent.

Keep an official ArtiFixer3D reproduction alongside the common-fitter experiments. The reproduction establishes an external reference; shared-fitter arms isolate mechanisms. Report native and postprocessed endpoints separately. All starting scenes must exclude test images during reconstruction.

Use incremental additions to diagnose effects, but do not assume an addition that fails alone cannot help in combination. Explicitly test image editing × video propagation, geometry checks × confidence, SH restrictions × geometry restrictions, and suppression × generation. After selecting a final recipe, remove one component at a time on fresh evaluation scenes to verify attribution.

Measure native PSNR/SSIM/LPIPS, structural error where independent geometry exists, photographed-detail preservation, collateral regression, temporal stability, accepted/rejected repair rates and multi-round drift. Report scene-level variation, API cost, GPU time, peak memory, resolution, splat count and all retries. Compare equal-input/equal-budget arms and quality–cost curves; caching makes experiments affordable but does not erase deployment costs.

If a module improves postprocessed images but not the native scene, retain it as an explicit rendering result. If a module helps only one defect class, report that conditional gain rather than asserting a universal winner. If a cheap baseline matches it, prefer that baseline in the final recipe.

## Implementation sequence in the current project

1. Scene representation/rendering: full-SH preservation, camera parity, photograph/pose input and immutable original asset.
2. Scene-run configuration: shared image backend with existing CliRelay integration; independent image, video and fitting adapters.
3. Agent/runner: preserve navigation; add repair bundles, intervention types, provenance, budgets and diagnostic feedback.
4. GPU worker: replace the single-camera apply path with isolated multiview candidates, real-image supervision and atomic acceptance/rollback.
5. Experiment runner: cache/replay proposals and targets; expose every research addition as a configuration switch; emit native versus postprocessed evaluations.
6. Implement Stages 1–4 as successive controlled additions; launch the bounded Stage 5 branches once the hybrid path is functional; evaluate live agent decisions in Stage 6.

These are planned changes. No production pipeline modification or new GPU benchmark was performed while writing this strategy.
