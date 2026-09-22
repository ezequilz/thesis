# Method selection: the strongest next experiment

Research update · 22 September 2026 · supplements and refines the [executive plan](/Users/juliuskleinle/Desktop/thesis-1/docs/multiview-executive-summary.md)

## Recommendation

**Use a specialist, reference-conditioned video restoration model to propose coherent camera trajectories, then distill selected frames into a constrained candidate of the existing scene. Start with ArtiFixer as the quality candidate; compare ArtifactWorld for localized repair and FixAnything as a simple rendered-video baseline. Keep GPT image editing as a separate controlled arm.**

This is my best-supported experimental choice, not a proven optimum for your assets. Published methods generally have original clean reference images. Substituting good renders of an exported splat is a meaningful distribution shift, and none of the inspected papers establishes a winner for precisely that setting.

The ideal objective is native scene fidelity, geometric consistency, preservation of good content, and useful camera coverage, subject to a compute budget. Individual-image sharpness is secondary when it conflicts with these. The useful contribution of the VLM is choosing *where sufficient evidence and valuable repair opportunities coincide*.

## Changes to the earlier plan

1. **Promote specialist multi-view restoration to the primary method.** The earlier canonical-GPT-view route remains useful, but relies on general image models to infer camera consistency they do not explicitly enforce.
2. **Generate dense paths; fit diverse keyframes.** A 4–8-image edit bundle is not necessarily the right input format for video models. Supply their expected temporal sequence, then select a smaller, nonredundant set for fitting. Correlated frames must not overwhelm unchanged anchors.
3. **Test across intersecting paths.** Smooth motion along one trajectory is insufficient evidence of a static 3D scene. Generate overlapping paths with shared trusted anchors; validate from translations and elevations outside those paths.
4. **Separate appearance repair, geometry repair, and completion.** Each needs a different editable parameter set and acceptance standard. The old geometry must constrain reliable regions without becoming an immutable target inside genuinely wrong regions.
5. **Test recovery from known corruption before judging unknown defects.** With only exported assets, create controlled local corruptions of a good source region and reserve source renders at unseen poses. This makes the fitting stage testable without claiming the source is real-world ground truth.
6. **Keep a fresh reconstruction branch as a serious comparator.** In-place fitting is the preservation-first default, but a sufficiently bad initialization may trap optimization. Compare local reinitialization under the same targets before concluding the generator failed.

## What the leading methods offer

### ArtiFixer / ArtiFixer3D — primary quality candidate

ArtiFixer uses rendered RGB, opacity, camera information and reference images; its autoregressive design supports long trajectories. ArtiFixer3D distills the generated views into an explicit scene; ArtiFixer3D+ adds another image-refinement stage. For this project, the relevant endpoint is **ArtiFixer3D-style native reconstruction**, not the “+” presentation. [Project](https://research.nvidia.com/labs/sil/projects/artifixer/)

The official release supplies 14B and 1.3B variants and a reconstruction workflow based on real anchors plus generated targets. The repository describes the smaller variant as fitting on one 80 GB GPU; this is not evidence that every 14B configuration fits your A100 allocation. Start with a small compatibility run, then profile the 14B quality configuration. [Official implementation](https://github.com/nv-tlabs/ArtiFixer)

My preference follows from its explicit conditioning and demonstrated native-scene results. Its disadvantages here are integration effort and the unknown quality of synthetic reference images. Keep its official reconstruction pipeline as a later reproduction baseline; do not require a complete renderer migration before testing its target images in your own optimizer.

Bidirectional teacher checkpoints are also now released. They offer a useful short-bundle ablation against autoregressive generation, but their availability does **not** establish that they produce better reconstructions. They require different sampling from the distilled student. [Official model card](https://huggingface.co/nvidia/ArtiFixer)

### ArtifactWorld — strongest localized-repair challenger

ArtifactWorld predicts artifact heatmaps and uses them in video restoration; its reconstruction objective mixes restored targets with original sparse views. This directly addresses the need to distinguish valid scene content from regions that may change. Its paper uses different dataset/view-ratio protocols from ArtiFixer, so its scores do not form a shared ranking. [Paper](https://arxiv.org/html/2604.12251v1)

The current repository provides two-stage inference and weights; training code/data remain planned. Its example preprocessing inserts ground-truth first/last frames. For your asset-only test, replace these with explicitly tagged source-render anchors and record that deviation. [Release and input contract](https://github.com/fyting/ArtifactWorld)

My assessment: this may outperform ArtiFixer when artifacts are localized and most of the image should stay intact. A predicted heatmap is an edit proposal, not calibrated correctness; combine it with geometry and preservation checks.

### FixAnything — simplest direct video baseline

FixAnything learns rendered-video cleanup with trusted-frame conditioning and a geometry-aware preference objective based on recovered camera-pose accuracy. That training signal is pertinent to your problem: the output should support the supplied camera motion, not just look plausible. [Project](https://fix-anything.github.io/)

Its released interface accepts a rendered video or frame folder; the documented path uses 61 frames, 832×480 processing, and clean-frame indices. This is a convenient integration baseline, although its documented output is a refined video, not a validated repaired PLY. You must supply the distillation and acceptance stage. [Inference implementation](https://github.com/kvuong2711/fix-anything)

My assessment: use it to test the video-to-splat hypothesis quickly. Do not infer superiority to ArtiFixer from separate benchmark tables. Good rendered endpoints replacing true clean frames are still an unproven adaptation.

### GSFixer — established reference-guided reconstruction baseline

This is the reference-guided video work, distinct from the GSFixer image component inside GSFix3D. It conditions restoration on semantic and geometric features of reference views. The official release includes model links, restoration inference and reconstruction scripts. [Paper](https://arxiv.org/html/2508.09667v1), [code](https://github.com/GVCLab/GSFixer)

My assessment: a valuable reproducible baseline and fallback. Learned geometry features can help, but predictions derived from degraded source renders cannot be treated as independent geometric evidence.

### GaussVid — recent camera-aware challenger

GaussVid adds camera-conditioned geometric signals and boundary anchors. Its paper evaluates a different clip protocol and reimplements GSFixer on a different backbone, so its superiority claims do not establish superiority to the released GSFixer or ArtiFixer in your setting. [Paper](https://arxiv.org/html/2608.21849v1)

The official repository links code and LoRA weights. Include it in a second comparison round if known-camera consistency is the main remaining failure. [Code](https://github.com/Xinhui-99/GaussVid)

### GSFix3D / Difix3D — essential controls

GSFix3D uses original keyframes alongside repaired views. The present single-view Qwen implementation does not reproduce its complete data conditions. Difix3D provides a specialized image-restoration-and-reconstruction baseline; Difix3D+ includes a further enhancement stage. Keep these controls to measure the actual benefit of coordinated generation. [GSFix3D](https://arxiv.org/html/2508.14717v1), [Difix3D+](https://research.nvidia.com/labs/toronto-ai/difix3d/)

### GPT editing, new-world generation, and renderer upgrades

GPT remains useful for semantic fixes and detail proposals. A promising later hybrid is to create one improved canonical appearance, then propagate it through a geometry-aware restoration system. Treat the edited reference as a generated hypothesis with lower trust than good source content. This hybrid is our proposed experiment, not a validated published combination.

CAT3D generates multi-view inputs for reconstruction; Lyra 2.0 addresses persistent world generation using geometry to retrieve relevant history and reduce drift. These are relevant to rebuilding large missing regions, but their creative freedom and reconstruction scope make them less direct choices for preserving an existing asset. Borrow the idea of geometry-indexed memory for later agent loops. [CAT3D](https://cat3d.github.io/), [Lyra 2.0](https://arxiv.org/abs/2604.13036)

3DGUT/3DGRUT is a reconstruction/rendering choice, separate from the restoration model. A renderer upgrade cannot make contradictory generated observations compatible. Preserve the current coordinate system and establish renderer parity first; test alternative reconstruction backends only with identical target data. [Official 3DGRUT repository](https://github.com/nv-tlabs/3dgrut)

## What quantitative evidence actually supports

ArtiFixer's paper reports the following **native-reconstruction** results on its DL3DV artifact-removal protocol (Table 1): 3DGS **17.18 dB / 0.384 LPIPS**, Difix3D (3DGS) **17.80 / 0.314**, and ArtiFixer3D **20.14 / 0.256**. This supports testing ArtiFixer3D, but is not an isolated generator comparison: reconstruction choices and full pipelines differ. It also does not test exported-assets-only references. The paper's timing uses a GB300, so those speeds should not be assumed for your A100. [Paper and evaluation protocol](https://arxiv.org/html/2603.00492v2)

Do not combine numbers from DL3DV 3/6/9-view tests, percentage-based splits, and camera-interval clip tests into one leaderboard. Report native 3DGS, restored video, and native-plus-postprocessing separately. Those answer different questions.

## The proposed method for your assets

### A. Build a trustworthy view atlas

Preserve source SH and exact camera/rendering conventions. Survey the scene before repair and store a spatially indexed atlas of good views, with confidence masks rather than all-or-nothing “clean” labels. Use source depth, opacity, projected Gaussian support and cross-view stability as clues. These cannot certify true geometry, but help distinguish reliable structure from unsupported content.

Retain two provenance layers: immutable source references and generated-but-accepted references. Acceptance never turns a generated view into a real observation. Retrieve both spatially, rather than conditioning only on the most recent edited screenshot.

### B. Plan short paths around one object or connected surface

For the first region, choose six trusted source-render anchors around the relevant structure and its boundary. Capture two overlapping translated paths, such as a shallow lateral arc and an elevated arc, with shared context. Avoid moving through walls, grazing angles, or large empty areas in the first test.

Use the backend's required sequence length; FixAnything's documented 61-frame input is a concrete starting point for its arm. For other backends use their native input contract over the same camera envelope. Different input sequence lengths are acceptable if the retained optimization cameras and evaluation protocol are controlled.

After generation, retain roughly 8–12 diverse fitting frames across the paths; add more only when coverage or consistency requires it. Dense frames provide the video model with continuity, but are not dozens of independent observations. These counts are proposed experimental settings.

The agent should rank **expected useful, consistent repair per unit cost**, accounting for salience, source support and uncertainty. It should postpone regions with no trustworthy supporting views. This is more useful than exclusively selecting either the worst artifact or the prettiest view.

### C. Choose one coherent hypothesis, not a pixelwise average

Sample a few candidate bundles, initially three for the selected region. Score preservation, feature tracks at known camera poses, cross-path overlap, object identity, occlusion handling and the fraction of pixels with usable support. Select the most coherent candidate. Averaging different plausible completions produces blurry targets and can create impossible geometry.

Use disagreement across samples to identify ambiguity, but not as a proof of truth: models can agree on the same hallucination. Combine uncertainty with structural checks. Uncertainty-aware pseudo-supervision has precedent in UAR-Scenes, although its single-image reconstruction setting differs from yours. [UAR-Scenes](https://sarosijbose.github.io/UAR-Scenes-Website/)

Do not warp reflections as though they were surface texture. Do not demand agreement with wrong source depth inside a repair mask. Instead require the new hypothesis to support its own consistent correspondences while matching the unchanged surrounding shell.

### D. Distill with the minimum necessary freedom

Classify the region before optimization:

- **Appearance defect, credible geometry:** freeze positions, covariance and opacity. Fit a bounded color residual with existing directional SH preserved initially.
- **Wrong geometry:** unlock only the implicated splats and a small boundary band. Use triangulated tracks or multi-view depth with confidence; do not let opacity growth substitute for geometric agreement.
- **Missing content:** seed a local patch from consistent multi-view hypotheses in the existing coordinate frame. Label it as completion; retain uncertainty and source overlap.

Initially restrict degree and parameter freedom. For existing splats, regularize SH toward the source, not toward zero. For new splats, start with degree zero and increase only when useful. Geometry changes and missing directional coefficients cannot both be solved reliably from a narrow arc by simply adding iterations.

A more precise future SH diagnostic is a visibility-weighted design matrix: `H_g = Σ_v w_gv y(d_gv)y(d_gv)ᵀ`. Its conditioning measures directional support for each Gaussian's SH basis. Use it to flag poorly supported updates, not as a guarantee of identifiability in an alpha-composited scene. Original SH already contains information worth preserving; refitting it is unnecessary unless evidence demands a change. This is a proposed diagnostic, not a validated component of the cited methods.

Normalize loss contributions by region and view group. Keep full-resolution source anchors so video-model downsampling does not erase good detail. A scene rendered at high resolution after fitting low-resolution targets has not automatically gained true high-resolution detail. If detail enhancement is later added, require it to pass the same multi-view checks.

### E. Compare refinement with local reinitialization

If coherent targets still cannot be fitted, branch from the same accepted scene: one candidate conservatively refines the existing region; another reconstructs a replacement patch while freezing surrounding geometry. Same cameras, anchors, targets and evaluation. Prefer whichever preserves context and improves unseen views.

This distinguishes an optimizer/initialization failure from a generator failure. A full-scene rebuild becomes appropriate only if the failure is global. Never merge by simple concatenation: evaluate visibility/opacity competition and refine the overlap jointly, as detailed in the executive plan.

## First experiment: a decision, not a large integration project

**Stage 1 — verify that the fitter can succeed.** Take a reasonably good source region, save its untouched reference scene, and inject a localized known defect into a copy: first a color corruption, later missing/misplaced splats. Use a subset of clean source renders as oracle targets and reserve translated views of the untouched scene for evaluation. This measures recovery of the source, not recovery of reality. If even consistent oracle targets fail, fix representation, rendering or optimization before adding a generator.

**Stage 2 — test target generators with rendered anchors.** Freeze the original scene, cameras, masks and fitting schedule. Compare independent GPT edits, ArtiFixer, ArtifactWorld, and FixAnything where deployment permits. Run the first completed specialist adapter before waiting for all competitors. Start with a small ArtiFixer feasibility run; use the 14B model for quality evaluation when resources permit, keeping 1.3B results separately labeled.

Use equal retained fitting views and record actual generator compute, resolutions and costs. Include both a common-resolution comparison and each method's preferred operating point. A method should not win simply by being given more targets or sharper input anchors.

**Stage 3 — test true unknown defects.** Evaluate one appearance-dominated region, one geometry-dominated region and one incomplete region. Use blind native-render comparisons and geometric checks. Reserve off-path validation cameras and a separate locked final test path. Do not ask the image model to repair those views before judging the candidate.

**Stage 4 — enable repeated agent transactions.** Only after a fixed region succeeds, measure several updates with global replay, explicit rejection, rollback and spatial memory. The criterion is retained quality over accepted revisions, not the number of repair calls.

Before the study, derive acceptance tolerances from no-op renderer repeatability, controlled-corruption recovery and blind review. This replaces treating the earlier illustrative PSNR/LPIPS thresholds as universal rules. Improvements must survive off-path inspection; no-reference aesthetic scores alone cannot decide promotion.

## How we decide which method to keep

Select ArtiFixer if it gives the best native-view improvement and source preservation at acceptable cost. Prefer ArtifactWorld if local masks preserve more valid content. Prefer FixAnything or GSFixer if their simpler integration or better behavior with rendered references produces stronger accepted results. Promote GaussVid if the specific remaining limitation is camera fidelity and it wins the same evaluation.

If all models fail only with synthetic references, the likely bottleneck is the reference distribution, not insufficient loop length. Test adaptation on deliberately degraded exports with clean rendered anchors, followed by capture-backed evaluation. If images are coherent but native reconstruction fails, change the fitting/initialization branch. If geometry is stable but detail is weak, work on multi-view detail supervision. These are different experiments.

**Research hypothesis:** a VLM-guided, source-preserving selection and distillation loop can turn strong restoration models into reliable asset improvement by controlling which hypotheses are accepted. That is the approach worth testing. “State of the art” remains an outcome to measure on the later capture-backed benchmark, not a property inherited from the chosen generator.
