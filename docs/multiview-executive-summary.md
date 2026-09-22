# From single-view repair to consistent 3D scene refinement

> Updated scope: the next research phase includes original photographs and calibrated poses. See [the ArtiFixer research proposal](artifixer-research-proposal.md) for the revised contribution, matched-input comparison and experimental plan. Asset-only recommendations below remain a secondary track.

Executive review and proposed research plan · 22 September 2026

**Refined recommendation:** see the [method comparison and first-experiment design](/Users/juliuskleinle/Desktop/thesis-1/docs/multiview-method-selection.md). It prioritizes specialist trajectory restoration (ArtiFixer, with ArtifactWorld and FixAnything comparators), separates dense generation from diverse fitting views, and adds oracle-target and intersecting-path tests. Its experiment sequence and calibrated acceptance guidance supersede the illustrative sequence/settings below.

**Decision:** Retain the VLM exploration harness, but replace immediate single-image scene mutation with **validated, multi-view repair transactions**. First preserve the complete source representation, including spherical harmonics (SH). Then optimize one candidate scene against consistent edited views, trusted unchanged observations, and geometric constraints. Publish only candidates that improve independent views without damaging preserved regions.

This is a design recommendation informed by repository inspection and primary research, not a demonstrated quality improvement. The reported scene degradation comes from the project owner; this review did not run a new CUDA benchmark or establish which factor contributes most. The owner confirmed two research tracks: **exported splat assets only now**, followed by **scenes with original photographs and calibrated poses for the final research comparison**. The primary proposal below works within the current asset-only setting.

## Executive summary

The current system already provides useful infrastructure: a VLM navigates a rendered scene, identifies artifacts, invokes an image editor, fits the generated image back into the Gaussian representation, reloads the checkpoint, and continues exploring. The missing component is a reliable decision about whether each update improves the **3D scene**, rather than fitting one attractive image.

Three problems take priority over choosing a stronger image model:

1. **The working representation loses view-dependent appearance before repair.** `GaussianScene` holds RGB derived from SH degree zero; PLY/SOG loaders discard higher-order coefficients, and the PLY writer cannot preserve them. This is a concrete implementation limitation. It is different from high-order SH overfitting: the current path does not optimize those missing coefficients at all.
2. **The default automated repair is underconstrained.** It updates positions, scales, rotations, opacity and base color from one edited RGB target, with preservation terms disabled in the default `original` mode. Many incompatible 3D explanations can match the same image. Repeating that operation can accumulate geometry errors and undo previous views.
3. **There is no scene-quality acceptance gate.** A successful worker response promotes the repaired PLY. Lower target-fitting loss is not evidence of improved geometry, novel-view fidelity, or temporal stability.

Returning to GPT Image through CliRelay is a worthwhile controlled experiment, but cannot solve these constraints by itself. Multiple independently edited images are also insufficient: they must depict compatible geometry at their known camera poses.

The recommended next milestone is deliberately narrow: demonstrate a repeatable improvement on **one local region**, across unseen nearby viewpoints and a preserved global trajectory, using a fixed multi-view bundle. Only then let the VLM select and execute further transactions autonomously.

## What the repository currently does

The root dashboard supports exploratory episodes; `/repair` supports isolated repair experiments. `/scene-runs` adds a persistent manager, an LRZ GPU worker, deadlines, checkpoint transfer, and cumulative scene reloads. These are valuable foundations to keep.

The automated loop is:

`current scene → rendered camera view + map → CliRelay VLM action → repair trigger → Qwen edit → single-view CUDA fit → checkpoint promotion → continued exploration`

The trigger supports artifact reports, `regenerate=yes`, or every observation after arming. The worker's default image-editor factory directly constructs Qwen. Its default `original` repair mode selects the implementation in `repair_gsfix3d_working_backup.py`, with 20 iterations, a 0.8 L1 / 0.2 SSIM objective, densification every five iterations, and final pruning. The separate `looped` mode uses different safeguards and repeated chunks; it should not be conflated with `original`.

The image-edit prompt in scene-runs is static and ignores the action's detail: it asks broadly for artifact repair, plausible geometry reconstruction, and upscaling. The request contains one camera and one RGB target. The worker calls `apply_until` on that view and does not invoke joint keyframe replay. A replay method exists elsewhere, but the automated path inspected here does not use it.

Evidence anchors:

- [Scene representation](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene/types.py:16), [PLY loading](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene/ply_loader.py:99), [PLY writing](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene/ply_loader.py:136), and [SOG SH omission](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene/sog_loader.py:19).
- [Scene-run initialization](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene_runs/runner.py:260) writes `scene_original.ply` from the decoded scene. It is immutable within the run, but is **not a lossless copy of the original asset**. Preserve the actual source bytes separately.
- [Default optimizer settings](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene_runs/gpu_worker.py:53), [Qwen factory](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene_runs/gpu_worker.py:170), [repair dispatch](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene_runs/gpu_worker.py:182), [single-view call](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene_runs/gpu_worker.py:478), and [promotion](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene_runs/runner.py:448).
- [Static edit prompt](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/scene_runs/runner.py:100), [trainable parameters](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/repair_gsfix3d_working_backup.py:625), and [loss selection](/Users/juliuskleinle/Desktop/thesis-1/src/splat_explorer/repair_gsfix3d_working_backup.py:772).

One inspected saved request, [repair-00002](/Users/juliuskleinle/Desktop/thesis-1/outputs/scene-runs/run_20260916_093429/requests/repair-00002/metrics.json), records L1 decreasing from 0.035500 to 0.024445 over 20 iterations, with 797 spawned Gaussians. Its Qwen output visibly sharpens a column and door frames. That observation supports improved single-image appearance and recorded target fit only; it does not measure adjacent-view improvement. The reported before/after loss may also involve different rendering paths, so renderer parity must be established before interpreting the magnitude.

## Why single-view fitting fails, and what SH changes

A Gaussian carries position, covariance, opacity, and directional color. A pixel is a visibility-weighted mixture of contributions along its ray. A single image does not uniquely determine which depths, shapes, transparencies, or colors produced that pixel. Moving or enlarging splats, raising opacity, and adding thin surfaces can reduce image loss while making other views worse. Densification increases capacity; it does not create missing geometric evidence.

For SH appearance, a common parameterization is:

`c_g(d) = 0.5 + Σ(l=0…L) Σ(m=−l…l) a_g,l,m Y_l,m(d)`

Each color channel has `(L+1)²` coefficients: degree three has 16. At one direction, an isolated Gaussian's observed color supplies just one linear constraint per channel, before accounting for unknown visibility and other Gaussians. Nearby cameras can also give highly correlated directional constraints. Image count alone does not establish identifiability.

This produces two distinct failure modes:

- **Current DC-only path:** a color correction applies in every viewing direction. An editor's interpretation of a highlight or reflection can be baked into the base color; geometry and opacity may then compensate for incompatible targets. Discarding source SH can itself remove valid appearance information.
- **Future unrestricted SH path:** many coefficients can explain the edited direction while behaving poorly elsewhere. SH can hide inconsistent pseudo-targets instead of correcting geometry. High-frequency appearance may flicker or change implausibly between views.

The [original 3DGS work](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) establishes the representation and joint appearance/geometry optimization. The identifiability argument above is a mathematical diagnosis, not a measured attribution for this repository.

**Recommended SH policy:** preserve all source coefficients and their degree throughout load, copy, filtering, densification, save, and reload. Preserve trusted regions' SH unchanged. For edited regions, initially freeze directional terms and geometry; optimize a constrained base-color residual. If geometry is demonstrably wrong, unlock it only under multi-view geometric support. Introduce new directional coefficients progressively, with angular coverage checks and regularization toward the source. New Gaussians start at degree zero; higher degrees must earn their complexity on validation views.

Do not force specular pixels to have identical RGB across views. Use correspondence, depth, silhouette, and material-aware confidence to constrain geometry; retain legitimate directional appearance. Do not globally remove SH as a permanent solution.

## The upgraded loop

`survey → choose region → capture fixed view bundle → propose consistent edits → reject inconsistent targets → optimize candidate → validate → commit or rollback → explore again`

### 1. Establish a faithful baseline

Keep a byte-preserved source asset, a full-SH decoded checkpoint, and the last accepted checkpoint. Record exact intrinsics, world-to-camera convention, scene scale, SH basis ordering and direction convention, color processing, background, exposure, and rendering settings.

The present viser path receives base RGB; adding SH only to CUDA would leave the agent inspecting a different appearance model. Prefer one authoritative CUDA renderer for observation, editing inputs, training, and evaluation; retain the browser viewer for interaction after its parity is checked. Compare outputs at identical cameras before image inference. Include load/save/reload tests with nonzero SH, coefficient-layout tests, and matched-camera rendering checks.

### 2. Make the agent a survey planner

Keep artifact detection, but change its decision from “repair this screenshot” to “investigate this region.” Maintain a scene memory of semantic objects, surface coverage, trusted appearances, uncertainty, previous edits, and rejected attempts.

The VLM names salient targets and proposes acquisition goals. Numeric geometry code chooses valid camera poses and verifies overlap, translation baseline, visibility, collision clearance, and novelty. Pure camera rotation does not supply triangulation parallax; nearly identical screenshots are redundant evidence.

A practical initial bundle is **4–8 translated, overlapping target views**, **4–8 unchanged anchor views**, and **3–5 local validation poses** that neither the generator nor the optimizer sees. Also retain a fixed global validation path. These counts are starting hypotheses, not literature guarantees. Aim initially for roughly 60–80% region overlap between neighboring targets, then adapt to depth, occlusion, thin geometry, and scene scale. Measure surface visibility, not just frustum overlap.

Select batches for complementary coverage: salient surface area × uncertainty reduction × editability, discounted by redundant views, occlusion risk, travel, and inference cost. A greedy coverage baseline should precede more expensive uncertainty estimation. [FisherRF](https://arxiv.org/abs/2311.17874) provides a research basis for information-driven selection; its acquisition setting does not make newly rendered synthetic views independent real observations.

### 3. Choose which images are edited and which stay unchanged

Maintain four explicit roles:

- **Real anchors:** original photographs with calibrated poses, when available. Use for generation context and strong reconstruction supervision; exclude known corrupt pixels.
- **Preservation anchors:** trusted renders from the frozen source or last accepted checkpoint when photographs are unavailable. Use outside edited regions and at region boundaries. They preserve prior appearance but cannot certify hidden geometry.
- **Edited targets:** overlapping views of the chosen defective region. Keep the camera and frame dimensions fixed, edit only approved content, and attach per-pixel confidence and provenance.
- **Validation views:** withheld real images where available, plus held-out camera paths. Never pass final test images to generation, training, or checkpoint selection.

**Yes, unchanged views should be included.** Select good views that constrain the same geometry, boundary views that connect to the surrounding scene, and a small global replay set. They need not all be sent to the image model: context images and optimizer supervision have different roles. Do not use known-bad pixels in unchanged renders to oppose the intended repair; mask them. In overlapping repair regions, track which prior target has been superseded so obsolete targets do not fight newer accepted geometry.

The current pipeline has only exported splat assets, as confirmed by the owner. Use selected scene renders as preservation anchors, backed by full-SH appearance, Gaussian support, opacity, depth confidence, surface orientation estimates and scene scale. A rendered depth/normal is a property of the estimated scene, not an independent measurement. Do not mistake high opacity for correct geometry: a floater can be opaque. Call unobserved content **plausible completion**, not recovered ground truth.

For this asset-only track, first gather a stable atlas of good source views before any edit. Keep that atlas immutable, and add separately tagged references from accepted revisions. Select views that expose different sides and boundary relationships of the same salient object; do not select only the prettiest frontal view. Render all targets in a transaction from the same parent checkpoint. In trustworthy geometry, depth-warp canonical texture changes into supporting views before requesting edits for residual defects or disocclusions. Where geometry is unreliable, require multi-view generated agreement and independent correspondence checks, with lower confidence and stronger preservation of the surrounding shell.

In the later capture-backed track, replace or supplement preservation renders with original calibrated photographs and independently withheld ground truth. Keep an asset-only ablation on those same scenes by hiding the capture images from the method while retaining them for evaluation. This directly measures how much true information the generated completion recovers and how much original photographs improve results.

### 4. Generate targets as a coherent set

The near-term CliRelay route should edit a canonical view with nearby references, then condition neighboring edits on that accepted canonical proposal, their own renders, and depth-warped support where reliable. Reconcile the full bundle before fitting; processing views sequentially must not cause immediate scene mutation.

The edit contract preserves projection, object identity/count, silhouette, layout, illumination, and reliable pixels. Remove the blanket upscaling request during the geometry experiment. If dimensions change, update intrinsics for the actual transform; reject unexplained crop or perspective changes. Masked generation alone is not a guarantee of unchanged pixels, so composite trusted pixels explicitly where appropriate.

Common prompts, random seeds, multi-image input, or a contact sheet are **not geometric consistency guarantees**. An image API is not inherently a calibrated multi-view generator. Keep a geometry-aware video/multi-view restoration backend behind the same interface as a research alternative.

Before optimization, test feature tracks, epipolar residuals at known poses, bidirectional reprojection, depth agreement, silhouette/object identity, and drift outside edit masks. Exclude occlusions and low-confidence depth. Expected splat depth can blend surfaces and be unreliable near floaters, glass, and holes; it is a prior, not truth. Where the old geometry is wrong, permit independently supported new geometry rather than rejecting every deviation from the old depth.

Reject inconsistent image regions, lower their weights, or regenerate the bundle. A sharper incompatible image is a bad training target. Generated views remain correlated pseudo-observations even if they agree.

### 5. Jointly optimize a candidate in the existing coordinate frame

Initialize from the accepted scene. Optimize only a visibility-aware 3D region plus a boundary band. A 2D mask alone is insufficient: different depth layers can project to the same pixels. Accumulate per-Gaussian contribution through transmittance across views and use confident depths/tracks to assign the editable set. Freeze the remainder.

Use one persistent optimizer across sampled bundle views; compute all losses with respect to the same candidate. A proposed objective is:

`L = λreal Lreal + λedit Lconfidence-masked-edit + λkeep Lpreserve + λgeo Lgeometry + λtrust Lparameter-change + λSH Ldirectional-change + λbudget Lcomplexity`

Normalize losses by valid pixel count and balance their gradient scales. Start with real anchors strongest, generated targets confidence-weighted, and preservation losses restricted to trusted pixels. Use masked L1/SSIM for RGB, robust multi-view depth/track/silhouette constraints where available, and scene-scale-normalized position/covariance trust regions. For existing SH, penalize deviation from the accepted coefficients; for new SH, penalize unsupported directional energy.

Fit base appearance first. Unlock geometry only after consistent evidence persists across translated views. Densify where residuals and coverage support a real missing surface; triangulate or fuse confident multi-view depth to seed empty regions. Ordinary clone/split operations near existing splats cannot reliably reconstruct a large disconnected hole. Prune only after checking visibility over the whole anchor/target bundle. Finish with a constrained appearance pass.

Use validation improvement and a compute ceiling to stop optimization. Repeating the same single-image objective for longer can intensify overfitting. The 20-iteration schedule is a baseline setting, not a quality criterion.

### 6. Commit only after independent validation

Write an isolated candidate checkpoint and evaluation record. Keep the accepted scene active until the candidate passes. Evaluate local unseen poses, preserved regions, region boundaries, a global sweep, and a short camera trajectory. A successful API or worker response means the job completed, not that the scene improved.

If validation fails, restore both the host and GPU worker to the accepted checkpoint, discard its optimizer state, and retain the failed bundle with reasons. Merely reverting the viewer would leave future requests training from a rejected worker scene. Tag requests with the parent scene hash; reject stale results. Promote the PLY and metadata atomically, then invalidate renderer caches and update the agent's scene revision.

## Rebuilding and merging scenes

**Default: refine or replace a local region inside one scene.** This creates a new scene version while preserving registration and reducing seams. A completely independent scene reconstruction followed by PLY concatenation is a higher-risk route.

If a region is missing or irreparably malformed, reconstruct a local replacement using the bundle's existing camera poses and trusted depth/triangulation. Include overlapping context as a frozen registration shell. Keep replacement splats and source splats separate until validation, then jointly optimize the replacement and a narrow boundary band against shared observations. Remove old contributions only where replacement coverage is verified; preserve source content elsewhere.

Simply overlapping both sets can double opacity: two identical splats with opacity α have combined opacity `1−(1−α)²`, which exceeds α. Position-nearest-neighbor deduplication can also remove distinct foreground/background surfaces. Resolve redundancy through visibility, depth, covariance, and appearance, followed by joint refinement; a spatial fade alone does not guarantee correct compositing.

If independent reconstruction is unavoidable, estimate a robust SE(3) or Sim(3) transform from trusted overlap/camera correspondences. For `x′ = sRx+t`, transform covariance as `Σ′ = s²RΣRᵀ`; transform cameras consistently and rotate the SH directional basis as well. Transforming centers alone is incorrect. Validate overlap registration, exposure, depth, and seams before replacement. Do not infer a new camera system from generated images when the renderer already supplies calibrated poses.

A global rebuild becomes reasonable when global geometry is poor and enough consistent targets **and anchors** exist. It is an offline candidate branch requiring full-scene evaluation, not the next default loop step. Surface-oriented representations such as [2DGS](https://surfsplatting.github.io/) are useful geometry baselines, but changing representation is a separate experiment and does not remove inconsistent supervision.

## Research that changes the design

- **GSFix3D:** its method adds repaired images to original captured keyframes and optimizes the augmented dataset. Reproducing the single-view iteration count with a different editor and no real-keyframe stage is not a full reproduction. Implement the missing data constraints before attributing failures to the paper. [Paper, §3.3](https://arxiv.org/html/2508.14717v1)
- **Difix3D+:** a relevant repair-and-distillation baseline using a specialized restoration model. Compare its underlying reconstructed scene as well as its displayed image quality. [Project](https://research.nvidia.com/labs/toronto-ai/difix3d/)
- **GSFixer, reference-guided video diffusion:** a different work from the GSFixer component named inside GSFix3D. It combines semantic and geometric reference conditioning with restoration over camera trajectories. It is directly relevant to cross-view target coherence. [Paper](https://arxiv.org/html/2508.09667v1)
- **ArtiFixer, SIGGRAPH 2026:** uses camera control, opacity conditioning, reference views, and autoregressive generation; benchmark this as a current high-quality research alternative. Its authors report strong benchmark gains, which are not verified on this project. [Project](https://research.nvidia.com/labs/sil/projects/artifixer/)
- **ArtiFixer implementation detail:** ArtiFixer3D reconstructs from real anchors plus generated targets; ArtiFixer3D+ applies an additional image-refinement stage. Do not report the latter's postprocessed render quality as if it were native splat quality. [Official repository](https://github.com/nv-tlabs/artifixer)

These support the design direction; none guarantees state-of-the-art performance on arbitrary exported assets or proves that generative completion recovers the true scene.

## Image-backend decision

Keep Qwen as the low-cost integration baseline and run a controlled comparison with the established `gpt-image-2` CliRelay path. Official documentation checked on this review date lists `gpt-image-2.5-sunburst` and `gpt-image-2.5-flare` for generation/editing. Their routing and access through this project's CliRelay have not been verified. [Official image-generation documentation](https://developers.openai.com/api/docs/guides/image-generation)

The repository needs real plumbing changes: scene-runs hardcodes the Qwen factory, the GPT configuration constructor uses the `IMAGE_MODEL` constant, and the public edit interface accepts one image path. Separate provider/backend from model ID, route through the shared factory, and add a versioned bundle interface for references, masks, output sizing, provenance, and backend capabilities. Reuse the root/repair CliRelay client code. A model dropdown alone will not switch the scene-run worker or add multi-view conditioning.

Check the relay's model catalog and run a small explicit edit capability test during implementation, recording the actual returned model, dimensions, reference support and request metadata. Do not silently substitute models. Compare frozen target bundles so backend quality is separated from optimizer and camera-selection changes.

## Implementation sequence and experiments

**Milestone 0 — trustworthy baseline.** Extend `scene/types.py`, both loaders, PLY export, and all renderer/repair paths for full SH. Byte-preserve source assets. Establish matched-camera renderer parity. Separate accepted/candidate checkpoints and create fixed evaluation cameras. Exit: load/save/reload preserves coefficients and renders within measured numerical tolerance; a no-op transaction changes nothing meaningful.

**Milestone 1 — controlled multi-view transaction.** Add bundle records, confidence masks, unchanged anchors, candidate fitting, validation and rollback across `scene_runs/models.py`, `store.py`, `runner.py`, `gpu_worker.py`, and `lrz_transport.py`. Begin with hand-selected translated views and frozen geometry. Exit: repeatable unseen-view gain on one local region with no preservation regression.

**Milestone 2 — coherent targets and geometry.** Extend `image_edit.py` and provider adapters; benchmark CliRelay and one specialist multi-view restoration model. Add target-consistency rejection, constrained geometry updates and depth-supported insertion. Exit: a geometry-defective region improves across translated views without new floaters or doubled surfaces.

**Milestone 3 — active planning and repeated repairs.** Extend agent actions and navigation memory for region proposals and batch acquisition. Begin with a deterministic coverage planner, then compare VLM-assisted ranking and uncertainty methods. Add persistent replay and several accepted transactions. Exit: aggregate quality remains stable or improves across loop steps; unsuccessful proposals are rejected rather than accumulated.

**Milestone 4 — reconstruction research.** Compare local replacement with in-place refinement, and only then evaluate a global rebuild. Benchmark full scenes and competing methods with matched inputs and compute. This is where a state-of-the-art claim becomes testable.

Recommended ablations, changing one factor at a time:

1. Original asset; current decoded scene; full-SH no-op scene — isolate representation loss.
2. Current single-view Qwen versus single-view GPT, same poses and schedule — isolate editor effects.
3. Joint fitting of independent edits versus consistent bundle edits, same anchor budget — isolate target consistency.
4. Edited views only versus edited views plus masked unchanged anchors — isolate preservation.
5. Frozen geometry versus constrained geometry; no densification versus supported densification.
6. Source SH frozen versus progressively released versus unrestricted — isolate appearance overfitting.
7. Random/coverage camera selection versus agent-assisted selection, same view and inference budgets.
8. In-place refinement versus local replacement, on the same defective regions.

Use at least three representative scenes for an initial pilot: textured diffuse structure, reflective/thin structure, and an under-observed region. Repeat stochastic generation (for example, three samples per condition) and report variability. Expand to official dataset splits and baseline protocols before claiming state of the art. Suitable comparison families include ScanNet++/Replica for indoor repair and Mip-NeRF 360/DL3DV for sparse-view synthesis, with the exact split and capture availability reported.

## Acceptance, evaluation, and “finished”

Evaluate the **native 3DGS render**, with postprocessed output reported separately. Where held-out real or synthetic ground truth exists, report PSNR, SSIM, LPIPS, and geometry error against independent depth/mesh data. Also report boundary continuity, disocclusion holes, floaters, opacity/scale growth, object identity, and visibility-aware temporal flicker. Optical flow or reprojection must account for occlusion; specular changes are not automatically artifacts.

For assets without ground truth, combine blinded path comparisons, preservation metrics outside repair masks, feature-track/depth consistency, and explicit completion confidence. These are proxies: agreement with the original proves preservation, not correctness. A VLM quality score or no-reference image score alone cannot establish 3D improvement.

Define thresholds on a development set before the main comparison. As provisional pilot guardrails, consider rejecting an anchor-region PSNR drop over 0.2 dB or LPIPS increase over 0.01, where those references are meaningful; calibrate these to renderer repeatability, scene content, and human review. Require local improvement beyond stochastic variability plus no new structural failure. Assess worst affected regions as well as averages. Use separate validation views for repeated promotion decisions and a locked final test trajectory to avoid selection leakage.

Record accepted repairs per GPU-hour, image-inference cost, peak memory, latency, splat count, and rejection rate alongside quality. A conservative method can improve accepted-scene reliability while rejecting many proposals; report both.

A run is finished when the intended camera envelope and salient surfaces are sufficiently covered, validation shows no material unresolved defect within that envelope, and additional proposals fail to yield worthwhile improvement under the budget. Unobserved areas remain explicitly uncertain. Deadline expiry is a stopped run, not proof of a finished scene.

**Immediate next action:** implement representation preservation, renderer parity, and a fixed multi-view evaluation bundle before restoring autonomous cumulative repair. This makes the subsequent GPT-versus-Qwen and multi-view experiments interpretable.
