"""Small, serializable controls; executable paths come from server config only."""
from __future__ import annotations
import math
from pathlib import Path

MODEL_VARIANTS = {
    "14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}
DEFAULT_MODEL_VARIANT = "1.3b"

DEFAULTS = {"frames": 25, "span_fraction": 0.04, "fit_iterations": 15000,
            "model_variant": DEFAULT_MODEL_VARIANT,
            "fitting_safeguards": True,
            "source_conditioning": "rendered", "generated_cache": "last_denoising",
            "anchor_prefit": "none", "block_schedule": "periodic_starter",
            "inference_steps": 4, "seed": 42, "camera_scale": 1.0,
            "max_repair_pixels": 960 * 720, "local_view_count": 5, "local_max_turns": 30,
            "repair_limit": 0, "local_candidate_count": 10}
# 4:3 at 720 lines, matching the paper's stated resolution range.
# Both dimensions are VAE-aligned; preserve the exploration camera aspect ratio.
REPAIR_WIDTH = 960
REPAIR_HEIGHT = 720
RUNTIME_DEFAULTS = {
    "repo": "/workspace/third_party/ArtiFixer",
    "python": "/workspace/artifixer-venv/bin/python",
    "hf_home": "/workspace/models/huggingface",
    "checkpoint": f"/workspace/models/artifixer/artifixer-{DEFAULT_MODEL_VARIANT}.pt",
    "model_id": MODEL_VARIANTS[DEFAULT_MODEL_VARIANT],
}
UPSTREAM_REVISION = "a392c4dfe17459ef9952407accdb9fcdcdddba98"


def validate_model_variant(value):
    if not isinstance(value, str) or value not in MODEL_VARIANTS:
        raise ValueError("ArtiFixer model_variant must be 14b or 1.3b")
    return value


def model_runtime(model_variant=DEFAULT_MODEL_VARIANT, overrides=None):
    """Select a matched release pair, retaining server-configured directories."""
    variant = validate_model_variant(model_variant)
    cfg = {**RUNTIME_DEFAULTS, **(overrides or {})}
    cfg.update(model_variant=variant, model_id=MODEL_VARIANTS[variant],
               checkpoint=str(Path(cfg["checkpoint"]).with_name(f"artifixer-{variant}.pt")))
    return cfg


def prepared_model_variant(marker):
    """Read old setup markers by base model ID, never assume the new default."""
    runtime = (marker or {}).get("artifixer_runtime") or {}
    for variant, model_id in MODEL_VARIANTS.items():
        if runtime.get("model_id") == model_id:
            return variant
    return None


def validate_options(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("extended must be an object")
    # Ignore controls from the superseded micro-step collector in saved runs.
    value = {k:v for k,v in (value or {}).items()
             if k not in {"local_step_fraction", "local_rotation_degrees"}}
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown extended options: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **(value or {})}
    validate_model_variant(result["model_variant"])
    if type(result["fitting_safeguards"]) is not bool:
        raise ValueError("fitting_safeguards must be a boolean")
    for key, choices in (("source_conditioning", ("none", "rendered")),
                         ("generated_cache", ("clean", "last_denoising")),
                         ("block_schedule", ("exact_starter", "periodic_starter", "upstream")),
                         ("anchor_prefit", ("none", "gsfix3d"))):
        if result[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    # Upstream keeps its final denoising cache; clean refresh is adapter-only.
    if result['block_schedule'] == 'upstream':
        result['generated_cache'] = 'last_denoising'
    if result['block_schedule'] == 'periodic_starter' and result['source_conditioning'] != 'rendered':
        raise ValueError('Periodic starter requires rendered source conditioning and opacity')
    if result['anchor_prefit'] != 'none' and result['source_conditioning'] != 'rendered':
        raise ValueError('anchor_prefit requires rendered source conditioning')
    # Long fitting runs are valid; the optimizer has no fixed iteration ceiling.
    if type(result["fit_iterations"]) is not int or result["fit_iterations"] < 1:
        raise ValueError("fit_iterations must be a positive integer")
    for key, lower, upper in [("frames", 9, 81),
                              ("inference_steps", 1, 50), ("seed", 0, 2**31-1),
                              ("max_repair_pixels", 0, 16777216), ("local_view_count", 5, 9),
                              ("local_candidate_count", 5, 10), ("local_max_turns", 5, 100), ("repair_limit", 0, 100)]:
        n = result[key]
        if isinstance(n, bool) or not isinstance(n, int) or not lower <= n <= upper:
            raise ValueError(f"{key} must be an integer between {lower} and {upper}")
    if (result["frames"] - 1) % 4:
        raise ValueError("frames must be 1 + 4*n (e.g. 25)")
    for key, lower, upper in [("span_fraction", .005, .15), ("camera_scale", .0001, 10000)]:
        n = result[key]
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or not lower <= n <= upper:
            raise ValueError(f"{key} must be between {lower} and {upper}")
        result[key] = float(n)
    return result


def validate_repair_resolution(width, height, repair_width, repair_height):
    """Image-model resolution. Zero keeps the VLM frame for that axis pair."""
    if repair_width == 0 and repair_height == 0:
        return 0, 0
    for name, value, lower, upper in (
        ("repair_width", repair_width, 64, 3840),
        ("repair_height", repair_height, 64, 2880),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    if repair_width % 16 or repair_height % 16:
        raise ValueError("Image repair width and height must be multiples of 16")
    if int(repair_width) * int(height) != int(repair_height) * int(width):
        raise ValueError(
            "Image repair resolution must keep the same aspect ratio as the VLM resolution"
        )
    return int(repair_width), int(repair_height)


def proposal(action):
    args = action.args or {}
    scope = args.get("repair_scope", "local")
    steps = args.get("view_steps", [])
    if scope not in ("local", "scene"):
        raise ValueError("repair_scope must be local or scene")
    if (not isinstance(steps, list) or len(steps) > 8
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in steps)):
        raise ValueError("view_steps must contain at most eight observed nonnegative step IDs")
    anchor = args.get("anchor_step")
    if anchor is not None and (type(anchor) is not int or anchor < 0):
        raise ValueError("anchor_step must be an observed nonnegative step ID")
    return {
            **({"anchor_step": anchor} if anchor is not None else {}),
            "description": str(args.get("description") or "Improve visible rendering artifacts")[:2000],
            "image_region": str(args.get("image_region") or "current view")[:300],
            "repair_scope": scope, "view_steps": list(dict.fromkeys(steps))}


def edit_prompt(action):
    p = proposal(action)
    return ("Repair this rendering of a static 3D scene. "
            "Repair fuzzy or inconsistent structure and appearance while preserving the scene's layout and objects. "
            "Keep the exact camera viewpoint, perspective, framing and image dimensions. "
            "Preserve unaffected content. Do not add objects or change lighting. "
            f"Observed defect: {p['description']}. "
            f"The target was initially identified at {p['image_region']} in the discovery view. "
            "In this different view, locate that same physical object; its screen position may "
            "have changed. Repair that object rather than treating the initial region as a fixed pixel mask.")


def configure_policy(policy):
    """Use the original v3 artifact hunter unchanged for the outer loop."""
    from ..tasks import artifact_hunt_3
    from ..agent.actions import filter_tools
    if hasattr(policy, "_task"):
        policy._task = artifact_hunt_3
    if hasattr(policy, "_tools"):
        policy._tools = filter_tools(artifact_hunt_3.HIDDEN_TOOLS)
