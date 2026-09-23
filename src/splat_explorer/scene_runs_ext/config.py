"""Small, serializable controls; executable paths come from server config only."""
from __future__ import annotations
import math

DEFAULTS = {"frames": 25, "span_fraction": 0.04, "fit_iterations": 1000,
            "inference_steps": 4, "seed": 42, "camera_scale": 1.0,
            "max_repair_pixels": 0, "local_view_count": 5, "local_max_turns": 30,
            "repair_limit": 0}
# 4:3, 3× the 640×480 VLM default, and a multiple of 16 for the GPU rasterizer.
REPAIR_WIDTH = 1920
REPAIR_HEIGHT = 1440
RUNTIME_DEFAULTS = {
    "repo": "/workspace/third_party/ArtiFixer",
    "python": "/workspace/artifixer-venv/bin/python",
    "hf_home": "/workspace/models/huggingface",
    "checkpoint": "/workspace/models/artifixer/artifixer-1.3b.pt",
    "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}
UPSTREAM_REVISION = "a392c4dfe17459ef9952407accdb9fcdcdddba98"


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
    for key, lower, upper in [("frames", 9, 81), ("fit_iterations", 1, 2000),
                              ("inference_steps", 1, 50), ("seed", 0, 2**31-1),
                              ("max_repair_pixels", 0, 16777216), ("local_view_count", 5, 9),
                              ("local_max_turns", 5, 100), ("repair_limit", 0, 100)]:
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
    return {
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
