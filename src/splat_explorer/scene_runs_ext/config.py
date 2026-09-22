"""Small, serializable controls; executable paths come from server config only."""
from __future__ import annotations
import math

DEFAULTS = {"frames": 25, "span_fraction": 0.04, "fit_iterations": 200,
            "inference_steps": 4, "seed": 42, "camera_scale": 1.0}
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
    unknown = set(value or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown extended options: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **(value or {})}
    for key, lower, upper in [("frames", 9, 81), ("fit_iterations", 1, 2000),
                              ("inference_steps", 1, 50), ("seed", 0, 2**31-1)]:
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


def proposal(action):
    args = action.args or {}
    intervention = str(args.get("intervention") or "structure")
    if intervention not in {"appearance", "structure"}:
        intervention = "structure"
    return {"intervention": intervention,
            "description": str(args.get("description") or "Improve visible rendering artifacts")[:2000],
            "image_region": str(args.get("image_region") or "current view")[:300]}


def edit_prompt(action):
    p = proposal(action)
    intent = ("Improve texture, color and sharpness without changing object shape."
              if p["intervention"] == "appearance" else
              "Repair fuzzy or inconsistent structure while preserving the scene's layout and objects.")
    return (f"Repair this rendering of a static 3D scene. {intent} "
            "Keep the exact camera viewpoint, perspective, framing and image dimensions. "
            "Preserve unaffected content. Do not add objects or change lighting. "
            f"Region: {p['image_region']}. Observed defect: {p['description']}")


def configure_policy(policy):
    """Extend this policy instance, leaving baseline/global tool schemas intact."""
    import copy
    if not hasattr(policy, "_tools"):
        return
    policy._tools = copy.deepcopy(policy._tools)
    for tool in policy._tools:
        fn = tool["function"]
        if fn["name"] == "report_artifact":
            fn["description"] = (
                "Select a visible region for repair. regenerate=yes pauses exploration, edits "
                "this anchor, propagates it across nearby calibrated views with ArtiFixer, "
                "and fits an updated scene before continuing. Choose an intervention."
            )
            fn["parameters"]["properties"]["intervention"] = {
                "type": "string", "enum": ["appearance", "structure"],
                "description": "appearance freezes geometry; structure also optimizes shape and opacity.",
            }
