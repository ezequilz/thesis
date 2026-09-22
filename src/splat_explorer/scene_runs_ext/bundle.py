"""A small calibrated translated loop around the selected anchor."""
from __future__ import annotations
import numpy as np
import math
from dataclasses import replace
from ..rendering.base import Camera


def repair_camera(camera, image_size, *, max_pixels=0):
    """Largest exact-aspect, VAE-aligned camera no larger than the edited image.

    No cropping, stretching, or synthetic upsampling. FOV/pose stay fixed and
    Camera recomputes pixel intrinsics at the new resolution.
    """
    width, height = image_size
    if width * camera.height != height * camera.width:
        raise ValueError("Edited anchor aspect ratio changed; cannot assign the original calibrated camera")
    divisor = math.gcd(width, height)
    rw, rh = width // divisor, height // divisor
    unit = math.lcm(16 // math.gcd(rw, 16), 16 // math.gcd(rh, 16))
    count = divisor // unit
    if max_pixels:
        count = min(count, math.isqrt(max_pixels // (rw * rh * unit * unit)))
    if count < 1:
        raise ValueError("No exact-aspect multiple-of-16 resolution fits the repaired image/pixel budget")
    return replace(camera, width=rw * unit * count, height=rh * unit * count)


def selected_views(request_dir, proposal, current_step):
    """Resolve agent IDs against worker-owned observations, never model poses/paths."""
    import json
    from pathlib import Path
    from ..repair_lrz import camera_from_dict
    if proposal.get("repair_scope", "local") not in ("local", "scene"):
        raise ValueError("repair_scope must be local or scene")
    steps = proposal.get("view_steps", [])
    if (not isinstance(steps, list) or len(steps) > 8
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in steps)):
        raise ValueError("Invalid selected view_steps")
    result = []
    for step in dict.fromkeys(steps):
        if step == current_step:
            continue
        if current_step is None or step >= current_step:
            raise ValueError(f"Selected step {step} is not an earlier observed view")
        root = Path(request_dir).parent
        observed = root / f"render-{step:05d}" / "request.json"
        response = observed.with_name("response.json")
        if not observed.is_file() or not response.is_file():
            raise ValueError(f"Selected step {step} has no recorded GPU observation")
        body = json.loads(observed.read_text())
        if (body.get("operation") != "render" or body.get("step") != step
                or json.loads(response.read_text()).get("status") != "ok"):
            raise ValueError(f"Selected step {step} is not a completed observation")
        view = {"step": step, "camera": camera_from_dict(body["camera"])}
        bundled = Path(request_dir) / f"reference-{step:05d}.png"
        if bundled.is_file():
            view["reference_path"] = bundled
            result.append(view)
            continue
        edited = root / f"repair-{step:05d}" / "regenerated.png"
        edit_request = edited.with_name("request.json")
        if edited.is_file() and edit_request.is_file():
            edit_body = json.loads(edit_request.read_text())
            # Only reuse a reference registered at exactly this observed pose.
            if edit_body.get("camera") == body["camera"]:
                view["reference_path"] = edited
        result.append(view)
    if proposal.get("repair_scope", "local") == "scene" and not result:
        raise ValueError("Scene reconstruction requires earlier view_steps; explore relevant views first")
    return result


def camera_bundle(anchor, depth, *, frames=25, span_fraction=.04):
    depth = np.asarray(depth)
    h, w = depth.shape
    center = depth[h//4:3*h//4, w//4:3*w//4]
    valid = center[np.isfinite(center) & (center > 0)]
    if valid.size < 16:
        raise ValueError("Insufficient visible depth to build a local ArtiFixer trajectory; choose a supported view")
    distance = float(np.median(valid))
    radius = distance * float(span_fraction)
    target = anchor.position + anchor.rotation[:, 2] * distance
    result = []
    for angle in np.linspace(0, 2*np.pi, frames):
        offset = radius * (np.sin(angle)*anchor.rotation[:, 0]
                           - .25*(1-np.cos(angle))*anchor.rotation[:, 1])
        result.append(Camera.look_at(anchor.position + offset, target, -anchor.rotation[:, 1],
                                     width=anchor.width, height=anchor.height, fov_deg=anchor.fov_deg))
    # Exact anchor pose, including roll, and an explicit return to it.
    result[0] = anchor
    result[-1] = anchor
    return result


def transforms(cameras):
    """Official ArtiFixer camera helper consumes OpenCV c2w and pixel intrinsics."""
    c = cameras[0]
    return {"w": c.width, "h": c.height, "fl_x": c.fx, "fl_y": c.fy,
            "cx": c.width/2, "cy": c.height/2,
            "frames": [{"transform_matrix": x.c2w.tolist()} for x in cameras]}
