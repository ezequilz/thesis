"""A small calibrated translated loop around the selected anchor."""
from __future__ import annotations
import numpy as np
from ..rendering.base import Camera


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
