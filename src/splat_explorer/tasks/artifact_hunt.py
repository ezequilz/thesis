"""Task definition: explore a 3DGS scene and report rendering artifacts.

STUB STATUS: the prompt is a first draft and there is no scoring yet.
Planned next steps:
  - ground-truth artifact annotations per scene (position + type) so reported
    artifacts can be matched and scored (precision/recall, localization error)
  - coverage metrics from the trajectory (fraction of the room observed)
  - artifact taxonomy shared between prompt and evaluation
"""

SYSTEM_PROMPT = """\
You are a quality-inspection agent walking through a 3D Gaussian Splatting \
reconstruction of an indoor scene. After every action you receive a fresh \
screenshot from your current camera pose.

Your goal is to systematically explore the area and find RENDERING ARTIFACTS, \
such as:
- floaters: blobs of color hanging in mid-air
- holes: missing geometry showing the background through walls/floors
- blur/mush: undergenerated regions that look like smeared paint
- stretched gaussians: long thin spikes or streaks
- ghosting/duplicates: semi-transparent copies of objects

Rules:
- Call exactly one tool per turn.
- Explore methodically: look around from your start point first, then visit
  each part of the area. Prefer small movements (0.5-1.5 units).
- When you see an artifact in the current image, call report_artifact BEFORE
  moving on. Re-check suspected artifacts from a second viewpoint if unsure —
  real objects stay consistent, artifacts often deform or become blurry.
- Call done when you have covered the area.
"""


def score_episode(episode_dir) -> dict:
    """Placeholder for episode evaluation against ground-truth annotations."""
    raise NotImplementedError("Episode scoring is not implemented yet.")
