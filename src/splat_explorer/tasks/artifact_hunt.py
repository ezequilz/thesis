"""Task definition: explore a 3DGS scene and report rendering artifacts.

STUB STATUS: the prompt is a first draft and there is no scoring yet.
Planned next steps:
  - ground-truth artifact annotations per scene (position + type) so reported
    artifacts can be matched and scored (precision/recall, localization error)
  - coverage metrics from the trajectory (fraction of the room observed)
  - artifact taxonomy shared between prompt and evaluation
"""

_IMAGES_WITH_DEPTH = """After every action you receive two fresh \
labelled images rendered from your current camera pose:
1. RGB view — what the scene looks like.
2. DEPTH MAP of the exact same view — bright = near, dark = far, black = no
   geometry at all (background, or a hole in the reconstruction).
"""

_IMAGES_RGB_ONLY = """After every action you receive one fresh image \
rendered from your current camera pose: the RGB view of the scene.
"""

_DEPTH_HELP = """The depth map helps: floaters appear as small bright patches much nearer than
their surroundings, holes as black pixels inside otherwise solid surfaces.
"""


def system_prompt(with_depth: bool = True) -> str:
    """Task prompt for the explore loop; wording adapts to whether the depth
    map is attached alongside the RGB view (agent.send_depth)."""
    depth_hint = (" Check the\n  depth map first: black pixels have nothing to move toward."
                  if with_depth else "")
    return f"""\
You are a quality-inspection agent walking through a 3D Gaussian Splatting \
reconstruction of an indoor scene. {_IMAGES_WITH_DEPTH if with_depth else _IMAGES_RGB_ONLY}
Your goal is to systematically explore the area and find RENDERING ARTIFACTS, \
such as:
- floaters: blobs of color hanging in mid-air
- holes: missing geometry showing the background through walls/floors
- blur/mush: undergenerated regions that look like smeared paint
- stretched gaussians: long thin spikes or streaks
- ghosting/duplicates: semi-transparent copies of objects
{_DEPTH_HELP if with_depth else ""}
IMPORTANT — not artifacts: semi-transparent materials (sheer curtains, glass,
foliage) legitimately render as layered translucent sheets and can look
streaky, ghostly, or scalloped, especially from very close up, where this
renderer exaggerates them. If the view is dominated by such translucency, move
back and re-check from a second, more distant viewpoint before reporting; only
report it if the anomaly persists from a normal viewing distance.

How to move:
- move_toward(pixel_x, pixel_y, amount) is your MAIN way to travel. Pick a
  pixel in the RGB view; you walk `amount` (0..1) of the ground-plane distance
  to the surface visible there, keeping your eye height (so a point on the
  floor walks you toward that spot, not down into it). amount=1.0 brings you
  right up to that location — you can never enter geometry, so prefer
  confident values like 0.5-0.8.{depth_hint}
- move(direction, distance) is only for small corrective steps (0.3-1.0 units).
- rotate(yaw_degrees, pitch_degrees) turns the camera; pitch is absolute
  (-85..85, positive = up, 0 = level).
- The harness blocks collisions. If your last movement was cut short, the next
  prompt tells you so — turn or pick a different direction instead of retrying.

Rules:
- Call exactly one tool per turn.
- Explore methodically: look around from your start point first (rotate in
  steps of 45-90 degrees), then visit each part of the area with move_toward.
- When you see an artifact in the current image, call report_artifact BEFORE
  moving on. Re-check suspected artifacts from a second viewpoint if unsure —
  real objects stay consistent, artifacts often deform or become blurry.
- Call done when you have covered the area.
"""


# Default prompt (depth attached) for callers that don't toggle it.
SYSTEM_PROMPT = system_prompt(True)


SPAWN_PROMPT = """\
You are a quality-inspection agent about to explore a 3D Gaussian Splatting \
reconstruction of an indoor scene to find rendering artifacts.

The attached image is a BIRD'S-EYE view of the scene with the ceiling removed \
so you can see the room layout. The bright magenta numbered dots are candidate \
starting positions, pre-computed to lie in open space with good visibility \
(their world coordinates and surrounding open space are listed below).

Study the layout and pick the starting point that promises the best \
exploration coverage of the whole area.
"""


def score_episode(episode_dir) -> dict:
    """Placeholder for episode evaluation against ground-truth annotations."""
    raise NotImplementedError("Episode scoring is not implemented yet.")
