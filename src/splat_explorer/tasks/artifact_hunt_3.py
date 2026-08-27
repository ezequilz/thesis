"""Task prompt v3: short inspector. The harness ends the run at max_steps.

`done` is omitted from the tool list so the model cannot finish early.
Same public surface as v1/v2 plus HIDDEN_TOOLS. Switch via agent.prompt.
"""

# Not offered to the VLM; the episode loop still stops at max_steps.
HIDDEN_TOOLS = ("done",)

_IMAGES_WITH_DEPTH = """After every action you receive a fresh RGB view of the \
scene. This turn also includes a DEPTH MAP of the exact same view — bright = \
near, dark = far, black = no geometry at all (background, or a hole in the \
reconstruction).
"""

_IMAGES_RGB_ONLY = """After every action you receive one fresh image \
rendered from your current camera pose: the RGB view of the scene.
"""

_DEPTH_HELP = """The depth map helps: floaters appear as small bright patches much nearer than
their surroundings, holes as black pixels inside otherwise solid surfaces.
"""

_MAP_ATTACHED = """This turn also includes a BIRD'S-EYE MAP of the scene (ceiling \
removed): the connected red line is the path you have walked, each numbered \
marker is a past camera position, and each triangle is the camera frustum \
(viewing direction) at that step. Cyan highlights your CURRENT pose. \
move_toward pixel coordinates still refer to the RGB view, not the map.
"""

_COVERAGE_ATTACHED = """This turn also includes a COVERAGE MAP of the scene (ceiling \
removed): yellow-green paint is floor you have looked at from close range. \
Each past view adds a large cone that is strongest nearby and fades with \
distance, so the far side of a room stays faint. Overlapping views stack \
(still translucent). The number on the image is viewed-area coverage \
(0-100%). Unshaded rooms have not been visited. move_toward pixel coordinates \
still refer to the RGB view, not the coverage map.
"""


def system_prompt(
    with_depth: bool = True,
    with_map: bool = False,
    with_coverage: bool = False,
) -> str:
    """Minimal task text; tool schemas are supplied separately by the loop."""
    extras = ""
    if with_map:
        extras += _MAP_ATTACHED
    if with_coverage:
        extras += _COVERAGE_ATTACHED
    return f"""\
Inspect this 3D Gaussian Splatting indoor scene. Report real rendering \
artifacts (holes, floaters, mush, stretched splats, ghosting) once each, \
then keep moving. Unusual objects are not artifacts. \
{_IMAGES_WITH_DEPTH if with_depth else _IMAGES_RGB_ONLY}\
{extras}\
{_DEPTH_HELP if with_depth else ""}\
Walk with move_toward toward a pixel in the RGB view (amount 0..1). Use \
move for small steps, rotate to look around, and view_depth / view_map / \
view_coverage_map when useful. Pixel coordinates always refer to RGB. \
Call exactly one tool per turn. Keep exploring every accessible room.
"""


SYSTEM_PROMPT = system_prompt(False)


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
