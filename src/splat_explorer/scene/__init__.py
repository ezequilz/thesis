from .catalog import SceneSpec, list_scenes
from .ply_loader import load_ply, save_ply
from .sog_loader import load_sog, load_sog_lod
from .types import GaussianScene

__all__ = [
    "GaussianScene",
    "SceneSpec",
    "list_scenes",
    "load_sog",
    "load_sog_lod",
    "load_ply",
    "save_ply",
    "load_scene",
]


def load_scene(path, min_opacity: float = 0.0, lod_level: int = 0) -> GaussianScene:
    """Load a Gaussian splat scene from disk, dispatching on format.

    Supported:
      - bundled SOG v2 (``.sog`` zip)
      - unbundled SOG v2 (directory or ``meta.json``)
      - streamed SOG (directory with ``lod-meta.json``, or the file itself)
      - standard 3DGS ``.ply``
    """
    from pathlib import Path

    path = Path(path)
    if path.is_file() and path.name == "lod-meta.json":
        scene = load_sog_lod(path.parent, lod_level=lod_level)
    elif path.is_file() and path.name == "meta.json":
        scene = load_sog(path.parent)
    elif path.is_dir():
        if (path / "lod-meta.json").is_file():
            scene = load_sog_lod(path, lod_level=lod_level)
        elif (path / "meta.json").is_file():
            scene = load_sog(path)
        else:
            raise ValueError(
                f"Directory is not a SOG scene (no lod-meta.json or meta.json): {path}"
            )
    elif path.suffix.lower() == ".sog":
        scene = load_sog(path)
    elif path.suffix.lower() == ".ply":
        scene = load_ply(path)
    else:
        raise ValueError(f"Unsupported splat format: {path.suffix or path}")

    if min_opacity > 0:
        scene = scene.filtered_by_opacity(min_opacity)
    return scene
