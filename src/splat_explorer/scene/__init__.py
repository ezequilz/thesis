from .types import GaussianScene
from .sog_loader import load_sog

__all__ = ["GaussianScene", "load_sog", "load_scene"]


def load_scene(path, min_opacity: float = 0.0) -> GaussianScene:
    """Load a Gaussian splat scene from disk, dispatching on file extension."""
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() == ".sog":
        scene = load_sog(path)
    elif path.suffix.lower() == ".ply":
        raise NotImplementedError("PLY loading is planned but not implemented yet.")
    else:
        raise ValueError(f"Unsupported splat format: {path.suffix}")

    if min_opacity > 0:
        scene = scene.filtered_by_opacity(min_opacity)
    return scene
