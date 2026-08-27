"""Select which artifact-hunt prompt the agent loop uses.

Each variant is a self-contained module with the same public surface
(system_prompt, SYSTEM_PROMPT, SPAWN_PROMPT, score_episode). Add a new
file, register it here, and set agent.prompt (or DEFAULT_PROMPT) to try it.
Remove an inactive variant from PROMPT_VARIANTS, then delete the file.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

# Canonical names plus filename aliases. Values are import paths.
PROMPT_VARIANTS = {
    "v1": "splat_explorer.tasks.artifact_hunt",
    "artifact_hunt": "splat_explorer.tasks.artifact_hunt",
    "v2": "splat_explorer.tasks.artifact_hunt_2",
    "artifact_hunt_2": "splat_explorer.tasks.artifact_hunt_2",
}

# Active default when agent.prompt is unset. Flip to "v1" to restore the original.
DEFAULT_PROMPT = "v2"


def canonical_names() -> list[str]:
    """Unique short names (v1, v2, ...) in registration order."""
    seen: list[str] = []
    seen_mods: set[str] = set()
    for name, mod in PROMPT_VARIANTS.items():
        if mod in seen_mods:
            continue
        seen_mods.add(mod)
        seen.append(name)
    return seen


def load_prompt(name: str | None = None) -> ModuleType:
    """Import the prompt module for `name`, or DEFAULT_PROMPT if omitted."""
    key = (name or DEFAULT_PROMPT).strip()
    if key not in PROMPT_VARIANTS:
        known = ", ".join(canonical_names())
        raise ValueError(f"Unknown prompt {key!r}. Known variants: {known}")
    return import_module(PROMPT_VARIANTS[key])
