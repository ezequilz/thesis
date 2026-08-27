"""Task prompts and evaluation stubs.

The explore loop loads the active prompt through registry.load_prompt
(agent.prompt / DEFAULT_PROMPT). Direct imports from artifact_hunt or
artifact_hunt_2 still return that file's own variant.
"""

from .registry import DEFAULT_PROMPT, canonical_names, load_prompt

__all__ = ["DEFAULT_PROMPT", "canonical_names", "load_prompt"]
