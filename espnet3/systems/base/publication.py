"""Publication helpers for the base system."""

from __future__ import annotations


def get_pack_model_artifacts(system) -> dict:
    """Return pack-model artifacts for the base system.

    Args:
        system: ESPnet3 system instance.

    Returns:
        dict: Empty artifact dict with keys ``files``, ``yaml_files``,
            and ``copy_paths``.

    Examples:
        >>> artifacts = get_pack_model_artifacts(system)
        >>> artifacts["files"]
        {}
    """
    return {"files": {}, "yaml_files": {}, "copy_paths": []}
