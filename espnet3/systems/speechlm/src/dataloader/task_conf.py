# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Task configuration definitions for multimodal SpeechLM data loading."""


import math


def parse_data_specifier(specifier: str, *, registered: bool) -> tuple:
    """Return task, dataset name, optional manifest, and sampling factor."""
    parts = specifier.split(":")
    required = 2 if registered else 3
    if len(parts) not in (required, required + 1):
        kind = "registered" if registered else "unregistered"
        syntax = "task:name[:factor]" if registered else "task:name:data_json[:factor]"
        raise ValueError(
            f"Invalid {kind} specifier '{specifier}'. Expected format: '{syntax}'"
        )
    if any(not name or "/" in name for name in parts[:2]):
        raise ValueError(f"Invalid task or dataset name in {specifier!r}")
    factor = float(parts[-1]) if len(parts) > required else 1.0
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"Expected a finite positive factor in {specifier!r}")
    return (*parts[:required], factor)


# List of supported entries for task configurations
SUPPORTED_ENTRIES = (
    [f"audio{i}" for i in range(1, 11)]  # audio1 to audio10
    + [f"text{i}" for i in range(1, 11)]  # text1 to text10
    + ["dialogue", "speaker"]
)


# Centralized task configuration mapping
TASK_CONFIGS = {
    "text_to_audio": {
        "required_entries": ["audio1", "text1"],
    },
    "audio_to_text": {
        "required_entries": ["audio1", "text1"],
    },
    "text_only": {
        "required_entries": ["text1"],
    },
    "dialogue": {
        "required_entries": ["dialogue"],
    },
}


# Sanity check: ensure all entries in TASK_CONFIGS are supported
def _validate_task_configs():
    """Validate that all entries in TASK_CONFIGS are in SUPPORTED_ENTRIES."""
    for task_name, config in TASK_CONFIGS.items():
        for entry in config.get("required_entries", []):
            if entry not in SUPPORTED_ENTRIES:
                raise ValueError(
                    f"Invalid entry '{entry}' in task '{task_name}'. "
                    f"Must be one of: {SUPPORTED_ENTRIES}"
                )


_validate_task_configs()
