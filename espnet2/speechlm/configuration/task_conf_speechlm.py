"""SpeechLM-specific task configuration module."""

from espnet2.speechlm.configuration.task_conf import (
    SUPPORTED_ENTRIES,
    TASK_CONFIGS,
)

VALID_ROLES = ["assistant", "user", "system"]


# Task template definitions for SpeechLM tasks
SPEECHLM_TASK_TEMPLATES = {
    "text_to_audio": [("user", "text1"), ("assistant", "audio1")],
    "audio_to_text": [("user", "audio1"), ("assistant", "text1")],
    "text_only": [("assistant", "text1")],
}


# Sanity check: ensure all entries in SPEECHLM_TASK_TEMPLATES are supported
def _validate_task_templates():
    """Validate entries and roles in SPEECHLM_TASK_TEMPLATES."""
    for task_name, template in SPEECHLM_TASK_TEMPLATES.items():
        for role, entry in template:
            if role not in VALID_ROLES:
                raise ValueError(
                    f"Invalid role '{role}' in task '{task_name}'. "
                    f"Must be one of: {VALID_ROLES}"
                )
            if entry not in SUPPORTED_ENTRIES:
                raise ValueError(
                    f"Invalid entry '{entry}' in task '{task_name}'. "
                    f"Must be one of: {SUPPORTED_ENTRIES}"
                )


def _validate_task_consistency():
    """Validate that SPEECHLM_TASK_TEMPLATES entries match TASK_CONFIGS."""
    for task_name, template in SPEECHLM_TASK_TEMPLATES.items():
        # Extract entries from template
        template_entries = set(entry for role, entry in template)

        # Get required entries from TASK_CONFIGS
        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Task '{task_name}' in SPEECHLM_TASK_TEMPLATES not "
                f"found in TASK_CONFIGS"
            )

        config_entries = set(TASK_CONFIGS[task_name]["required_entries"])

        # Check if entries match
        if template_entries != config_entries:
            raise ValueError(
                f"Entries mismatch for task '{task_name}': "
                f"SPEECHLM_TASK_TEMPLATES has {sorted(template_entries)}, "
                f"TASK_CONFIGS has {sorted(config_entries)}"
            )


_validate_task_templates()
_validate_task_consistency()
