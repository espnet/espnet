"""Tests for espnet2/speechlm/model/speechlm/task_conf_speechlm.py."""

from espnet2.speechlm.dataloader.task_conf import SUPPORTED_ENTRIES, TASK_CONFIGS
from espnet2.speechlm.model.speechlm.task_conf_speechlm import (
    SPEECHLM_TASK_CONFIGS,
    VALID_ROLES,
    _validate_task_consistency,
    _validate_task_templates,
)


def test_speechlm_task_configs_nonempty():
    assert len(SPEECHLM_TASK_CONFIGS) > 0


def test_valid_roles():
    assert "assistant" in VALID_ROLES
    assert "user" in VALID_ROLES
    assert "system" in VALID_ROLES


def test_all_roles_valid():
    for task_name, template in SPEECHLM_TASK_CONFIGS.items():
        for role, entry in template:
            assert role in VALID_ROLES, f"Invalid role '{role}' in task '{task_name}'"


def test_all_entries_in_supported():
    for task_name, template in SPEECHLM_TASK_CONFIGS.items():
        for role, entry in template:
            assert (
                entry in SUPPORTED_ENTRIES
            ), f"Invalid entry '{entry}' in task '{task_name}'"


def test_task_consistency_with_task_configs():
    for task_name, template in SPEECHLM_TASK_CONFIGS.items():
        template_entries = set(entry for role, entry in template)
        assert task_name in TASK_CONFIGS, f"Task '{task_name}' not in TASK_CONFIGS"
        config_entries = set(TASK_CONFIGS[task_name]["required_entries"])
        assert (
            template_entries == config_entries
        ), f"Entries mismatch for task '{task_name}'"


def test_validate_task_templates_no_error():
    _validate_task_templates()


def test_validate_task_consistency_no_error():
    _validate_task_consistency()


def test_template_format():
    for task_name, template in SPEECHLM_TASK_CONFIGS.items():
        assert isinstance(template, list), f"Template for '{task_name}' is not a list"
        for item in template:
            assert isinstance(item, tuple), f"Item in '{task_name}' is not a tuple"
            assert len(item) == 2, f"Item in '{task_name}' does not have 2 elements"
            role, entry = item
            assert isinstance(role, str)
            assert isinstance(entry, str)
