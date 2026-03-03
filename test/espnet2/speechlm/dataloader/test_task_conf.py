"""Tests for espnet2/speechlm/dataloader/task_conf.py — self-consistent validation."""

from espnet2.speechlm.dataloader.task_conf import (
    SUPPORTED_ENTRIES,
    TASK_CONFIGS,
    _validate_task_configs,
)


def test_supported_entries_is_nonempty():
    assert len(SUPPORTED_ENTRIES) > 0


def test_supported_entries_no_duplicates():
    assert len(SUPPORTED_ENTRIES) == len(set(SUPPORTED_ENTRIES))


def test_task_configs_is_nonempty():
    assert len(TASK_CONFIGS) > 0


def test_all_task_entries_are_supported():
    for task_name, config in TASK_CONFIGS.items():
        for entry in config.get("required_entries", []):
            assert (
                entry in SUPPORTED_ENTRIES
            ), f"Entry '{entry}' in task '{task_name}' not in SUPPORTED_ENTRIES"


def test_all_tasks_have_required_entries():
    for task_name, config in TASK_CONFIGS.items():
        assert (
            "required_entries" in config
        ), f"Task '{task_name}' missing 'required_entries'"
        assert (
            len(config["required_entries"]) > 0
        ), f"Task '{task_name}' has empty 'required_entries'"


def test_validate_task_configs_no_error():
    _validate_task_configs()
