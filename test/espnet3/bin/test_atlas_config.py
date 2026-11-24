"""Tests for Atlas config helpers."""

from __future__ import annotations

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.config import split_patch_by_search_space


def _space(allowed=None, denied=None):
    return OmegaConf.create(
        {
            "allowed_keys": allowed if allowed is not None else [],
            "denied_keys": denied if denied is not None else [],
        }
    )


def test_keys_outside_the_allowed_globs_are_removed():
    kept, removed = split_patch_by_search_space(
        {"optimizer.lr": 0.1, "trainer.max_epochs": 5, "model.dropout": 0.1},
        _space(allowed=["optimizer.*", "model.*"]),
    )

    assert kept == {"optimizer.lr": 0.1, "model.dropout": 0.1}
    assert removed == ["trainer.max_epochs"]


def test_a_denied_glob_wins_over_an_allowed_one():
    kept, removed = split_patch_by_search_space(
        {"trainer.accumulate_grad_batches": 2, "trainer.max_epochs": 5},
        _space(allowed=["trainer.*"], denied=["trainer.max_epochs"]),
    )

    assert kept == {"trainer.accumulate_grad_batches": 2}
    assert removed == ["trainer.max_epochs"]


def test_without_an_allowed_list_everything_not_denied_is_kept():
    kept, removed = split_patch_by_search_space(
        {"a.b": 1, "dataloader.x": 2}, _space(denied=["dataloader.*"])
    )

    assert kept == {"a.b": 1}
    assert removed == ["dataloader.x"]


def test_a_missing_search_space_keeps_the_whole_patch():
    kept, removed = split_patch_by_search_space({"a": 1}, OmegaConf.create({}))

    assert (kept, removed) == ({"a": 1}, [])


def test_keys_are_matched_as_strings_and_values_are_untouched():
    kept, removed = split_patch_by_search_space(
        {7: {"nested": [1, 2]}}, _space(allowed=["7"])
    )

    assert kept == {"7": {"nested": [1, 2]}} and removed == []
