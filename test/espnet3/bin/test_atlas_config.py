"""Tests for Atlas config helpers."""

from __future__ import annotations

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.utils import split_patch_by_search_space


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


def _write(tmp_path, text):
    path = tmp_path / "anywhere" / "study.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_study_config_is_merged_over_the_atlas_defaults_wherever_it_lives(tmp_path):
    from espnet3.bin.atlas.core.config import load_atlas_config

    config = load_atlas_config(
        _write(
            tmp_path,
            "atlas:\n  study_name: s\n  budget: {max_trials: 3}\n"
            "  submission: {resources: {}}\n",
        )
    )

    assert config.atlas.study_name == "s"
    assert config.atlas.budget.max_trials == 3
    assert config.atlas.budget.max_parallel_trials == 1  # from the defaults
    assert config.atlas.tick_interval_sec_trial_active == 600
    assert config.atlas.agent.provider == "codex"


def test_a_study_config_overrides_the_defaults(tmp_path):
    from espnet3.bin.atlas.core.config import load_atlas_config

    config = load_atlas_config(
        _write(
            tmp_path,
            "atlas:\n  tick_interval_sec_idle: 5\n  agent: {provider: claude}\n"
            "  submission: {resources: {}}\n",
        )
    )

    assert config.atlas.tick_interval_sec_idle == 5
    assert config.atlas.agent.provider == "claude"


def test_submission_has_no_default(tmp_path):
    import pytest

    from espnet3.bin.atlas.core.config import load_atlas_config

    with pytest.raises(ValueError, match="submission is required"):
        load_atlas_config(_write(tmp_path, "atlas: {study_name: s}\n"))


def test_the_controller_runs_in_local_or_queue(tmp_path):
    import pytest

    from espnet3.bin.atlas.core.config import load_atlas_config

    with pytest.raises(ValueError, match="run_in must be 'local' or 'queue'"):
        load_atlas_config(
            _write(
                tmp_path / "a",
                "atlas: {submission: {controller: {run_in: slurm}}}\n",
            )
        )
    with pytest.raises(ValueError, match="instructions is required"):
        load_atlas_config(
            _write(
                tmp_path / "b",
                "atlas: {submission: {controller: {run_in: queue}}}\n",
            )
        )


def test_pairs_become_a_dict_and_a_repeated_key_keeps_the_last_value():
    from espnet3.bin.atlas.core.utils import convert_pairs_to_dict

    pairs = [
        {"key": "optimizer.lr", "value": 0.1},
        {"key": "model.dims", "value": [1, 2]},
        {"key": "optimizer.lr", "value": 0.2},
    ]

    assert convert_pairs_to_dict(pairs) == {"optimizer.lr": 0.2, "model.dims": [1, 2]}
    assert convert_pairs_to_dict([]) == {}


def test_dotted_patch_sets_nested_keys_and_indexes_into_lists():
    import pytest
    from omegaconf import OmegaConf

    from espnet3.bin.atlas.core.utils import apply_dotted_patch

    cfg = OmegaConf.create(
        {
            "trainer": {"lr": 0.1, "alias": "${trainer.lr}"},
            "dataset": {"train": [{"t": {"p": [1, 2]}}, {"u": 1}]},
            "d": {"a": 1, "b": 2},
        }
    )

    apply_dotted_patch(
        cfg,
        {
            "trainer.lr": 0.2,
            "new.deep.key": 5,
            "dataset.train.1.u": 7,
            "dataset.train.0.t.p": [3],
            "d": {"c": 3},
        },
    )

    plain = OmegaConf.to_container(cfg, resolve=False)
    assert plain["trainer"] == {"lr": 0.2, "alias": "${trainer.lr}"}
    assert plain["new"] == {"deep": {"key": 5}}
    assert plain["dataset"]["train"] == [{"t": {"p": [3]}}, {"u": 7}]
    assert plain["d"] == {"c": 3}  # replaced, not merged
    with pytest.raises(Exception, match="out of range"):
        apply_dotted_patch(cfg, {"dataset.train.5.u": 1})
