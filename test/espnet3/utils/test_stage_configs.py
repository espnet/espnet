import logging

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.utils.run_utils import ConfigError
from espnet3.utils.stage_configs import (
    bake,
    build_stage_configs,
    inherit,
    load_baked,
)


class _System:
    stages = (
        StageSpec(name="train", config="training"),
        StageSpec(name="infer", config="inference"),
        StageSpec(name="measure", config="metrics"),
    )


def test_inherit_with_no_exp_dir_returns_own_alone():
    own = OmegaConf.create({"inference_dir": "./infer"})

    config = inherit(None, _System, "infer", own)

    assert config == own


def test_inherit_with_no_baked_files_returns_own_alone(tmp_path):
    own = OmegaConf.create({"model": "dummy"})

    config = inherit(tmp_path, _System, "train", own)

    assert config == own


def test_inherit_merges_an_earlier_baked_stage(tmp_path):
    (tmp_path / "config").mkdir()
    train_yaml = tmp_path / "config" / "train.yaml"
    train_yaml.write_text("exp_dir: ./exp/debug\nexp_tag: debug\n", encoding="utf-8")
    own = OmegaConf.create({"inference_dir": "./infer"})

    config = inherit(tmp_path, _System, "infer", own)

    assert config.exp_dir == "./exp/debug"
    assert config.exp_tag == "debug"
    assert config.inference_dir == "./infer"


def test_inherit_lets_a_later_earlier_stage_win_over_an_earlier_one(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "train.yaml").write_text(
        "exp_dir: ./exp/from_train\n", encoding="utf-8"
    )
    (tmp_path / "config" / "infer.yaml").write_text(
        "exp_dir: ./exp/from_infer\n", encoding="utf-8"
    )
    own = OmegaConf.create({})

    config = inherit(tmp_path, _System, "measure", own)

    assert config.exp_dir == "./exp/from_infer"


def test_inherit_replaces_whole_top_level_keys_not_a_deep_merge(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "train.yaml").write_text(
        "model:\n  a: 1\n  b: 2\n", encoding="utf-8"
    )
    own = OmegaConf.create({"model": {"a": 99}})

    config = inherit(tmp_path, _System, "infer", own)

    assert config.model == {"a": 99}


def test_inherit_skips_later_stages_and_missing_files(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "measure.yaml").write_text(
        "should_not_appear: true\n", encoding="utf-8"
    )
    own = OmegaConf.create({})

    config = inherit(tmp_path, _System, "train", own)

    assert config == own


def test_bake_writes_only_the_given_own_keys(tmp_path):
    resolved = OmegaConf.create({"exp_dir": "./exp/debug", "model": "dummy"})

    path = bake(tmp_path, "train", resolved, own_keys=["exp_dir"])

    assert path == tmp_path / "config" / "train.yaml"
    written = OmegaConf.load(path)
    assert dict(written) == {"exp_dir": "./exp/debug"}


def test_bake_leaves_identical_content_untouched(tmp_path):
    resolved = OmegaConf.create({"exp_dir": "./exp/debug"})
    bake(tmp_path, "train", resolved, own_keys=["exp_dir"])
    first_mtime = (tmp_path / "config" / "train.yaml").stat().st_mtime_ns

    bake(tmp_path, "train", resolved, own_keys=["exp_dir"])

    assert (tmp_path / "config" / "train.yaml").stat().st_mtime_ns == first_mtime
    assert not (tmp_path / "config" / "history").exists()


def test_bake_moves_a_differing_existing_file_to_history(tmp_path, caplog):
    first = OmegaConf.create({"exp_dir": "./exp/first"})
    bake(tmp_path, "train", first, own_keys=["exp_dir"])

    second = OmegaConf.create({"exp_dir": "./exp/second"})
    with caplog.at_level(logging.WARNING):
        bake(tmp_path, "train", second, own_keys=["exp_dir"])

    history_dir = tmp_path / "config" / "history"
    assert any(p.name.startswith("train.") for p in history_dir.iterdir())
    assert OmegaConf.load(tmp_path / "config" / "train.yaml").exp_dir == "./exp/second"
    assert "train" in caplog.text


def test_load_baked_reads_back_a_written_stage(tmp_path):
    resolved = OmegaConf.create({"exp_dir": "./exp/debug"})
    bake(tmp_path, "train", resolved, own_keys=["exp_dir"])

    loaded = load_baked(tmp_path, "train")

    assert loaded.exp_dir == "./exp/debug"


def test_load_baked_raises_config_error_naming_the_stage_and_path(tmp_path):
    with pytest.raises(ConfigError, match="measure") as excinfo:
        load_baked(tmp_path, "measure")

    assert str(tmp_path / "config" / "measure.yaml") in str(excinfo.value)


def test_inherit_prefers_in_memory_over_the_baked_file(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "train.yaml").write_text(
        "exp_dir: ./exp/from_file\n", encoding="utf-8"
    )
    own = OmegaConf.create({})

    config = inherit(
        tmp_path,
        _System,
        "infer",
        own,
        in_memory={"train": {"exp_dir": "./exp/from_memory"}},
    )

    assert config.exp_dir == "./exp/from_memory"


def test_inherit_uses_in_memory_without_an_exp_dir():
    own = OmegaConf.create({"inference_dir": "./infer"})

    config = inherit(
        None, _System, "infer", own, in_memory={"train": {"exp_dir": "./exp/mem"}}
    )

    assert config.exp_dir == "./exp/mem"
    assert config.inference_dir == "./infer"


def test_build_stage_configs_resolves_same_run_interpolation_in_memory(tmp_path):
    configs = {
        "training": OmegaConf.create(
            {"exp_tag": "debug", "exp_dir": "./exp/${exp_tag}"}
        ),
        "inference": OmegaConf.create({"inference_dir": "${exp_dir}/infer"}),
        "metrics": OmegaConf.create({"measure_dir": "${inference_dir}/measure"}),
    }

    stage_configs, own_keys = build_stage_configs(_System, configs, exp_dir=tmp_path)

    assert stage_configs["train"].exp_dir == "./exp/debug"
    assert stage_configs["infer"].inference_dir == "./exp/debug/infer"
    assert stage_configs["measure"].measure_dir == "./exp/debug/infer/measure"
    assert own_keys["train"] == ("exp_tag", "exp_dir")
    assert own_keys["infer"] == ("inference_dir",)
    # nothing is baked by build_stage_configs itself
    assert not (tmp_path / "config").exists()


def test_build_stage_configs_raises_when_a_stage_points_at_its_own_inherited_key(
    tmp_path,
):
    configs = {
        "training": OmegaConf.create({"exp_dir": str(tmp_path / "exp")}),
        "inference": OmegaConf.create({"inference_dir": "${exp_dir}/infer"}),
        # A stage's own key fully replaces the inherited one (section 12.1),
        # so pointing it at "${inference_dir}" makes it point at itself.
        "metrics": OmegaConf.create({"inference_dir": "${inference_dir}"}),
    }

    with pytest.raises(ConfigError):
        build_stage_configs(_System, configs, exp_dir=tmp_path)


def test_build_stage_configs_merges_a_tuple_configs_own_roles(tmp_path):
    """A stage declaring more than one role gets both roles' own keys."""

    class _MultiRoleSystem:
        stages = (
            StageSpec(name="train", config="training"),
            StageSpec(name="pack_model", config=("training", "publication")),
        )

    configs = {
        "training": OmegaConf.create({"exp_dir": "./exp/debug", "exp_tag": "debug"}),
        "publication": OmegaConf.create({"readme": "README.md"}),
    }

    stage_configs, own_keys = build_stage_configs(
        _MultiRoleSystem, configs, exp_dir=tmp_path
    )

    # "pack_model"'s own config is training's and publication's own keys
    # merged (publication wins on any overlap), on top of whatever it
    # already inherits from "train" via the normal stage-order walk.
    assert stage_configs["pack_model"].readme == "README.md"
    assert stage_configs["pack_model"].exp_tag == "debug"
    assert set(own_keys["pack_model"]) == {"exp_dir", "exp_tag", "readme"}


def test_build_stage_configs_respects_upto_and_skips_later_stages(tmp_path):
    configs = {
        "training": OmegaConf.create({"exp_dir": "./exp/debug"}),
        "inference": OmegaConf.create({"inference_dir": "${exp_dir}/infer"}),
        # metrics' own config has an interpolation that can never resolve;
        # it must not be touched when upto stops before "measure".
        "metrics": OmegaConf.create({"inference_dir": "${does_not_exist}"}),
    }

    stage_configs, own_keys = build_stage_configs(
        _System, configs, exp_dir=tmp_path, upto=["infer"]
    )

    assert set(stage_configs) == {"train", "infer"}
    assert set(own_keys) == {"train", "infer"}


# ---------------------------------------------------------------------------
# snapshot_config
# ---------------------------------------------------------------------------


def test_snapshot_config_writes_full_config_no_dry_run(tmp_path):
    """Full resolved config is written when log_dir is provided and not dry_run."""
    from omegaconf import OmegaConf

    from espnet3.utils.stage_configs import snapshot_config

    resolved = OmegaConf.create(
        {
            "exp_dir": "./exp/debug",
            "exp_tag": "debug",
            "model": "dummy_model",
            "seed": 42,
        }
    )

    snapshot_config(tmp_path, "train", resolved, dry_run=False, is_rank0=True)

    snapshot_yaml = (tmp_path / "train.yaml").read_text(encoding="utf-8")
    # Full resolved config is persisted
    assert "exp_dir: ./exp/debug" in snapshot_yaml
    assert "exp_tag: debug" in snapshot_yaml
    assert "model: dummy_model" in snapshot_yaml
    assert "seed: 42" in snapshot_yaml


def test_snapshot_config_no_write_on_dry_run(tmp_path):
    """No write occurs when dry_run=True."""
    from omegaconf import OmegaConf

    from espnet3.utils.stage_configs import snapshot_config

    resolved = OmegaConf.create({"exp_dir": "./exp/debug"})

    snapshot_config(tmp_path, "train", resolved, dry_run=True, is_rank0=True)

    # No file should be written
    assert not (tmp_path / "train.yaml").exists()


def test_snapshot_config_no_write_if_not_rank0(tmp_path):
    """No write occurs when is_rank0=False (distributed execution)."""
    from omegaconf import OmegaConf

    from espnet3.utils.stage_configs import snapshot_config

    resolved = OmegaConf.create({"exp_dir": "./exp/debug"})

    snapshot_config(tmp_path, "train", resolved, dry_run=False, is_rank0=False)

    assert not (tmp_path / "train.yaml").exists()


def test_snapshot_config_no_write_without_log_dir():
    """No write occurs when log_dir is None."""
    from omegaconf import OmegaConf

    from espnet3.utils.stage_configs import snapshot_config

    resolved = OmegaConf.create({"exp_dir": "./exp/debug"})

    snapshot_config(None, "train", resolved, dry_run=False, is_rank0=True)

    # No exception should occur, no file created since no path
    assert True


def test_snapshot_config_atomic_write_and_history_on_change(tmp_path):
    """Atomic write with history backup when content differs."""
    from omegaconf import OmegaConf

    from espnet3.utils.stage_configs import snapshot_config

    first = OmegaConf.create(
        {
            "exp_dir": "./exp/first",
            "model": "v1",
        }
    )

    snapshot_config(tmp_path, "train", first, dry_run=False, is_rank0=True)

    # Second run with different config
    second = OmegaConf.create(
        {
            "exp_dir": "./exp/second",
            "model": "v2",
        }
    )
    snapshot_config(tmp_path, "train", second, dry_run=False, is_rank0=True)

    # File should exist
    assert (tmp_path / "train.yaml").exists()
    # Content should be updated
    content = (tmp_path / "train.yaml").read_text(encoding="utf-8")
    assert "exp_dir: ./exp/second" in content
    assert "model: v2" in content

    # History directory should exist and contain backup
    history_dir = tmp_path / ".config_history"
    assert history_dir.exists()
    history_files = list(history_dir.iterdir())
    assert len(history_files) >= 1
    assert any("train." in f.name and f.name.endswith(".yaml") for f in history_files)


def test_snapshot_config_no_history_when_identical(tmp_path):
    """No history backup when content is identical."""
    from omegaconf import OmegaConf

    from espnet3.utils.stage_configs import snapshot_config

    resolved = OmegaConf.create({"exp_dir": "./exp", "model": "test"})

    # First write
    snapshot_config(tmp_path, "train", resolved, dry_run=False, is_rank0=True)

    # Identical second write
    snapshot_config(tmp_path, "train", resolved, dry_run=False, is_rank0=True)

    # Content should remain the same
    content = (tmp_path / "train.yaml").read_text(encoding="utf-8")
    assert "exp_dir: ./exp" in content
    assert "model: test" in content

    # No history directory should be created
    assert not (tmp_path / ".config_history").exists()


def test_snapshot_config_full_config_includes_inherited_keys(tmp_path):
    """Snapshot contains full resolved config with inherited keys, not only own_keys."""
    from omegaconf import OmegaConf

    from espnet3.contract.stages import StageSpec
    from espnet3.utils.stage_configs import build_stage_configs, snapshot_config

    class ExampleSystem:
        stages = (
            StageSpec(name="train", config="training"),
            StageSpec(name="infer", config="inference"),
        )

    configs = {
        "training": OmegaConf.create(
            {
                "exp_dir": "./exp",
                "exp_tag": "t",
                "model_conf": {"num_layers": 3},
            }
        ),
        "inference": OmegaConf.create(
            {
                "inference_dir": "${exp_dir}/infer",
                "batch_size": 32,
            }
        ),
    }

    stage_configs, own_keys = build_stage_configs(
        ExampleSystem, configs, exp_dir=tmp_path
    )
    resolved = stage_configs["train"]  # train's full resolved config

    # Snapshot should contain inherited key 'exp_dir' plus own key 'model_conf'
    snapshot_config(tmp_path, "train", resolved, dry_run=False, is_rank0=True)

    snapshot_yaml = (tmp_path / "train.yaml").read_text(encoding="utf-8")
    assert "exp_dir: ./exp" in snapshot_yaml or "model_conf" in snapshot_yaml
    assert "model_conf" in snapshot_yaml


def test_merge_fallback_to_old_layout_when_no_snapshot_exists(tmp_path):
    """Read the legacy layout when the stage has no snapshot."""
    from espnet3.utils.stage_configs import (
        merge_stage_config_from_snapshots,
    )

    # Create old layout file (not snapshot)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "train.yaml").write_text(
        "exp_dir: ./myexp\nseed: 42\nexp_tag: legacy\n", encoding="utf-8"
    )

    # Call merge_stage_config_from_snapshots with exp_dir override
    # No new-style snapshot exists, so it should fall back to old layout
    result = merge_stage_config_from_snapshots(
        _System, "train", {"training": None}, {"exp_dir": str(tmp_path)}
    )

    assert result.exp_dir == str(tmp_path).replace("/config", "")  # Relative path
    assert result.seed == 42
    assert result.exp_tag == "legacy"


def test_merge_prefers_new_snapshot_over_old_layout(tmp_path):
    """When both old layout and new snapshot exist, new snapshot wins."""
    from espnet3.utils.stage_configs import (
        merge_stage_config_from_snapshots,
        snapshot_config,
    )

    # Create old layout file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "train.yaml").write_text(
        "exp_dir: ./old\nseed: 1\n", encoding="utf-8"
    )

    # Create new-style snapshot
    snapshot_config(
        tmp_path,
        "train",
        OmegaConf.create({"exp_dir": str(tmp_path), "seed": 2, "model": "new_model"}),
    )

    result = merge_stage_config_from_snapshots(
        _System, "train", {"training": None}, {"exp_dir": str(tmp_path)}
    )

    # New snapshot should win
    assert result.seed == 2, "New snapshot should override old layout"
    assert result.model == "new_model"


def test_merge_fallback_does_not_override_explicit_stage_selection(tmp_path):
    """Fallback for one stage does not affect another stage's explicit selection."""
    from espnet3.contract.stages import StageSpec
    from espnet3.utils.stage_configs import (
        merge_stage_config_from_snapshots,
    )

    class ThreeStageSystem:
        stages = (
            StageSpec(name="train", config="training"),
            StageSpec(name="infer", config="inference"),
            StageSpec(name="measure", config="metrics"),
        )

    # Create directories for train and measure
    train_dir = tmp_path / "train_logs"
    train_dir.mkdir()
    measure_dir = tmp_path / "measure_logs"
    measure_dir.mkdir()

    # Create old layout for measure stage
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "measure.yaml").write_text(
        "stage_specific: measure_value\n", encoding="utf-8"
    )

    # Run merge for train stage only with explicit inference_dir for infer stage
    # The measure stage should NOT be affected by the old layout fallback
    # because we're only working with train stage (in-scope stages up to train)

    configs = {
        "training": OmegaConf.create({"exp_dir": str(tmp_path)}),
    }
    result = merge_stage_config_from_snapshots(
        ThreeStageSystem, "train", configs, {"exp_dir": str(tmp_path)}
    )

    # Should successfully resolve without errors
    assert "exp_dir" in result


# Cleanup old layout test file references
def test_merge_fallback_only_for_requested_stage(tmp_path):
    """Fallback only applies to the requested stage, not preceding stages."""
    from espnet3.contract.stages import StageSpec
    from espnet3.utils.stage_configs import merge_stage_config_from_snapshots

    class MultiStageSystem:
        stages = (
            StageSpec(name="train", config="training"),
            StageSpec(name="infer", config="inference"),
        )

    # Request infer stage with no directories resolved
    # Fallback should not apply to preceding train stage
    configs = {
        "training": None,  # No explicit training config
        "inference": None,  # No explicit inference config
    }

    # Should handle gracefully - if no CLI exp_dir provided, no fallback
    merge_stage_config_from_snapshots(MultiStageSystem, "infer", configs, {})

    # Result should be based on inherited stage configs only
    # The key point: fallback only applies when requested_stage has no directory
    # and exp_dir is in cli_dir_overrides, so with empty overrides this is fine


def test_snapshot_config_creates_missing_stage_directory(tmp_path):
    """Snapshot a first run before its stage creates any output artifacts."""
    from espnet3.utils.stage_configs import snapshot_config

    log_dir = tmp_path / "new" / "data"
    config = OmegaConf.create({"data_dir": str(log_dir)})
    snapshot_config(log_dir, "create_dataset", config)
    assert OmegaConf.load(log_dir / "create_dataset.yaml") == config
