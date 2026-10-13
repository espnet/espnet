import logging

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.utils.run_utils import ConfigError
from espnet3.utils.stage_configs import bake, build_stage_configs, inherit, load_baked


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
