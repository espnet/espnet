"""Tests for saved-config search/priority merging.

Covers resolve_search_directories (design/config_tree_fix.md, "保存済み
config の探索" steps 1-4) and merge_stage_config_from_snapshots (steps 5-8
and "設定の優先順位と独立性").
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.utils.run_utils import ConfigError
from espnet3.utils.stage_configs import (
    merge_stage_config_from_snapshots,
    resolve_search_directories,
    snapshot_config,
)


class _System:
    stages = (
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
    )


class _SystemWithStatsRole:
    stages = (
        StageSpec(name="collect_stats", config="stats", log_dir="stats_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
    )


class _TrainOnlySystem:
    stages = (
        StageSpec(name="train", config="training", log_dir="exp_dir"),
    )


# ---------------------------------------------------------------------------
# resolve_search_directories
# ---------------------------------------------------------------------------


def test_resolve_search_directories_cli_override_wins(tmp_path):
    configs = {"training": OmegaConf.create({"exp_dir": str(tmp_path / "own")})}
    dirs = resolve_search_directories(
        _System, "train", configs, {"exp_dir": str(tmp_path / "cli")}
    )
    assert dirs["train"] == Path(tmp_path / "cli")


def test_resolve_search_directories_falls_back_to_own_config(tmp_path):
    configs = {"training": OmegaConf.create({"exp_dir": str(tmp_path / "own")})}
    dirs = resolve_search_directories(_System, "train", configs, {})
    assert dirs["train"] == Path(tmp_path / "own")


def test_resolve_search_directories_omits_unresolvable_stage():
    dirs = resolve_search_directories(_System, "train", {"training": None}, {})
    assert "train" not in dirs


def test_resolve_search_directories_shared_canonical_key_same_directory(tmp_path):
    configs = {
        "inference": OmegaConf.create({"inference_dir": str(tmp_path / "beam10")}),
        "metrics": OmegaConf.create({"inference_dir": str(tmp_path / "beam10")}),
    }
    dirs = resolve_search_directories(_System, "measure", configs, {})
    assert dirs["infer"] == dirs["measure"] == Path(tmp_path / "beam10")


def test_resolve_search_directories_never_requires_later_stage():
    # "measure" is declared after "train"; requesting "train" must not
    # require anything resolvable for "measure".
    dirs = resolve_search_directories(_System, "train", {"training": None}, {})
    assert "measure" not in dirs
    assert "infer" not in dirs


# ---------------------------------------------------------------------------
# merge_stage_config_from_snapshots
# ---------------------------------------------------------------------------


def test_merge_prefers_cli_over_explicit_over_own_snapshot(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    snapshot_config(
        exp_dir, "train", OmegaConf.create({"exp_dir": str(exp_dir), "seed": 1}),
    )

    # This run gives its own explicit training config (seed=2): must win
    # over the old snapshot's seed=1, and must not resurrect any field the
    # new config no longer has.
    configs = {"training": OmegaConf.create({"exp_dir": str(exp_dir), "seed": 2})}
    result = merge_stage_config_from_snapshots(
        _System, "train", configs, {"exp_dir": str(exp_dir)}
    )
    assert result.seed == 2


def test_merge_explicit_config_does_not_resurrect_deleted_snapshot_fields(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    snapshot_config(
        exp_dir, "train",
        OmegaConf.create({"exp_dir": str(exp_dir), "normalize": True}),
    )

    # New explicit config no longer has 'normalize' at all.
    configs = {"training": OmegaConf.create({"exp_dir": str(exp_dir)})}
    result = merge_stage_config_from_snapshots(_System, "train", configs, {})
    assert "normalize" not in result


def test_merge_falls_back_to_own_snapshot_when_no_explicit_config(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    snapshot_config(
        exp_dir, "train", OmegaConf.create({"exp_dir": str(exp_dir), "seed": 7}),
    )

    result = merge_stage_config_from_snapshots(
        _System, "train", {"training": None}, {"exp_dir": str(exp_dir)}
    )
    assert result.seed == 7


def test_merge_inherits_preceding_stage_snapshot(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    snapshot_config(
        exp_dir, "train",
        OmegaConf.create({"exp_dir": str(exp_dir), "model_conf": {"layers": 3}}),
    )
    infer_dir = tmp_path / "beam10"
    infer_dir.mkdir()

    configs = {"inference": OmegaConf.create({"inference_dir": str(infer_dir)})}
    result = merge_stage_config_from_snapshots(
        _System, "infer", configs,
        {"exp_dir": str(exp_dir), "inference_dir": str(infer_dir)},
    )
    assert result.model_conf.layers == 3
    assert result.inference_dir == str(infer_dir)


def test_merge_beam10_and_beam20_are_independent_copies(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    snapshot_config(
        exp_dir, "train",
        OmegaConf.create({"exp_dir": str(exp_dir), "model_conf": {"layers": 3}}),
    )

    beam10 = tmp_path / "beam10"
    beam10.mkdir()
    beam20 = tmp_path / "beam20"
    beam20.mkdir()

    cfg10 = merge_stage_config_from_snapshots(
        _System, "infer",
        {"inference": OmegaConf.create({"inference_dir": str(beam10), "beam": 10})},
        {"exp_dir": str(exp_dir), "inference_dir": str(beam10)},
    )
    cfg10.model_conf.layers = 999  # mutate - must not leak into beam20's result

    cfg20 = merge_stage_config_from_snapshots(
        _System, "infer",
        {"inference": OmegaConf.create({"inference_dir": str(beam20), "beam": 20})},
        {"exp_dir": str(exp_dir), "inference_dir": str(beam20)},
    )
    assert cfg20.model_conf.layers == 3
    assert cfg20.beam == 20


def test_merge_measure_reuses_saved_measure_config(tmp_path):
    infer_dir = tmp_path / "beam10"
    infer_dir.mkdir()
    snapshot_config(
        infer_dir, "infer",
        OmegaConf.create({"inference_dir": str(infer_dir), "beam": 10}),
    )
    snapshot_config(
        infer_dir, "measure",
        OmegaConf.create({"inference_dir": str(infer_dir), "metric": "wer"}),
    )

    # Standalone re-evaluation: no explicit inference/metrics config this
    # run, only --inference_dir.
    result = merge_stage_config_from_snapshots(
        _System, "measure", {"inference": None, "metrics": None},
        {"inference_dir": str(infer_dir)},
    )
    assert result.metric == "wer"
    assert result.beam == 10  # inherited from infer's snapshot


def test_merge_conflicting_parent_experiments_raises(tmp_path):
    exp_a = tmp_path / "exp_a"
    exp_a.mkdir()
    exp_b = tmp_path / "exp_b"
    exp_b.mkdir()
    snapshot_config(exp_a, "train", OmegaConf.create({"exp_dir": str(exp_a)}))

    infer_dir = tmp_path / "beam10"
    infer_dir.mkdir()
    # infer's own snapshot claims a DIFFERENT parent exp_dir than train's.
    snapshot_config(
        infer_dir, "infer",
        OmegaConf.create({"exp_dir": str(exp_b), "inference_dir": str(infer_dir)}),
    )

    with pytest.raises(ConfigError, match="conflicting saved configs"):
        merge_stage_config_from_snapshots(
            _System, "infer", {"inference": None},
            {"exp_dir": str(exp_a), "inference_dir": str(infer_dir)},
        )


def test_collect_stats_own_role_separate_from_train(tmp_path):
    # System where collect_stats has its own "stats" config role, separate from train's "training"
    stats_dir = tmp_path / "stats_an4"
    stats_dir.mkdir()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    # Simulate collect_stats run: saves its config to stats_dir/collect_stats.yaml
    collect_stats_config = OmegaConf.create({
        "stats_dir": str(stats_dir),
        "model_conf": {"layers": 5},
    })
    snapshot_config(stats_dir, "collect_stats", collect_stats_config)

    # Simulate train run in a separate process: no configs_by_role entries in memory,
    # only --stats_dir CLI override pointing to where collect_stats saved its config
    train_config = OmegaConf.create({
        "exp_dir": str(exp_dir),
    })

    result = merge_stage_config_from_snapshots(
        _SystemWithStatsRole,
        "train",
        {"training": train_config, "stats": None},  # stats role not provided (separate process)
        {"stats_dir": str(stats_dir)},  # CLI override pointing to collect_stats output
    )

    # train should inherit model_conf from collect_stats via stats_dir snapshot
    assert result.model_conf.layers == 5
    assert result.exp_dir == str(exp_dir)


def test_merge_missing_stage_raises_clear_error():
    with pytest.raises(ConfigError, match="not declared"):
        merge_stage_config_from_snapshots(_System, "nonexistent", {}, {})


def test_merge_corrupt_snapshot_raises_naming_stage_and_path(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "train.yaml").write_text("foo: [1, 2\n bar", encoding="utf-8")

    with pytest.raises(ConfigError, match="train"):
        merge_stage_config_from_snapshots(
            _System, "train", {"training": None}, {"exp_dir": str(exp_dir)}
        )


def test_train_only_system_no_infer_or_measure(tmp_path):
    # System with ONLY a train stage; no infer or measure at all.
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    # Provide an explicit training config for this run
    configs = {"training": OmegaConf.create({"exp_dir": str(exp_dir), "seed": 42})}

    # Resolve search directories for train
    dirs = resolve_search_directories(_TrainOnlySystem, "train", configs, {})
    assert dirs["train"] == Path(exp_dir)

    # Merge stage config from snapshots for train
    result = merge_stage_config_from_snapshots(
        _TrainOnlySystem, "train", configs, {"exp_dir": str(exp_dir)}
    )
    assert result.seed == 42


# ---------------------------------------------------------------------------
# Items 5, 6, 7: beam10/beam20 infer+measure flow
# ---------------------------------------------------------------------------


def test_beam10_beam20_infer_measure_separate_configs_and_outputs(tmp_path):
    """
    Item 5: beam10's infer+measure and beam20's infer+measure each use their
    own explicit infer.yaml-equivalent config and produce separate, correctly-
    scoped results.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    
    # Train produces a model
    train_config = OmegaConf.create({
        "exp_dir": str(exp_dir),
        "model_conf": {"layers": 3, "unit": "transformer"},
    })
    snapshot_config(exp_dir, "train", train_config)

    # beam10 inference
    beam10_dir = tmp_path / "beam10"
    beam10_dir.mkdir()
    beam10_infer_config = OmegaConf.create({
        "inference_dir": str(beam10_dir),
        "beam": 10,
        "batch_size": 16,
    })
    snapshot_config(beam10_dir, "infer", beam10_infer_config)
    
    # beam20 inference
    beam20_dir = tmp_path / "beam20"
    beam20_dir.mkdir()
    beam20_infer_config = OmegaConf.create({
        "inference_dir": str(beam20_dir),
        "beam": 20,
        "batch_size": 32,
    })
    snapshot_config(beam20_dir, "infer", beam20_infer_config)

    # beam10 measure
    beam10_measure_config = OmegaConf.create({
        "inference_dir": str(beam10_dir),
        "metric": "wer",
        "dataset": "test-clean",
    })
    snapshot_config(beam10_dir, "measure", beam10_measure_config)
    
    # beam20 measure
    beam20_measure_config = OmegaConf.create({
        "inference_dir": str(beam20_dir),
        "metric": "cer",
        "dataset": "test-other",
    })
    snapshot_config(beam20_dir, "measure", beam20_measure_config)

    # Verify beam10 measure uses beam10's snapshot and configs
    result10 = merge_stage_config_from_snapshots(
        _System, "measure",
        {"inference": None, "metrics": None},
        {"inference_dir": str(beam10_dir)},
    )
    assert result10.beam == 10
    assert result10.batch_size == 16
    assert result10.metric == "wer"
    assert result10.dataset == "test-clean"

    # Verify beam20 measure uses beam20's snapshot and configs
    result20 = merge_stage_config_from_snapshots(
        _System, "measure",
        {"inference": None, "metrics": None},
        {"inference_dir": str(beam20_dir)},
    )
    assert result20.beam == 20
    assert result20.batch_size == 32
    assert result20.metric == "cer"
    assert result20.dataset == "test-other"



def test_beam20_does_not_modify_beam10_snapshot_or_metrics(tmp_path):
    """
    Item 6: After beam10's full flow is written, beam20 execution must NOT
    change beam10's already-written snapshot file content or its metrics.json
    stand-in.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    
    # Train
    train_config = OmegaConf.create({
        "exp_dir": str(exp_dir),
        "model_conf": {"layers": 3},
    })
    snapshot_config(exp_dir, "train", train_config)

    # beam10
    beam10_dir = tmp_path / "beam10"
    beam10_dir.mkdir()
    beam10_infer_config = OmegaConf.create({
        "inference_dir": str(beam10_dir),
        "beam": 10,
    })
    snapshot_config(beam10_dir, "infer", beam10_infer_config)
    
    beam10_measure_config = OmegaConf.create({
        "inference_dir": str(beam10_dir),
        "metric": "wer",
    })
    snapshot_config(beam10_dir, "measure", beam10_measure_config)
    
    # Create a metrics.json stand-in file
    (beam10_dir / "metrics.json").write_text('{"wer": 0.05}', encoding="utf-8")
    
    # Save original content and mtime
    original_metrics_content = (beam10_dir / "metrics.json").read_text(encoding="utf-8")
    original_mtime = (beam10_dir / "metrics.json").stat().st_mtime
    original_infer_content = (beam10_dir / "infer.yaml").read_text(encoding="utf-8")
    original_measure_content = (beam10_dir / "measure.yaml").read_text(encoding="utf-8")

    # Run beam20
    beam20_dir = tmp_path / "beam20"
    beam20_dir.mkdir()
    beam20_infer_config = OmegaConf.create({
        "inference_dir": str(beam20_dir),
        "beam": 20,
    })
    snapshot_config(beam20_dir, "infer", beam20_infer_config)
    
    # Verify beam10 files were NOT modified by beam20 execution
    assert (beam10_dir / "metrics.json").read_text(encoding="utf-8") == original_metrics_content
    assert (beam10_dir / "infer.yaml").read_text(encoding="utf-8") == original_infer_content
    assert (beam10_dir / "measure.yaml").read_text(encoding="utf-8") == original_measure_content
    assert (beam10_dir / "metrics.json").stat().st_mtime == original_mtime



def test_measure_reuses_saved_snapshot_on_subsequent_run(tmp_path):
    """
    Item 7: beam10's first measure run is given a metrics config; re-running
    measure for beam10 later with no explicit metrics config (only --inference_dir)
    must reuse the saved measure.yaml snapshot rather than requiring the config.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    
    # Train
    train_config = OmegaConf.create({
        "exp_dir": str(exp_dir),
        "model_conf": {"layers": 3},
    })
    snapshot_config(exp_dir, "train", train_config)

    # beam10
    beam10_dir = tmp_path / "beam10"
    beam10_dir.mkdir()
    beam10_infer_config = OmegaConf.create({
        "inference_dir": str(beam10_dir),
        "beam": 10,
    })
    snapshot_config(beam10_dir, "infer", beam10_infer_config)

    # First measure run with explicit metrics config
    beam10_measure_config = OmegaConf.create({
        "inference_dir": str(beam10_dir),
        "metric": "wer",
        "dataset": "test-clean",
    })
    snapshot_config(beam10_dir, "measure", beam10_measure_config)

    # Verify first run works
    result1 = merge_stage_config_from_snapshots(
        _System, "measure",
        {"inference": None, "metrics": beam10_measure_config},
        {"inference_dir": str(beam10_dir)},
    )
    assert result1.metric == "wer"
    assert result1.dataset == "test-clean"

    # Second measure run WITHOUT explicit metrics config (only inference_dir)
    result2 = merge_stage_config_from_snapshots(
        _System, "measure",
        {"inference": None, "metrics": None},
        {"inference_dir": str(beam10_dir)},
    )
    # Should still have the metric from the saved snapshot
    assert result2.metric == "wer"
    assert result2.dataset == "test-clean"
    # Should also inherit beam from infer snapshot
    assert result2.beam == 10


# ---------------------------------------------------------------------------
# Item 1: Stage config independence
# ---------------------------------------------------------------------------


def test_stage_config_independence_second_stage_omits_setting(tmp_path):
    """
    Item 1: Build two stages' configs where the SECOND stage's own explicit
    config omits a setting that the FIRST stage's own snapshot still has.
    Confirm the first stage's own resolved config AND its already-written
    snapshot file are unaffected by the second stage's config.
    
    The key point: when the second stage runs with a config that omits a
    setting, it should not cause the first stage's snapshot to be modified
    or resurrect deleted settings in the first stage.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    
    # First stage (train) with normalize setting
    train_config = OmegaConf.create({
        "exp_dir": str(exp_dir),
        "normalize": True,
        "model_conf": {"layers": 3},
    })
    snapshot_config(exp_dir, "train", train_config)
    
    # Save original snapshot content and mtime
    original_snapshot_path = exp_dir / "train.yaml"
    original_content = original_snapshot_path.read_text(encoding="utf-8")
    original_mtime = original_snapshot_path.stat().st_mtime

    # Second stage (infer) with its own config that omits 'normalize'
    # This simulates a scenario where collect_stats removes normalize,
    # but train should still keep it
    infer_dir = tmp_path / "beam10"
    infer_dir.mkdir()
    infer_config = OmegaConf.create({
        "inference_dir": str(infer_dir),
        "beam": 10,
    })
    
    # Build configs for both stages
    configs = {"training": None, "inference": infer_config}
    result_train = merge_stage_config_from_snapshots(
        _System, "train", configs, {"exp_dir": str(exp_dir)}
    )
    result_infer = merge_stage_config_from_snapshots(
        _System, "infer", configs,
        {"exp_dir": str(exp_dir), "inference_dir": str(infer_dir)}
    )
    
    # First stage's resolved config should still have normalize
    assert result_train.normalize is True
    
    # Verify the snapshot file still contains normalize and hasn't changed
    assert original_snapshot_path.exists()
    current_content = original_snapshot_path.read_text(encoding="utf-8")
    assert current_content == original_content
    assert original_snapshot_path.stat().st_mtime == original_mtime
    snapshot = OmegaConf.load(original_snapshot_path)
    assert snapshot.normalize is True
    
    # Most importantly: verify that running the second stage did NOT modify
    # the first stage's snapshot file
    assert original_snapshot_path.stat().st_mtime == original_mtime


# ---------------------------------------------------------------------------
# Item 2: CLI override with OmegaConf interpolation
# ---------------------------------------------------------------------------


def test_cli_override_resolves_interpolation_before_final_resolution(tmp_path):
    """
    Item 2: Build a config with an OmegaConf interpolation that references
    a key which the CLI override of that same key would change (e.g., a key
    like "out": "${exp_dir}/out" where exp_dir is what the CLI overrides).
    Confirm the CLI-overridden value interpolation resolves using the
    CLI-overridden value, not the original config's value.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    
    # Config with interpolation using exp_dir
    training_config = OmegaConf.create({
        "exp_dir": "./original_exp",  # Original value
        "out": "${exp_dir}/output",
        "model_conf": {"layers": 3},
    })
    
    # CLI overrides exp_dir
    configs = {"training": training_config}
    result = merge_stage_config_from_snapshots(
        _System, "train", configs, {"exp_dir": str(exp_dir)}
    )
    
    # The interpolation should use the CLI-overridden exp_dir
    assert result.exp_dir == str(exp_dir)
    assert result.out == f"{exp_dir}/output"


# ---------------------------------------------------------------------------
# Item 11: Clear error for unresolvable required values
# ---------------------------------------------------------------------------


def test_unresolvable_interpolation_raises_error_with_stage_and_key(tmp_path):
    """
    Item 11: Confirm that when a required value cannot be resolved at all
    (e.g., an unresolvable interpolation like "${missing_key}"), the raised
    error includes the specific stage and key involved, not a generic message.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    
    # Config with unresolvable interpolation
    training_config = OmegaConf.create({
        "exp_dir": str(exp_dir),
        "out": "${missing_key}/output",  # This will fail to resolve
        "model_conf": {"layers": 3},
    })
    
    configs = {"training": training_config}
    
    # Should raise ConfigError with stage and key information
    with pytest.raises(ConfigError, match="train config cannot resolve"):
        merge_stage_config_from_snapshots(
            _System, "train", configs, {"exp_dir": str(exp_dir)}
        )


def test_unresolvable_interpolation_in_nested_key_raises_error_with_key_path(tmp_path):
    """
    Item 11 (continued): Test that nested key paths are also reported in errors.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    
    # Config with unresolvable interpolation in a nested key
    training_config = OmegaConf.create({
        "exp_dir": str(exp_dir),
        "model_conf": {
            "layers": 3,
            "output_path": "${nonexistent_key}/model.pth",
        },
    })
    
    configs = {"training": training_config}
    
    # Should raise ConfigError mentioning the nested key
    with pytest.raises(ConfigError, match="train config cannot resolve"):
        merge_stage_config_from_snapshots(
            _System, "train", configs, {"exp_dir": str(exp_dir)}
        )
