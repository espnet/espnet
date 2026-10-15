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
