from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from espnet3.systems.asr.publication import (
    _resolve_stats_pack_paths,
    get_pack_model_artifacts,
)


def _make_system(training_config):
    return SimpleNamespace(training_config=training_config)


# ---------------------------------------------------------------------------
# get_pack_model_artifacts – exp_dir files
# ---------------------------------------------------------------------------


def test_returns_empty_when_exp_dir_has_no_known_files(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    artifacts = get_pack_model_artifacts(system)

    assert artifacts["files"] == {}
    assert artifacts["yaml_files"] == {}
    assert artifacts["copy_paths"] == []


def test_includes_model_checkpoint(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "last.ckpt").write_text("weights", encoding="utf-8")
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    artifacts = get_pack_model_artifacts(system)

    assert artifacts["files"]["asr_model_file"] == str(exp_dir / "last.ckpt")


def test_includes_train_config(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "config.yaml").write_text("dummy: true", encoding="utf-8")
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    artifacts = get_pack_model_artifacts(system)

    assert artifacts["yaml_files"]["asr_train_config"] == str(exp_dir / "config.yaml")


def test_includes_both_checkpoint_and_config(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "last.ckpt").write_text("weights", encoding="utf-8")
    (exp_dir / "config.yaml").write_text("dummy: true", encoding="utf-8")
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    artifacts = get_pack_model_artifacts(system)

    assert "asr_model_file" in artifacts["files"]
    assert "asr_train_config" in artifacts["yaml_files"]


def test_skips_checkpoint_when_missing(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "config.yaml").write_text("dummy: true", encoding="utf-8")
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    artifacts = get_pack_model_artifacts(system)

    assert "asr_model_file" not in artifacts["files"]


def test_skips_train_config_when_missing(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "last.ckpt").write_text("weights", encoding="utf-8")
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    artifacts = get_pack_model_artifacts(system)

    assert "asr_train_config" not in artifacts["yaml_files"]


# ---------------------------------------------------------------------------
# get_pack_model_artifacts – tokenizer
# ---------------------------------------------------------------------------


def test_includes_tokenizer_save_path(tmp_path):
    exp_dir = tmp_path / "exp"
    tokenizer_dir = tmp_path / "tok"
    exp_dir.mkdir()
    tokenizer_dir.mkdir()
    training_config = OmegaConf.create(
        {"exp_dir": str(exp_dir), "tokenizer": {"save_path": str(tokenizer_dir)}}
    )
    system = _make_system(training_config)

    artifacts = get_pack_model_artifacts(system)

    assert tokenizer_dir in artifacts["copy_paths"]


def test_skips_tokenizer_when_not_configured(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    assert get_pack_model_artifacts(system)["copy_paths"] == []


def test_skips_tokenizer_when_save_path_is_none(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    training_config = OmegaConf.create(
        {"exp_dir": str(exp_dir), "tokenizer": {"save_path": None}}
    )
    system = _make_system(training_config)

    assert get_pack_model_artifacts(system)["copy_paths"] == []


# ---------------------------------------------------------------------------
# get_pack_model_artifacts – data_dir/tokenizer
# ---------------------------------------------------------------------------


def test_includes_data_tokenizer_when_it_exists(tmp_path):
    exp_dir = tmp_path / "exp"
    data_tokenizer = tmp_path / "data" / "tokenizer"
    exp_dir.mkdir()
    data_tokenizer.mkdir(parents=True)
    training_config = OmegaConf.create(
        {"exp_dir": str(exp_dir), "data_dir": str(tmp_path / "data")}
    )
    system = _make_system(training_config)

    artifacts = get_pack_model_artifacts(system)

    assert data_tokenizer in artifacts["copy_paths"]


def test_skips_data_tokenizer_when_missing(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    training_config = OmegaConf.create(
        {"exp_dir": str(exp_dir), "data_dir": str(tmp_path / "data")}
    )
    system = _make_system(training_config)

    assert get_pack_model_artifacts(system)["copy_paths"] == []


def test_skips_data_dir_when_not_configured(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    system = _make_system(OmegaConf.create({"exp_dir": str(exp_dir)}))

    assert get_pack_model_artifacts(system)["copy_paths"] == []


def test_includes_both_tokenizer_save_path_and_data_tokenizer(tmp_path):
    exp_dir = tmp_path / "exp"
    tokenizer_dir = tmp_path / "tok"
    data_tokenizer = tmp_path / "data" / "tokenizer"
    exp_dir.mkdir()
    tokenizer_dir.mkdir()
    data_tokenizer.mkdir(parents=True)
    training_config = OmegaConf.create(
        {
            "exp_dir": str(exp_dir),
            "tokenizer": {"save_path": str(tokenizer_dir)},
            "data_dir": str(tmp_path / "data"),
        }
    )
    system = _make_system(training_config)

    copy_paths = get_pack_model_artifacts(system)["copy_paths"]

    assert tokenizer_dir in copy_paths
    assert data_tokenizer in copy_paths


# ---------------------------------------------------------------------------
# get_pack_model_artifacts – stats
# ---------------------------------------------------------------------------


def test_includes_stats_npz_files(tmp_path):
    exp_dir = tmp_path / "exp"
    stats_dir = tmp_path / "stats"
    train_dir = stats_dir / "train"
    exp_dir.mkdir()
    train_dir.mkdir(parents=True)
    npz = train_dir / "feats_stats.npz"
    npz.write_text("", encoding="utf-8")
    training_config = OmegaConf.create(
        {"exp_dir": str(exp_dir), "stats_dir": str(stats_dir)}
    )
    system = _make_system(training_config)

    assert npz in get_pack_model_artifacts(system)["copy_paths"]


def test_includes_all_npz_files_from_stats(tmp_path):
    exp_dir = tmp_path / "exp"
    stats_dir = tmp_path / "stats"
    train_dir = stats_dir / "train"
    exp_dir.mkdir()
    train_dir.mkdir(parents=True)
    (train_dir / "a.npz").write_text("", encoding="utf-8")
    (train_dir / "b.npz").write_text("", encoding="utf-8")
    training_config = OmegaConf.create(
        {"exp_dir": str(exp_dir), "stats_dir": str(stats_dir)}
    )
    system = _make_system(training_config)

    copy_paths = get_pack_model_artifacts(system)["copy_paths"]

    assert train_dir / "a.npz" in copy_paths
    assert train_dir / "b.npz" in copy_paths


# ---------------------------------------------------------------------------
# _resolve_stats_pack_paths
# ---------------------------------------------------------------------------


def test_resolve_returns_sorted_npz_files(tmp_path):
    stats_dir = tmp_path / "stats"
    train_dir = stats_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "b_stats.npz").write_text("", encoding="utf-8")
    (train_dir / "a_stats.npz").write_text("", encoding="utf-8")
    (train_dir / "not_npz.txt").write_text("", encoding="utf-8")
    system = _make_system(OmegaConf.create({"stats_dir": str(stats_dir)}))

    paths = _resolve_stats_pack_paths(system)

    assert [p.name for p in paths] == ["a_stats.npz", "b_stats.npz"]


def test_resolve_excludes_non_npz_files(tmp_path):
    stats_dir = tmp_path / "stats"
    train_dir = stats_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "stats.npz").write_text("", encoding="utf-8")
    (train_dir / "stats.txt").write_text("", encoding="utf-8")
    system = _make_system(OmegaConf.create({"stats_dir": str(stats_dir)}))

    paths = _resolve_stats_pack_paths(system)

    assert all(p.suffix == ".npz" for p in paths)
    assert len(paths) == 1


def test_resolve_returns_empty_when_no_stats_dir(tmp_path):
    system = _make_system(OmegaConf.create({"exp_dir": str(tmp_path)}))

    assert _resolve_stats_pack_paths(system) == []


def test_resolve_returns_empty_when_train_dir_missing(tmp_path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    system = _make_system(OmegaConf.create({"stats_dir": str(stats_dir)}))

    assert _resolve_stats_pack_paths(system) == []


def test_resolve_returns_empty_when_train_dir_has_no_npz(tmp_path):
    stats_dir = tmp_path / "stats"
    train_dir = stats_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "log.txt").write_text("", encoding="utf-8")
    system = _make_system(OmegaConf.create({"stats_dir": str(stats_dir)}))

    assert _resolve_stats_pack_paths(system) == []


def test_resolve_returns_empty_when_train_dir_is_empty(tmp_path):
    stats_dir = tmp_path / "stats"
    (stats_dir / "train").mkdir(parents=True)
    system = _make_system(OmegaConf.create({"stats_dir": str(stats_dir)}))

    assert _resolve_stats_pack_paths(system) == []
