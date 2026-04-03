import logging

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.system as sysmod
from espnet3.systems.base.system import BaseSystem


def test_base_system_rejects_args():
    system = BaseSystem()
    with pytest.raises(TypeError):
        system.create_dataset(1)


def test_base_system_invokes_helpers(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp"), "model": {}})
    infer_cfg = OmegaConf.create({"inference_dir": str(tmp_path / "infer")})
    measure_cfg = OmegaConf.create({"inference_dir": str(tmp_path / "infer")})

    calls = {}

    def fake_collect(cfg):
        calls["collect"] = cfg
        return "collect"

    def fake_train(cfg):
        calls["train"] = cfg
        return "train"

    def fake_infer(cfg):
        calls["infer"] = cfg
        return "infer"

    def fake_metric(cfg):
        calls["measure"] = cfg
        return {"metric": 1.0}

    monkeypatch.setattr(sysmod, "collect_stats", fake_collect)
    monkeypatch.setattr(sysmod, "train", fake_train)
    monkeypatch.setattr(sysmod, "infer", fake_infer)
    monkeypatch.setattr(sysmod, "measure", fake_metric)

    system = BaseSystem(
        training_config=train_cfg,
        inference_config=infer_cfg,
        metrics_config=measure_cfg,
    )

    assert system.exp_dir.is_dir()
    assert system.collect_stats() == "collect"
    assert system.train() == "train"
    assert system.infer() == "infer"
    assert system.measure() == {"metric": 1.0}
    assert calls["collect"] is train_cfg
    assert calls["train"] is train_cfg
    assert calls["infer"] is infer_cfg
    assert calls["measure"] is measure_cfg


def test_base_system_create_dataset_requires_dataset_config(tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
        }
    )
    system = BaseSystem(training_config=train_cfg)
    with pytest.raises(RuntimeError, match="training_config.dataset must be set"):
        system.create_dataset()


def test_base_system_create_dataset_stage_logs_use_data_dir(tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "data_dir": str(tmp_path / "data"),
            "recipe_dir": str(tmp_path / "recipe"),
        }
    )

    system = BaseSystem(training_config=train_cfg)

    assert system.stage_log_dirs["create_dataset"] == tmp_path / "data"


def test_base_system_create_dataset_prepares_dataset_references(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "create_dataset": {
                "recipe_dir": str(tmp_path / "recipe"),
                "archive_path": "a.tar.gz",
            },
            "dataset": {
                "train": [{"ref": "mini_an4/asr"}],
                # Same ref in valid — dedup means only one prepare run
                "valid": [{"ref": "mini_an4/asr"}],
                "test": None,
            },
        }
    )
    system = BaseSystem(training_config=train_cfg)
    calls = []

    class DummyBuilder:
        def is_source_prepared(self, **kwargs):
            calls.append(("is_source_prepared", kwargs))
            return True

        def prepare_source(self, **kwargs):
            calls.append(("prepare_source", kwargs))

        def is_built(self, **kwargs):
            calls.append(("is_built", kwargs))
            return True

        def build(self, **kwargs):
            calls.append(("build", kwargs))

    class DummyModule:
        DatasetBuilder = DummyBuilder

    monkeypatch.setattr(
        sysmod,
        "load_dataset_module",
        lambda ref=None, recipe_dir=None: DummyModule(),
    )

    assert system.create_dataset() is None
    expected_kwargs = {
        "archive_path": "a.tar.gz",
        "recipe_dir": str(tmp_path / "recipe"),
    }
    assert calls == [
        ("is_source_prepared", expected_kwargs),
        ("is_built", expected_kwargs),
    ]


def test_base_system_create_dataset_logs_progress(tmp_path, monkeypatch, caplog):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "create_dataset": {"recipe_dir": str(tmp_path / "recipe")},
            "dataset": {
                "train": [{"ref": "mini_an4/asr"}],
                "valid": None,
                "test": None,
            },
        }
    )
    system = BaseSystem(training_config=train_cfg)

    class DummyBuilder:
        def is_source_prepared(self, **kwargs):
            return True

        def prepare_source(self, **kwargs):
            return None

        def is_built(self, **kwargs):
            return True

        def build(self, **kwargs):
            return None

    class DummyModule:
        DatasetBuilder = DummyBuilder

    monkeypatch.setattr(
        sysmod,
        "load_dataset_module",
        lambda ref=None, recipe_dir=None: DummyModule(),
    )

    with caplog.at_level(logging.INFO):
        system.create_dataset()

    assert "starting dataset creation process" in caplog.text
    assert "Ensuring dataset is prepared: mini_an4/asr" in caplog.text
    assert "Dataset creation completed" in caplog.text


def test_base_system_create_dataset_runs_prepare_and_build_when_needed(
    tmp_path, monkeypatch
):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "create_dataset": {"recipe_dir": str(tmp_path / "recipe")},
            "dataset": {
                "train": [{"ref": "mini_an4/asr"}],
                "valid": None,
                "test": None,
            },
        }
    )
    system = BaseSystem(training_config=train_cfg)
    calls = []

    class DummyBuilder:
        def is_source_prepared(self, **kwargs):
            calls.append(("is_source_prepared", kwargs))
            return False

        def prepare_source(self, **kwargs):
            calls.append(("prepare_source", kwargs))

        def is_built(self, **kwargs):
            calls.append(("is_built", kwargs))
            return False

        def build(self, **kwargs):
            calls.append(("build", kwargs))

    class DummyModule:
        DatasetBuilder = DummyBuilder

    monkeypatch.setattr(
        sysmod,
        "load_dataset_module",
        lambda ref=None, recipe_dir=None: DummyModule(),
    )

    assert system.create_dataset() is None
    expected_kwargs = {"recipe_dir": str(tmp_path / "recipe")}
    assert calls == [
        ("is_source_prepared", expected_kwargs),
        ("prepare_source", expected_kwargs),
        ("is_built", expected_kwargs),
        ("build", expected_kwargs),
    ]


def test_base_system_create_dataset_raises_when_no_dataset_entries(tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": {"train": None, "valid": None, "test": None},
        }
    )
    system = BaseSystem(training_config=train_cfg)
    with pytest.raises(RuntimeError, match="must include at least one entry"):
        system.create_dataset()


def test_base_system_create_dataset_local_ref_dedup(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "create_dataset": {"recipe_dir": str(tmp_path / "recipe")},
            "dataset": {
                "train": [{"kwargs": {"split": "train"}}],
                "valid": [{"kwargs": {"split": "valid"}}],
                "test": None,
            },
        }
    )
    system = BaseSystem(training_config=train_cfg)
    calls = []

    class DummyBuilder:
        def is_source_prepared(self, **kwargs):
            calls.append(("is_source_prepared", kwargs))
            return True

        def prepare_source(self, **kwargs):
            calls.append(("prepare_source", kwargs))

        def is_built(self, **kwargs):
            calls.append(("is_built", kwargs))
            return True

        def build(self, **kwargs):
            calls.append(("build", kwargs))

    class DummyModule:
        DatasetBuilder = DummyBuilder

    monkeypatch.setattr(
        sysmod,
        "load_dataset_module",
        lambda ref=None, recipe_dir=None: DummyModule(),
    )

    assert system.create_dataset() is None
    expected_kwargs = {"recipe_dir": str(tmp_path / "recipe")}
    # ref=None entries should be deduplicated and prepared only once.
    assert calls == [
        ("is_source_prepared", expected_kwargs),
        ("is_built", expected_kwargs),
    ]


def test_base_system_rejects_subclass_args():
    class CustomSystem(BaseSystem):
        def train(self, *, extra=None):
            return super().train(extra=extra)

    system = CustomSystem()
    with pytest.raises(TypeError):
        system.train(extra="oops")
