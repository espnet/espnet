import logging

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.system as sysmod
from espnet3.systems.base.system import BaseSystem
from espnet3.utils.run_utils import ConfigError


def test_base_system_requires_configs_kwarg():
    with pytest.raises(TypeError):
        BaseSystem()


def test_base_system_rejects_args(tmp_path):
    system = BaseSystem(configs={}, exp_dir=tmp_path / "exp")
    with pytest.raises(TypeError):
        system.create_dataset(1)


def test_base_system_require_returns_value():
    config = OmegaConf.create({"save_path": "data/out"})
    assert BaseSystem._require(config, "save_path", "msg") == "data/out"


def test_base_system_require_raises_on_missing_key():
    config = OmegaConf.create({"other": 1})
    with pytest.raises(RuntimeError, match="save_path must be set"):
        BaseSystem._require(config, "save_path", "save_path must be set")


def test_base_system_require_raises_on_none_config():
    with pytest.raises(RuntimeError, match="section must be set"):
        BaseSystem._require(None, "section", "section must be set")


def test_base_system_builds_stage_configs_from_configs_kwarg(tmp_path):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    infer_cfg = OmegaConf.create({"inference_dir": "${exp_dir}/infer"})

    system = BaseSystem(configs={"training": train_cfg, "inference": infer_cfg})

    assert system.stage_configs["train"].exp_dir == str(tmp_path / "exp")
    assert (
        system.stage_configs["infer"].inference_dir == str(tmp_path / "exp") + "/infer"
    )


def test_base_system_explicit_exp_dir_wins_over_a_configs_value(tmp_path):
    train_cfg = OmegaConf.create({"exp_dir": "./ignored"})

    system = BaseSystem(configs={"training": train_cfg}, exp_dir=tmp_path / "exp")

    assert system.exp_dir == tmp_path / "exp"


def test_base_system_raises_config_error_without_any_exp_dir():
    with pytest.raises(ConfigError):
        BaseSystem(configs={})


def test_base_system_collect_stats_passes_its_stage_config(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    system = BaseSystem(configs={"training": train_cfg})
    seen = {}

    def fake_collect(cfg):
        seen["cfg"] = cfg

    monkeypatch.setattr(sysmod, "collect_stats", fake_collect)

    system.collect_stats()

    assert seen["cfg"] is system.stage_configs["collect_stats"]


def test_base_system_measure_passes_metrics_and_inference_configs(
    tmp_path, monkeypatch
):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    infer_cfg = OmegaConf.create({"inference_dir": "${exp_dir}/infer"})
    metrics_cfg = OmegaConf.create({})
    system = BaseSystem(
        configs={
            "training": train_cfg,
            "inference": infer_cfg,
            "metrics": metrics_cfg,
        }
    )
    seen = {}

    def fake_measure(cfg, inference_config=None):
        seen["metrics_cfg"] = cfg
        seen["inference_cfg"] = inference_config
        return {}

    monkeypatch.setattr(sysmod, "measure", fake_measure)

    system.measure()

    assert seen["metrics_cfg"] is system.stage_configs["measure"]
    assert seen["inference_cfg"] is system.stage_configs["infer"]
    # measure has no inference_dir of its own; it inherits infer's.
    assert (
        system.stage_configs["measure"].inference_dir
        == str(tmp_path / "exp") + "/infer"
    )


def test_base_system_invokes_helpers(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    infer_cfg = OmegaConf.create({"inference_dir": "${exp_dir}/infer"})
    metrics_cfg = OmegaConf.create({})

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

    def fake_metric(cfg, inference_config=None):
        calls["measure"] = cfg
        calls["measure_inference_config"] = inference_config
        return {"metric": 1.0}

    monkeypatch.setattr(sysmod, "collect_stats", fake_collect)
    monkeypatch.setattr(sysmod, "train", fake_train)
    monkeypatch.setattr(sysmod, "infer", fake_infer)
    monkeypatch.setattr(sysmod, "measure", fake_metric)

    system = BaseSystem(
        configs={
            "training": train_cfg,
            "inference": infer_cfg,
            "metrics": metrics_cfg,
        }
    )

    assert system.collect_stats() == "collect"
    assert system.train() == "train"
    assert system.infer() == "infer"
    assert system.measure() == {"metric": 1.0}
    assert calls["collect"] is system.stage_configs["collect_stats"]
    assert calls["train"] is system.stage_configs["train"]
    assert calls["infer"] is system.stage_configs["infer"]


def test_base_system_pack_model_reads_train_infer_and_measure_by_name(
    tmp_path, monkeypatch
):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    infer_cfg = OmegaConf.create({"inference_dir": "${exp_dir}/infer"})
    metrics_cfg = OmegaConf.create({})
    publication_cfg = OmegaConf.create({"pack_model": {"out_dir": "./pack"}})
    seen = {}

    def fake_pack_model(
        *, training_config, publication_config, inference_config, metrics_config
    ):
        seen["training_config"] = training_config
        seen["publication_config"] = publication_config
        seen["inference_config"] = inference_config
        seen["metrics_config"] = metrics_config
        return "packed"

    monkeypatch.setattr(sysmod, "_pack_model", fake_pack_model)

    system = BaseSystem(
        configs={
            "training": train_cfg,
            "inference": infer_cfg,
            "metrics": metrics_cfg,
            "publication": publication_cfg,
        }
    )

    assert system.pack_model() == "packed"
    assert seen["training_config"] is system.stage_configs["train"]
    assert seen["inference_config"] is system.stage_configs["infer"]
    assert seen["metrics_config"] is system.stage_configs["measure"]
    assert seen["publication_config"] is system.stage_configs["pack_model"]


def test_base_system_create_dataset_requires_dataset_config(tmp_path):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    system = BaseSystem(configs={"training": train_cfg})
    with pytest.raises(RuntimeError, match="has no dataset"):
        system.create_dataset()


def test_base_system_create_dataset_prepares_dataset_references(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "create_dataset": {"archive_path": "a.tar.gz"},
            "dataset": {
                "train": [{"data_src": "mini_an4/esp2_asr"}],
                # Same source in valid; dedup means only one prepare run.
                "valid": [{"data_src": "mini_an4/esp2_asr"}],
                "test": None,
            },
        }
    )
    system = BaseSystem(configs={"training": train_cfg})
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
        lambda data_src=None, recipe_dir=None: DummyModule(),
    )

    assert system.create_dataset() is None
    expected_kwargs = {"archive_path": "a.tar.gz"}
    assert calls == [
        ("is_source_prepared", expected_kwargs),
        ("is_built", expected_kwargs),
    ]


def test_base_system_create_dataset_logs_progress(tmp_path, monkeypatch, caplog):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "dataset": {
                "train": [{"data_src": "mini_an4/esp2_asr"}],
                "valid": None,
                "test": None,
            },
        }
    )
    system = BaseSystem(configs={"training": train_cfg})

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
        lambda data_src=None, recipe_dir=None: DummyModule(),
    )

    with caplog.at_level(logging.INFO):
        system.create_dataset()

    assert "starting dataset creation process" in caplog.text
    assert "Ensuring dataset is prepared: mini_an4/esp2_asr" in caplog.text
    assert "Dataset creation completed" in caplog.text


def test_base_system_create_dataset_runs_prepare_and_build_when_needed(
    tmp_path, monkeypatch
):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "dataset": {
                "train": [{"data_src": "mini_an4/esp2_asr"}],
                "valid": None,
                "test": None,
            },
        }
    )
    system = BaseSystem(configs={"training": train_cfg})
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
        lambda data_src=None, recipe_dir=None: DummyModule(),
    )

    assert system.create_dataset() is None
    assert calls == [
        ("is_source_prepared", {}),
        ("prepare_source", {}),
        ("is_built", {}),
        ("build", {}),
    ]


def test_base_system_create_dataset_raises_when_no_dataset_entries(tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "dataset": {"train": None, "valid": None, "test": None},
        }
    )
    system = BaseSystem(configs={"training": train_cfg})
    with pytest.raises(RuntimeError, match="has no entry"):
        system.create_dataset()


def test_base_system_create_dataset_local_ref_dedup(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "recipe_dir": str(tmp_path / "recipe"),
            "dataset": {
                "train": [{"data_src_args": {"split": "train"}}],
                "valid": [{"data_src_args": {"split": "valid"}}],
                "test": None,
            },
        }
    )
    system = BaseSystem(configs={"training": train_cfg})
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
        lambda data_src=None, recipe_dir=None: DummyModule(),
    )

    assert system.create_dataset() is None
    # Local entries should be deduplicated and prepared only once.
    assert calls == [
        ("is_source_prepared", {}),
        ("is_built", {}),
    ]


def test_base_system_rejects_subclass_args(tmp_path):
    class CustomSystem(BaseSystem):
        def train(self, *, extra=None):
            return super().train(extra=extra)

    system = CustomSystem(
        configs={"training": OmegaConf.create({"exp_dir": str(tmp_path / "exp")})}
    )
    with pytest.raises(TypeError):
        system.train(extra="oops")
