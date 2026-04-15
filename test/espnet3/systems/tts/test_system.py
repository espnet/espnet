import torch
from omegaconf import OmegaConf

import espnet3.systems.tts.system as sysmod
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.systems.tts.system import TTSSystem


class DummyTrainer:
    def __init__(self):
        self.collect_stats_called = False
        self.fit_called = False
        self.fit_kwargs = None

    def collect_stats(self):
        self.collect_stats_called = True

    def fit(self, **kwargs):
        self.fit_called = True
        self.fit_kwargs = kwargs


class DummyGANModel(AbsGANESPnetModel):
    def forward(self, forward_generator: bool = True, **batch):
        return {
            "loss": torch.tensor(1.0, requires_grad=True),
            "stats": {},
            "weight": torch.tensor(1.0),
            "optim_idx": 0 if forward_generator else 1,
        }

    def collect_feats(self, **batch):
        return {}


def test_load_function_resolves_dotted_path():
    assert sysmod.load_function("math.sqrt")(9) == 3


def test_instantiate_model_uses_task_builder(monkeypatch):
    cfg = OmegaConf.create({"task": "dummy.task", "model": {"foo": "bar"}})
    calls = {}

    def fake_get_model(task, model_config):
        calls["task"] = task
        calls["model"] = model_config
        return "task-model"

    monkeypatch.setattr(sysmod, "get_espnet_model", fake_get_model)

    assert sysmod._instantiate_model(cfg) == "task-model"
    assert calls == {"task": "dummy.task", "model": {"foo": "bar"}}


def test_instantiate_model_uses_hydra_when_task_is_unset(monkeypatch):
    cfg = OmegaConf.create({"model": {"_target_": "dummy.Target"}})
    monkeypatch.setattr(sysmod, "instantiate", lambda model: ("instantiated", model))

    assert sysmod._instantiate_model(cfg) == ("instantiated", cfg.model)


def test_tts_system_create_dataset_invokes_helper(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "create_dataset": {"func": "dummy.dataset", "foo": "bar"},
        }
    )
    system = TTSSystem(training_config=train_cfg)
    calls = {}

    def fake_fn(**kwargs):
        calls["kwargs"] = kwargs
        return "created"

    monkeypatch.setattr(sysmod, "load_function", lambda path: fake_fn)

    assert system.create_dataset() == "created"
    assert calls["kwargs"] == {"foo": "bar"}


def test_tts_system_ensure_directories_creates_exp_and_stats(tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
        }
    )
    system = TTSSystem(training_config=train_cfg)

    system._ensure_directories()

    assert (tmp_path / "exp").is_dir()
    assert (tmp_path / "stats").is_dir()


def test_tts_system_build_trainer_uses_gan_builder(monkeypatch, tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "trainer": {"accelerator": "cpu"},
            "best_model_criterion": [["valid/loss", 1, "min"]],
        }
    )
    system = TTSSystem(training_config=train_cfg)
    calls = {}

    monkeypatch.setattr(sysmod, "_instantiate_model", lambda _cfg: DummyGANModel())
    monkeypatch.setattr(
        sysmod,
        "build_gan_trainer",
        lambda config, model: calls.setdefault("gan", (config, model)) or "gan-trainer",
    )

    trainer = system._build_trainer()

    assert trainer == calls["gan"]


def test_tts_system_build_trainer_uses_standard_trainer(monkeypatch, tmp_path):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "trainer": {"accelerator": "cpu"},
            "best_model_criterion": [["valid/loss", 1, "min"]],
        }
    )
    system = TTSSystem(training_config=train_cfg)
    calls = {}

    monkeypatch.setattr(
        sysmod, "_instantiate_model", lambda _cfg: torch.nn.Linear(1, 1)
    )

    class DummyLit:
        def __init__(self, model, config):
            calls["lit"] = (model, config)

    class DummyTrainerClass:
        def __init__(self, model, exp_dir, config, best_model_criterion):
            calls["trainer"] = (model, exp_dir, config, best_model_criterion)

    monkeypatch.setattr(sysmod, "ESPnetLightningModule", DummyLit)
    monkeypatch.setattr(sysmod, "ESPnet3LightningTrainer", DummyTrainerClass)

    trainer = system._build_trainer()

    assert isinstance(trainer, DummyTrainerClass)
    assert calls["trainer"][1:] == (
        str(tmp_path / "exp"),
        train_cfg.trainer,
        train_cfg.best_model_criterion,
    )


def test_tts_system_prepare_training_runtime_sets_parallel_seed_and_precision(
    monkeypatch, tmp_path
):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "parallel": {"backend": "dummy"},
            "seed": 123,
        }
    )
    system = TTSSystem(training_config=train_cfg)
    calls = {}

    monkeypatch.setattr(
        sysmod, "set_parallel", lambda arg: calls.setdefault("parallel", arg)
    )
    monkeypatch.setattr(
        sysmod.L,
        "seed_everything",
        lambda seed, workers=True: calls.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        sysmod.torch,
        "set_float32_matmul_precision",
        lambda precision: calls.setdefault("precision", precision),
    )

    system._prepare_training_runtime()

    assert calls == {
        "parallel": {"backend": "dummy"},
        "seed": 123,
        "precision": "high",
    }


def test_tts_system_collect_stats_runs_trainer(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
        }
    )
    system = TTSSystem(training_config=train_cfg)
    trainer = DummyTrainer()

    monkeypatch.setattr(system, "_prepare_training_runtime", lambda: None)
    monkeypatch.setattr(system, "_build_trainer", lambda: trainer)

    system.collect_stats()

    assert trainer.collect_stats_called is True


def test_tts_system_train_saves_config_and_calls_fit(tmp_path, monkeypatch):
    train_cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "task": "tts",
            "model": {"_target_": "dummy.Target"},
            "fit": {"max_epochs": 1},
        }
    )
    system = TTSSystem(training_config=train_cfg)
    trainer = DummyTrainer()
    calls = {}

    monkeypatch.setattr(system, "_prepare_training_runtime", lambda: None)
    monkeypatch.setattr(system, "_build_trainer", lambda: trainer)
    monkeypatch.setattr(
        sysmod,
        "save_espnet_config",
        lambda task, cfg, exp_dir: calls.setdefault("save", (task, exp_dir)),
    )

    system.train()

    assert trainer.fit_called is True
    assert trainer.fit_kwargs == {"max_epochs": 1}
    assert calls["save"] == ("tts", str(tmp_path / "exp"))
