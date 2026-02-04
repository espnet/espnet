from omegaconf import OmegaConf

import espnet3.systems.base.training as train_mod


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


def test_ensure_directories_creates_exp_and_stats(tmp_path):
    cfg = OmegaConf.create(
        {"exp_dir": str(tmp_path / "exp"), "stats_dir": str(tmp_path / "stats")}
    )

    train_mod._ensure_directories(cfg)

    assert (tmp_path / "exp").is_dir()
    assert (tmp_path / "stats").is_dir()


def test_collect_stats_runs_pipeline(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "seed": 777,
            "parallel": {"backend": "dummy"},
            "model": {"normalize": True, "normalize_conf": {"foo": "bar"}},
        }
    )
    trainer = DummyTrainer()
    calls = {}

    def fake_set_parallel(arg):
        calls.setdefault("parallel", arg)

    def fake_seed_everything(seed, workers=True):
        calls.setdefault("seed", seed)

    def fake_set_precision(precision):
        calls.setdefault("precision", precision)

    monkeypatch.setattr(train_mod, "_build_trainer", lambda _cfg: trainer)
    monkeypatch.setattr(train_mod, "set_parallel", fake_set_parallel)
    monkeypatch.setattr(train_mod.L, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(
        train_mod.torch, "set_float32_matmul_precision", fake_set_precision
    )

    train_mod.collect_stats(cfg)

    assert trainer.collect_stats_called
    assert calls["parallel"] == {"backend": "dummy"}
    assert calls["seed"] == 777
    assert calls["precision"] == "high"
    assert "normalize" not in cfg.model
    assert "normalize_conf" not in cfg.model


def test_train_saves_config_and_calls_fit(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "seed": 123,
            "parallel": {"backend": "dummy"},
            "task": "asr",
            "model": {"_target_": "dummy.Target"},
            "fit": {"max_epochs": 1},
        }
    )
    trainer = DummyTrainer()
    calls = {}

    def fake_set_parallel(arg):
        calls.setdefault("parallel", arg)

    def fake_seed_everything(seed, workers=True):
        calls.setdefault("seed", seed)

    def fake_set_precision(precision):
        calls.setdefault("precision", precision)

    def fake_save_config(task, cfg_arg, exp_dir):
        calls.setdefault("save", (task, exp_dir))

    monkeypatch.setattr(train_mod, "_build_trainer", lambda _cfg: trainer)
    monkeypatch.setattr(train_mod, "set_parallel", fake_set_parallel)
    monkeypatch.setattr(train_mod.L, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(
        train_mod.torch, "set_float32_matmul_precision", fake_set_precision
    )
    monkeypatch.setattr(train_mod, "save_espnet_config", fake_save_config)

    train_mod.train(cfg)

    assert trainer.fit_called
    assert trainer.fit_kwargs == {"max_epochs": 1}
    assert calls["parallel"] == {"backend": "dummy"}
    assert calls["seed"] == 123
    assert calls["precision"] == "high"
    assert calls["save"] == ("asr", str(tmp_path / "exp"))
