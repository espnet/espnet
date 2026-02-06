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
    metric_cfg = OmegaConf.create({"inference_dir": str(tmp_path / "infer")})

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
        calls["metric"] = cfg
        return {"metric": 1.0}

    monkeypatch.setattr(sysmod, "collect_stats", fake_collect)
    monkeypatch.setattr(sysmod, "train", fake_train)
    monkeypatch.setattr(sysmod, "infer", fake_infer)
    monkeypatch.setattr(sysmod, "metric", fake_metric)

    system = BaseSystem(
        train_config=train_cfg,
        infer_config=infer_cfg,
        metric_config=metric_cfg,
    )

    assert system.exp_dir.is_dir()
    assert system.collect_stats() == "collect"
    assert system.train() == "train"
    assert system.infer() == "infer"
    assert system.metric() == {"metric": 1.0}
    assert calls["collect"] is train_cfg
    assert calls["train"] is train_cfg
    assert calls["infer"] is infer_cfg
    assert calls["metric"] is metric_cfg


def test_base_system_rejects_subclass_args():
    class CustomSystem(BaseSystem):
        def train(self, *, extra=None):
            return super().train(extra=extra)

    system = CustomSystem()
    with pytest.raises(TypeError):
        system.train(extra="oops")
