"""Tests for the ESPnet3 enhancement system."""

from omegaconf import OmegaConf

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.esp2_enh import EnhancementSystem


def test_enhancement_system_keeps_stage_configs(tmp_path):
    train_cfg = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    infer_cfg = OmegaConf.create({"inference_dir": str(tmp_path / "infer")})
    metrics_cfg = OmegaConf.create({"metrics": []})

    system = EnhancementSystem(
        training_config=train_cfg,
        inference_config=infer_cfg,
        metrics_config=metrics_cfg,
    )

    assert isinstance(system, BaseSystem)
    assert system.training_config is train_cfg
    assert system.inference_config is infer_cfg
    assert system.metrics_config is metrics_cfg
