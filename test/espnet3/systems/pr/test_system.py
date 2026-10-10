"""Tests for the phone recognition system.

# Test Case Summary
| Test | Description |
|---|---|
| test_pr_system_is_a_base_system | Subclasses BaseSystem, not ASRSystem |
| test_pr_system_has_no_tokenizer_stage | The family exposes no train_tokenizer stage |
| test_pr_system_accepts_inference_only_configs | Builds with no training config |
| test_pr_system_resolves_stage_log_dirs | Stage log directories come from the configs |
"""

from omegaconf import OmegaConf

from espnet3.systems.asr.system import ASRSystem
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.pr.system import PRSystem


def test_pr_system_is_a_base_system():
    assert issubclass(PRSystem, BaseSystem)
    assert not issubclass(PRSystem, ASRSystem)


def test_pr_system_has_no_tokenizer_stage():
    # A phone inventory is fixed by the phone set, so there is nothing to train.
    assert not hasattr(PRSystem, "train_tokenizer")


def test_pr_system_accepts_inference_only_configs(tmp_path):
    inference_config = OmegaConf.create(
        {"exp_tag": "pretrained", "inference_dir": str(tmp_path / "infer")}
    )
    metrics_config = OmegaConf.create(
        {"exp_tag": "pretrained", "inference_dir": str(tmp_path / "infer")}
    )

    system = PRSystem(inference_config=inference_config, metrics_config=metrics_config)

    assert system.training_config is None
    assert system.exp_dir is None


def test_pr_system_resolves_stage_log_dirs(tmp_path):
    inference_dir = tmp_path / "infer"
    system = PRSystem(
        inference_config=OmegaConf.create(
            {"exp_tag": "pretrained", "inference_dir": str(inference_dir)}
        ),
        metrics_config=OmegaConf.create(
            {"exp_tag": "pretrained", "inference_dir": str(inference_dir)}
        ),
    )

    assert system.stage_log_dirs["infer"] == inference_dir
    assert system.stage_log_dirs["measure"] == inference_dir
