"""Validate the LM model and training mapping to the shared LMSystem."""

from pathlib import Path

import yaml

from espnet3.utils.config_utils import load_and_merge_config

ROOT = Path(__file__).resolve().parents[5]
RECIPE = ROOT / "egs3/an4/esp2_asr"


def test_lm_settings_preserve_source_model():
    """Keep the source LM architecture and optimizer through ESPnet3 configs."""
    source = yaml.safe_load((ROOT / "egs2/an4/asr1/conf/train_lm.yaml").read_text())
    config = load_and_merge_config(RECIPE / "conf/training_lm.yaml", "training.yaml")
    assert config.task == "espnet3.systems.esp2_asr.lm_task.LMTask"
    assert config.model.lm == source["lm"]
    assert config.model.lm_conf == source["lm_conf"]
    assert config.optimizer.lr == source["optim_conf"]["lr"]
    assert config.optimizer.weight_decay == 0
    assert config.scheduler._target_ == "torch.optim.lr_scheduler.ConstantLR"
    assert config.scheduler.factor == 1.0
    assert "warmup_steps" not in config.scheduler
    assert config.trainer.max_epochs == source.get("max_epoch", 40)
    assert config.trainer.gradient_clip_val == source["grad_clip"]
    assert config.trainer.accumulate_grad_batches == source.get("accum_grad", 1)
    assert config.dataloader.collate_fn.int_pad_value == 0
    assert config.best_model_criterion == [
        ["valid/loss", source["keep_nbest_models"], "min"]
    ]
