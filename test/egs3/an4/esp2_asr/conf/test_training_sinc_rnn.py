"""Regression tests for full AN4 preparation and migration settings."""

from pathlib import Path

import pytest
import yaml
from hydra.utils import instantiate

from espnet3.utils.config_utils import load_and_merge_config

ROOT = Path(__file__).resolve().parents[5]
RECIPE = ROOT / "egs3/an4/esp2_asr"


def test_source_model_and_optimizer_equivalence():
    """Instantiate the merged config and compare it to the source recipe."""
    import torch

    source = yaml.safe_load(
        (ROOT / "egs2/an4/asr1/conf/train_asr_sinc_rnn.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_sinc_rnn.yaml", "training.yaml"
    )
    for key in (
        "init",
        "frontend",
        "frontend_conf",
        "preencoder",
        "encoder",
        "encoder_conf",
        "decoder",
        "decoder_conf",
    ):
        assert config.model[key] == source[key]
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = instantiate(config.optimizer, [parameter])
    scheduler = instantiate(config.scheduler, optimizer=optimizer)
    assert isinstance(optimizer, torch.optim.Adadelta)
    for key, value in source["optim_conf"].items():
        assert optimizer.param_groups[0][key] == value
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert "warmup_steps" not in config.scheduler
    for key, value in source["scheduler_conf"].items():
        assert getattr(scheduler, key) == value
    assert config.trainer.max_epochs == source["max_epoch"]
    assert config.trainer.callbacks[0].patience == source["patience"] + 1
    assert config.dataset._recursive_ is False
    assert config.dataloader.train.iter_factory.batches.fold_lengths == [334]


@pytest.mark.execution_timeout(60)
def test_model_matches_espnet2_initial_state():
    """Build both task models with one seed and compare all initialized tensors."""
    import torch
    from omegaconf import OmegaConf

    from espnet3.utils.task_utils import get_espnet_model

    source = yaml.safe_load(
        (ROOT / "egs2/an4/asr1/conf/train_asr_sinc_rnn.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_sinc_rnn.yaml", "training.yaml"
    )
    target = OmegaConf.to_container(config.model, resolve=True)
    tokens = ["<blank>", "<unk>", *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "▁", "<sos/eos>"]
    for arguments in (source, target):
        arguments.update(token_list=tokens, normalize=None, normalize_conf={})
    torch.manual_seed(0)
    original = get_espnet_model("espnet2.tasks.asr.ASRTask", source)
    torch.manual_seed(0)
    migrated = get_espnet_model("espnet3.systems.esp2_asr.task.ASRTask", target)
    assert original.ctc_weight == migrated.ctc_weight == 0.5
    before, after = original.state_dict(), migrated.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_early_stopping_matches_espnet2_patience():
    """Both trainers must tolerate four bad epochs and stop on the fifth."""
    import torch
    from lightning.pytorch.callbacks import EarlyStopping

    config = load_and_merge_config(
        RECIPE / "conf/training_sinc_rnn.yaml", "training.yaml"
    )
    callback = instantiate(config.trainer.callbacks[0])
    assert isinstance(callback, EarlyStopping)
    stopped = [
        callback._evaluate_stopping_criteria(torch.tensor(loss))[0]
        for loss in [1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    ]
    assert stopped == [False, False, False, False, False, True]
