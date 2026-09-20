"""Compare the migrated model and training configuration to ESPnet2."""

from pathlib import Path

import pytest
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

ROOT = Path(__file__).resolve().parents[5]
RECIPE = ROOT / "egs3/voices/asr"


def test_model_and_training_settings_match_source():
    """Catch dropped Conformer, SpecAugment, optimizer, or trainer settings."""
    source = yaml.safe_load(
        (ROOT / "egs2/voices/asr1/conf/train_asr_conformer.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_conformer.yaml", "training.yaml"
    )
    for key in (
        "encoder",
        "encoder_conf",
        "decoder",
        "decoder_conf",
        "model_conf",
        "frontend_conf",
        "specaug",
        "specaug_conf",
    ):
        assert config.model[key] == source[key]
    optimizer = instantiate(config.optimizer, [torch.nn.Parameter(torch.ones(1))])
    for key, value in source["optim_conf"].items():
        assert optimizer.param_groups[0][key] == value
    scheduler = instantiate(config.scheduler, optimizer=optimizer)
    assert scheduler.warmup_steps == source["scheduler_conf"]["warmup_steps"]
    assert config.trainer.max_epochs == source["max_epoch"]
    assert config.espnet2_compat.accum_grad == source["accum_grad"]
    assert config.trainer.precision == "16-mixed" and source["use_amp"]
    assert config.best_model_criterion == [["valid/acc", 10, "max"]]
    assert config.dataset._recursive_ is False
    assert config.tokenizer.model_type == "unigram"
    assert (
        config.dataloader.train.iter_factory.batches.batch_bins == source["batch_bins"]
    )


@pytest.mark.execution_timeout(60)
def test_model_matches_native_espnet2_initial_state():
    """Compare every parameter in the full Conformer under the same seed."""
    from espnet3.utils.task_utils import get_espnet_model

    original = yaml.safe_load(
        (ROOT / "egs2/voices/asr1/conf/train_asr_conformer.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_conformer.yaml", "training.yaml"
    )
    migrated = OmegaConf.to_container(config.model, resolve=True)
    tokens = ["<blank>", "<unk>", *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "▁", "<sos/eos>"]
    for arguments in (original, migrated):
        # ASRTask's parser converts the YAML string "none" to Python None.
        arguments.update(
            token_list=tokens, normalize=None, normalize_conf={}, init=None
        )
    torch.manual_seed(0)
    before = get_espnet_model("espnet2.tasks.asr.ASRTask", original).state_dict()
    torch.manual_seed(0)
    after = get_espnet_model("espnet3.systems.asr.task.ASRTask", migrated).state_dict()
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
