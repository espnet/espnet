"""Compare the migrated model and training configuration to ESPnet2."""

from pathlib import Path

import pytest
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

ROOT = Path(__file__).resolve().parents[5]
RECIPE = ROOT / "egs3/voices/esp2_asr"


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
    assert config.trainer.accumulate_grad_batches == source["accum_grad"]
    assert config.trainer.precision == "bf16-mixed" and source["use_amp"]
    assert config.best_model_criterion == [["valid/acc", 10, "max"]]
    assert config.dataset._recursive_ is False
    assert config.tokenizer.model_type == "unigram"
    batches = config.dataloader.train.iter_factory.batches
    assert batches.type == "numel"
    assert list(batches.shape_files) == [f"{config.stats_dir}/train/feats_shape"]


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
    after = get_espnet_model(
        "espnet3.systems.esp2_asr.task.ASRTask", migrated
    ).state_dict()
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_devkit_config_is_self_contained():
    """Keep full/devkit settings equivalent without a sibling defaults lookup."""
    full = load_and_merge_config(
        RECIPE / "conf/training_conformer.yaml", "training.yaml", resolve=False
    )
    devkit = load_and_merge_config(
        RECIPE / "conf/training_devkit.yaml", "training.yaml", resolve=False
    )
    assert "defaults" not in yaml.safe_load(
        (RECIPE / "conf/training_devkit.yaml").read_text()
    )
    assert devkit.create_dataset.corpus == "devkit"
    devkit.create_dataset.corpus = full.create_dataset.corpus
    devkit.exp_tag = full.exp_tag
    assert OmegaConf.to_container(devkit) == OmegaConf.to_container(full)
