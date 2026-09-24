"""Exercise shared LM training, resuming and ESPnet2 scorer loading."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from espnet2.tasks.lm import LMTask
from espnet3.systems.esp2_asr.lm_system import LMSystem
from espnet3.utils.config_utils import load_and_merge_config


@pytest.fixture
def lm_config(tmp_path):
    """Build a small word-tokenized dataset for the real public trainer."""
    text = tmp_path / "text"
    text.write_text("first HELLO WORLD\nsecond WORLD\n")
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blank>\n<unk>\nHELLO\nWORLD\n<sos/eos>\n")
    path = tmp_path / "training_lm.yaml"
    dataset_entry = {
        "data_src": "espnet3.systems.esp2_asr.lm_dataset",
        "data_src_args": {"text_path": str(text)},
    }
    config = {
        "task": "espnet3.systems.esp2_asr.lm_task.LMTask",
        "seed": 0,
        "exp_dir": str(tmp_path / "exp"),
        "stats_dir": str(tmp_path / "stats"),
        "dataset": {
            "train": [dataset_entry],
            "valid": [dataset_entry],
            "preprocessor": {
                "_target_": "espnet2.train.preprocessor.CommonPreprocessor",
                "token_type": "word",
                "token_list": str(tokens),
            },
        },
        "model": {
            "token_list": str(tokens),
            "lm": "seq_rnn",
            "lm_conf": {"unit": 8, "nlayers": 1},
        },
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
        "scheduler": "${constant_scheduler}",
        "constant_scheduler": {
            "_target_": "torch.optim.lr_scheduler.ConstantLR",
            "factor": 1.0,
            "total_iters": 1,
        },
        "best_model_criterion": [["valid/loss", 1, "min"]],
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 2,
        },
        "dataloader": {
            "collate_fn": {"int_pad_value": 0},
            "train": {
                "iter_factory": {
                    "batches": {
                        "type": "folded",
                        "batch_size": 2,
                        "fold_lengths": [150],
                        "shape_files": ["${stats_dir}/train/text_shape"],
                    }
                }
            },
            "valid": {
                "iter_factory": {
                    "batches": {
                        "fold_lengths": [150],
                        "shape_files": ["${stats_dir}/valid/text_shape"],
                    }
                }
            },
        },
    }
    OmegaConf.save(OmegaConf.create(config), path)
    return load_and_merge_config(
        path, "training.yaml", default_package="egs3.TEMPLATE.esp2_asr"
    )


def test_text_shapes_match_public_tokenization(lm_config):
    """Use public tokenizer lengths and vocabulary dimensions for numel batches."""
    LMSystem(training_config=lm_config).collect_stats()
    for split in ("train", "valid"):
        shape = Path(lm_config.stats_dir) / split / "text_shape"
        assert shape.read_text().splitlines() == ["0 2,5", "1 1,5"]


@pytest.mark.execution_timeout(90)
@pytest.mark.parametrize("architecture", ["seq_rnn", "transformer"])
def test_training_resume_and_shallow_fusion_checkpoint(lm_config, architecture):
    """Train each LM, reload its exported scorer, and resume Lightning state."""
    if architecture == "transformer":
        lm_config.model.lm = architecture
        lm_config.model.lm_conf = {
            "embed_unit": 8,
            "att_unit": 8,
            "head": 2,
            "unit": 16,
            "layer": 1,
        }
    system = LMSystem(training_config=lm_config)
    system.collect_stats()
    system.train()
    exp = Path(lm_config.exp_dir)
    model, _ = LMTask.build_model_from_file(
        exp / "config.yaml", exp / "valid.loss.ave_1best.pth"
    )
    loss, _, _ = model(torch.tensor([[2, 3]]), torch.tensor([2]))
    assert torch.isfinite(loss)
    state = torch.load(exp / "last.ckpt", weights_only=False, map_location="cpu")
    assert state["global_step"] == 2
    lm_config.fit.ckpt_path = str(exp / "last.ckpt")
    lm_config.trainer.max_epochs = 3
    system.train()
    resumed = torch.load(exp / "last.ckpt", weights_only=False, map_location="cpu")
    assert resumed["global_step"] == 3
