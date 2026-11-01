"""Shared tiny-codec helpers for the espnet3 LACE test suite."""

import copy
from pathlib import Path

import torch
import yaml

from espnet3.utils.task_utils import get_espnet_model

TASK = "espnet2.tasks.gan_codec.GANCodecTask"

TINY_ENCODEC_CONF = {
    "sampling_rate": 120,
    "generator_params": {
        "hidden_dim": 8,
        "encdec_n_filters": 4,
        "encdec_n_residual_layers": 1,
        "encdec_ratios": [2, 2],
        "encdec_kernel_size": 1,
        "encdec_residual_kernel_size": 1,
        "encdec_last_kernel_size": 1,
        "encdec_lstm": 1,
        "quantizer_n_q": 4,
        "quantizer_bins": 8,
        "quantizer_kmeans_init": False,
        "quantizer_target_bandwidth": [7.5, 15],
    },
    "discriminator_params": {
        "msstft_discriminator_params": {
            "filters": 4,
            "in_channels": 1,
            "out_channels": 1,
            "sep_channels": False,
            "norm": "weight_norm",
            "n_ffts": [32, 16],
            "hop_lengths": [8, 4],
            "win_lengths": [32, 16],
            "activation": "LeakyReLU",
            "activation_params": {"negative_slope": 0.3},
        }
    },
    "use_mel_loss": False,
    "use_dual_decoder": False,
}

# Encoder hop of TINY_ENCODEC_CONF (product of encdec_ratios).
HOP_LENGTH = 4
HIDDEN_DIM = 8
NUM_CODEBOOKS = 4


def build_tiny_codec_conf():
    """Return a fresh copy of the tiny Encodec ``codec_conf``."""
    return copy.deepcopy(TINY_ENCODEC_CONF)


def build_tiny_codec_model(seed=0):
    """Build the unwrapped tiny Encodec ``ESPnetGANCodecModel``."""
    torch.manual_seed(seed)
    return get_espnet_model(
        TASK, {"codec": "encodec", "codec_conf": build_tiny_codec_conf()}
    )


def write_tiny_bundle(bundle_dir, layout="espnet2", lightning=False, seed=1):
    """Write a tiny-Encodec model bundle (``meta.yaml`` + config + weights).

    ``layout="espnet2"`` mimics an espnet_model_zoo model (``train_config``
    is the espnet2 ``config.yaml``); ``layout="espnet3"`` mimics an espnet3
    ``pack_model`` output (``training_config`` carries ``task`` + ``model``).
    Returns the unwrapped model whose weights the bundle holds.
    """
    bundle_dir = Path(bundle_dir)
    (bundle_dir / "exp").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "conf").mkdir(exist_ok=True)
    model = build_tiny_codec_model(seed=seed)
    model_config = {"codec": "encodec", "codec_conf": build_tiny_codec_conf()}

    checkpoint = "exp/last.ckpt" if lightning else "exp/model.pth"
    state_dict = model.state_dict()
    torch.save(
        {"state_dict": state_dict} if lightning else state_dict,
        bundle_dir / checkpoint,
    )
    if layout == "espnet2":
        (bundle_dir / "exp" / "config.yaml").write_text(yaml.safe_dump(model_config))
        yaml_files = {"train_config": "exp/config.yaml"}
    elif layout == "espnet3":
        (bundle_dir / "conf" / "training.yaml").write_text(
            yaml.safe_dump({"recipe_dir": ".", "task": TASK, "model": model_config})
        )
        yaml_files = {"training_config": "conf/training.yaml"}
    else:
        raise ValueError(layout)
    (bundle_dir / "meta.yaml").write_text(
        yaml.safe_dump({"files": {"model_file": checkpoint}, "yaml_files": yaml_files})
    )
    return model
