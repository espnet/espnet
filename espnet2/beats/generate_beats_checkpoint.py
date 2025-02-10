"""Generate a checkpoint for BEATs Encoder from BEATs pretraining model checkpoint.

Usage:
    python generate_beats_checkpoint.py \
        --espnet_model_checkpoint_path espnet_model_checkpoint_path \
        --espnet_model_config_path espnet_model_config_path \
        --output_path output_path \
        [--deepspeed_ckpt]
    Deepspeed checkpoint is optional. If the checkpoint is generated using Deepspeed, 
    then pass --deepspeed_ckpt.
    We consider keys inside the "module" key for deepspeed checkpoints.
"""

import argparse
import logging
import sys
import torch
import os
import yaml


from espnet.utils.cli_utils import get_commandline_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def generate_beats_config(config):
    beats_config = config.get("encoder_conf", {}).get("beats_config", {})
    logger.info(f"Beats config: {beats_config}")
    return beats_config


def generate_beats_encoder_checkpoint(
    espnet_state_dict: dict,
    deepspeed_ckpt: bool = False,
):
    """Generate a checkpoint for Encoder from Pretraining model checkpoint."""
    model_state_dict = {}
    if deepspeed_ckpt:
        espnet_state_dict = espnet_state_dict["module"]
    for key, value in espnet_state_dict.items():
        if key.startswith("encoder."):
            model_state_dict[key[len("encoder.") :]] = value.to(dtype=torch.float32)
    return model_state_dict


def read_and_write_checkpoint(
    espnet_model_checkpoint_path: str,
    espnet_model_config_path: str,
    output_path: str,
    deepspeed_ckpt: bool,
):
    """Read and write checkpoint."""
    logger.info(f"Reading checkpoint from {espnet_model_checkpoint_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    espnet_state_dict = torch.load(
        espnet_model_checkpoint_path, map_location="cpu", weights_only=False
    )
    encoder_state_dict = generate_beats_encoder_checkpoint(
        espnet_state_dict, deepspeed_ckpt
    )
    logger.info(f"Reading config from {espnet_model_config_path}")
    with open(espnet_model_config_path, "r") as f:
        config = yaml.safe_load(f)
    encoder_config = generate_beats_config(config)
    logger.info(f"Writing checkpoint to {output_path}")
    checkpoint = {"model": encoder_state_dict, "cfg": encoder_config}
    torch.save(checkpoint, output_path)


def get_cmdline_parser():
    parser = argparse.ArgumentParser(
        description="Generate a checkpoint for Encoder from Pretraining model checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--espnet_model_checkpoint_path",
        type=str,
        required=True,
        help="Path to ESPnet model checkpoint",
    )
    parser.add_argument(
        "--deepspeed_ckpt",
        action="store_true",
        default=False,
        help="Is DeepSpeed checkpoint?",
    )
    parser.add_argument(
        "--espnet_model_config_path",
        type=str,
        required=True,
        help="Path to ESPnet model config",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the new checkpoint",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_cmdline_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    logger.info(f"Kwargs: {kwargs}")
    read_and_write_checkpoint(**kwargs)


if __name__ == "__main__":
    main()
