"""Generate a checkpoint for Encoder from Pretraining model checkpoint."""

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
    espnet_state_dict: dict, deepspeed_checkpoint: bool = False
):
    """Extract encoder weights from ESPnet checkpoint."""
    model_state_dict = {}
    if deepspeed_checkpoint:
        espnet_state_dict = espnet_state_dict["module"]
    for key, value in espnet_state_dict.items():
        if key.startswith("encoder."):
            model_state_dict[key[len("encoder.") :]] = value.to(dtype=torch.float32)
    return model_state_dict


def average_checkpoints(checkpoint_paths, deepspeed_checkpoint=False):
    """Average multiple checkpoints."""
    avg_state_dict = None
    num_checkpoints = len(checkpoint_paths)
    expected_keys = None

    for checkpoint_path in checkpoint_paths:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder_state_dict = generate_beats_encoder_checkpoint(
            state_dict, deepspeed_checkpoint
        )

        if avg_state_dict is None:
            avg_state_dict = {k: v.clone() for k, v in encoder_state_dict.items()}
            expected_keys = set(encoder_state_dict.keys())
        else:
            assert (
                set(encoder_state_dict.keys()) == expected_keys
            ), f"Checkpoint {checkpoint_path} has mismatched keys!"

            for k in avg_state_dict:
                avg_state_dict[k] += encoder_state_dict[k]

    for k in avg_state_dict:
        avg_state_dict[k] /= num_checkpoints

    return avg_state_dict


def read_and_write_checkpoint(
    espnet_model_checkpoint_paths: list,
    espnet_model_config_path: str,
    output_path: str,
    deepspeed_checkpoint: bool,
):
    """Read and write checkpoint, with optional averaging."""
    num_checkpoints = len(espnet_model_checkpoint_paths)
    output_path = os.path.join(
        os.path.dirname(output_path),
        f"{num_checkpoints}_avg{os.path.basename(output_path)}",
    )
    # Prepare checkpoint
    logger.info(f"Reading and Averaging {num_checkpoints} checkpoints.")
    encoder_state_dict = average_checkpoints(
        espnet_model_checkpoint_paths, deepspeed_checkpoint
    )
    # Prepare config
    logger.info(f"Reading config from {espnet_model_config_path}")
    with open(espnet_model_config_path, "r") as f:
        config = yaml.safe_load(f)
    encoder_config = generate_beats_config(config)
    logger.info(f"Writing checkpoint to {output_path}")
    # Write
    checkpoint = {"model": encoder_state_dict, "cfg": encoder_config}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)


def get_cmdline_parser():
    parser = argparse.ArgumentParser(
        description="Generate a checkpoint for Encoder from Pretraining model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--espnet_model_checkpoint_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to ESPnet model checkpoints. If multiple paths are provided, they will be averaged.",
    )
    parser.add_argument(
        "--deepspeed_checkpoint",
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
