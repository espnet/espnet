"""Generate a checkpoint for Encoder from Pretraining model checkpoint."""

import argparse
import logging
import os
import sys

import torch
import yaml

from espnet.utils.cli_utils import get_commandline_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def generate_beats_config(config, drop_tokenizer_keys=False):
    beats_config = config.get("encoder_conf", {}).get("beats_config", {})
    if len(beats_config) == 0:
        # Case for tokenizer checkpoint
        beats_config = config.get("encoder_conf", {}).get("tokenizer_config", {})
        if drop_tokenizer_keys:
            extra_keys_in_tokenizer = [
                "quant_dim",
                "quant_n",  # equal to codebook_vocab_size
                "embed_loss_beta",
            ]
            for key in extra_keys_in_tokenizer:
                beats_config.pop(key, None)

    logger.info(f"Beats config: {beats_config}")
    return beats_config


def generate_beats_encoder_checkpoint(
    espnet_state_dict: dict,
    deepspeed_checkpoint: bool = False,
    lightning_checkpoint: bool = False,
    key_prefix: str = "encoder.",
):
    """Extract encoder weights from ESPnet checkpoint."""
    model_state_dict = {}
    if deepspeed_checkpoint:
        espnet_state_dict = espnet_state_dict["module"]
    if lightning_checkpoint:
        espnet_state_dict = espnet_state_dict["state_dict"]
    for key, value in espnet_state_dict.items():
        if key.startswith(key_prefix):
            model_state_dict[key[len(key_prefix) :]] = value.to(dtype=torch.float32)
    return model_state_dict


def average_checkpoints(
    checkpoint_paths,
    deepspeed_checkpoint=False,
    lightning_checkpoint=False,
    key_prefix="encoder.",
):
    """Average multiple checkpoints."""
    avg_state_dict = None
    num_checkpoints = len(checkpoint_paths)
    expected_keys = None

    for checkpoint_path in checkpoint_paths:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder_state_dict = generate_beats_encoder_checkpoint(
            state_dict,
            deepspeed_checkpoint,
            lightning_checkpoint,
            key_prefix=key_prefix,
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


def convert_checkpoint(
    espnet_model_checkpoint_paths: list,
    espnet_model_config_path: str,
    output_path: str,
    deepspeed_checkpoint: bool,
    lightning_checkpoint: bool,
    drop_tokenizer_keys: bool = False,
    finetuned_checkpoint: bool = False,
):
    """Read and write checkpoint, with optional averaging."""
    num_checkpoints = len(espnet_model_checkpoint_paths)
    output_path = (
        os.path.join(
            os.path.dirname(output_path),
            f"{num_checkpoints}avg.{os.path.basename(output_path)}",
        )
        if num_checkpoints > 1
        else output_path
    )
    # Prepare checkpoint
    if finetuned_checkpoint:
        key_prefix = "model." if lightning_checkpoint else ""
    else:
        key_prefix = "encoder."
    logger.info(
        f"Reading and Averaging {num_checkpoints} checkpoints. Expected key prefix: {key_prefix}"
    )
    encoder_state_dict = average_checkpoints(
        espnet_model_checkpoint_paths,
        deepspeed_checkpoint,
        lightning_checkpoint,
        key_prefix=key_prefix,
    )
    logger.info("Number of keys in state dict: " f"{len(encoder_state_dict)}")
    # Prepare config
    logger.info(f"Reading config from {espnet_model_config_path}")
    with open(espnet_model_config_path, "r") as f:
        config = yaml.safe_load(f)
    encoder_config = generate_beats_config(config, drop_tokenizer_keys)

    checkpoint = {"model": encoder_state_dict, "cfg": encoder_config}
    if finetuned_checkpoint:
        # Overwrite if it is a finetuned checkpoint
        checkpoint = handle_finetuned_checkpoint(checkpoint, config)
    logger.info(f"Writing checkpoint to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)


def handle_finetuned_checkpoint(checkpoint, config):
    beats_pt_ckpt_path = config.get("encoder_conf", {}).get("beats_ckpt_path", {})
    if not beats_pt_ckpt_path:
        raise ValueError(
            "Finetuned checkpoint is specified, but no beats_ckpt_path found in config."
        )
    logger.info(
        f"Handling finetuned checkpoint, getting encoder config from "
        f"pretrained checkpoint at {beats_pt_ckpt_path}"
    )
    if not os.path.exists(beats_pt_ckpt_path):
        raise FileNotFoundError(
            f"Beats pretrained checkpoint {beats_pt_ckpt_path} does not exist."
        )
    pt_ckpt = torch.load(beats_pt_ckpt_path, map_location="cpu", weights_only=False)
    if "cfg" not in pt_ckpt:
        raise ValueError(
            f"Pretrained checkpoint {beats_pt_ckpt_path} does not contain 'cfg' key."
        )
    # extract pretrained encoder config
    pt_encoder_config = pt_ckpt["cfg"]
    logger.info(f"Pretrained encoder config: {pt_encoder_config}")
    # overwrite some components of pretrained config with finetuned config
    pt_encoder_config.update(generate_beats_config(config, drop_tokenizer_keys=True))
    logger.info(f"Updated encoder config: {pt_encoder_config}")
    token_list = config.get("token_list", None)
    checkpoint["token_list"] = token_list
    checkpoint["cfg"] = pt_encoder_config
    return checkpoint


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
        help="Is DeepSpeed checkpoint? If so, it will extract the state_dict from the 'module' key.",
    )
    parser.add_argument(
        "--lightning_checkpoint",
        action="store_true",
        default=False,
        help="Is Lightning checkpoint? If so, it will extract the state_dict from the 'state_dict' key.",
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
    parser.add_argument(
        "--drop_tokenizer_keys",
        action="store_true",
        default=False,
        help="Drop tokenizer keys from config",
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        action="store_true",
        default=False,
        help="Is this a finetuned checkpoint? If so, it will extract "
        "the pretrained encoder config from the beats_ckpt_path in the config.",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_cmdline_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    logger.info(f"Kwargs: {kwargs}")
    convert_checkpoint(**kwargs)


if __name__ == "__main__":
    main()
