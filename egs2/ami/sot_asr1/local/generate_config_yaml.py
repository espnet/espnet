#!/usr/bin/env python3
"""Generate ESPnet config.yaml from training config + token list.

Produces the full config.yaml that sot_inference's build_model_from_file()
expects, without running training. This is needed when loading a converted
(non-ESPnet-trained) checkpoint.

Usage:
    python local/generate_config_yaml.py \
        --config conf/tuning/train_sot.yaml \
        --token_list exp/token_list.txt \
        --output exp/sot_converted/config.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml

from espnet2.tasks.sot_asr import SOTASRTask


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESPnet config.yaml for inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (e.g. conf/tuning/train_sot.yaml)",
    )
    parser.add_argument(
        "--token_list",
        type=str,
        required=True,
        help="Path to token_list.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for config.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse training config through SOTASRTask's argument parser
    # to get the full namespace with all defaults filled in.
    task_parser = SOTASRTask.get_parser()
    task_args = task_parser.parse_args(
        ["--config", args.config, "--token_list", args.token_list]
    )

    # Convert namespace to dict for YAML serialization
    config_dict = vars(task_args)

    # Remove keys that are not serializable or not needed
    config_dict.pop("config", None)

    # Expand token_list from file path to actual list
    # (build_model_from_file expects the list inline, not a path)
    token_list_val = config_dict.get("token_list")
    if isinstance(token_list_val, str):
        with open(token_list_val, encoding="utf-8") as f:
            config_dict["token_list"] = [line.rstrip() for line in f]

    # Convert tuples to lists recursively so yaml.safe_load can parse it back
    def _tuples_to_lists(obj):
        if isinstance(obj, tuple):
            return [_tuples_to_lists(x) for x in obj]
        elif isinstance(obj, list):
            return [_tuples_to_lists(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _tuples_to_lists(v) for k, v in obj.items()}
        return obj

    config_dict = _tuples_to_lists(config_dict)

    # Write config.yaml
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    logging.info(f"Wrote config.yaml to {output_path}")
    logging.info(f"  token_list entries: {len(config_dict.get('token_list', []))}")
    logging.info(f"  encoder: {config_dict.get('encoder')}")
    logging.info(f"  decoder: {config_dict.get('decoder')}")
    logging.info(f"  model: {config_dict.get('model')}")


if __name__ == "__main__":
    main()
