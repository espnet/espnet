#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

from espnet2.speechlm.definitions import MODALITIES, SPEECHLM_TASKS
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump
from espnet2.speechlm.definitions import MODALITIES, SPEECHLM_TASKS

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extend the vocabulary and corresponding pre-trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_token_list_dir",
        type=Path,
        required=True,
        help="The input token list directory",
    )
    parser.add_argument(
        "--output_token_list_dir",
        type=Path,
        required=True,
        help="The output token list directory",
    )
    parser.add_argument(
        "--input_exp_dir",
        type=Path,
        required=True,
        help="The input experiment directory",
    )
    parser.add_argument(
        "--output_exp_dir",
        type=Path,
        required=True,
        help="The output experiment directory",
    )
    parser.add_argument(
        "--inference_model",
        type=str,
        required=True,
        help="The model name used for inference",
    )
    parser.add_argument(
        "--additional_vocabs",
        type=Path,
        nargs="+",
        required=True,
        help="The additional vocabulary files",
    )
    parser.add_argument(
        "--additional_tasks",
        type=str,
        nargs="+",
        default=[],
        help="Additional tasks that need task ids",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # (1) load and check files
    token_list = args.input_token_list_dir / "token_list.json"
    token_bias = args.input_token_list_dir / "token_bias.json"

    if not token_list.exists():
        raise ValueError(f"File {str(token_list)} doesn't exit")
    token_list = json.load(open(token_list))

    if not token_bias.exists():
        raise ValueError(f"File {str(token_bias)} doesn't exit")
    token_bias = json.load(open(token_bias))

    config = args.input_exp_dir / "config.yaml"
    if not config.exists():
        raise ValueError(f"File {str(config)} doesn't exit")
    config = yaml.safe_load(open(config))

    checkpoint = args.input_exp_dir / args.inference_model
    if not checkpoint.exists():
        raise ValueError(f"File {str(checkpoint)} doesn't exit")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    additional_vocabs = []
    for vocab_file in args.additional_vocabs:
        if not vocab_file.exists():
            raise ValueError(f"File {str(vocab_file)} doesn't exit")
        if not vocab_file.stem.endswith("_token_list"):
            raise ValueError(
                f"Additional vocab file should have name <modality_name>_token_list "
                f"or <modality_name>_token_list.json"
            )
        modality_name = vocab_file.stem.removesuffix("_token_list")
        if vocab_file.suffix == ".json":
            vocab = json.load(open(vocab_file))
        else:
            vocab = [w.strip() for w in open(vocab_file)]
        additional_vocabs.append((modality_name, vocab))

    # (1) extend token list
    # check compatibility
    for idx, (tok, tok_) in enumerate(zip(token_list, config["token_list"])):
        if tok != tok_:
            raise ValueError(
                f"original token lists are incompatible. "
                f"{idx}-th token: {tok} vs. {tok_} "
            )

    # add new tokens
    token_list_dict = {tok: idx for idx, tok in enumerate(token_list)}
    for modality_name, vocab in additional_vocabs:

        if modality_name in token_bias:
            raise ValueError(
                f"Modality {modality_name} is already in current vocabulary"
            )

        token_bias[modality_name] = len(token_list_dict)
        if modality_name not in MODALITIES:
            raise ValueError(
                f"The modality {modality_name} is not supported "
                f"Revise espnet2.speechlm.definitions if this is a new modality"
            )
        if f"<{modality_name}_start/end>" not in token_list_dict:
            for idx in range(32, 64):
                if f"<unused_token_{idx}>" in token_list_dict:
                    token_list_dict[f"<{modality_name}_start/end>"] = token_list_dict[
                        f"<unused_token_{idx}>"
                    ]
                    del token_list_dict[f"<unused_token_{idx}>"]
                    logging.info(
                        f"replace <unused_token_{idx}> by <{modality_name}_start/end>"
                    )
                    break

        for tok in vocab:
            if tok in token_list_dict:
                logging.warning(
                    f"Modality: {modality_name}, token: {tok} already exists "
                    f"make it as ext_{tok}. Please ensure the duplication is expected"
                )
                tok = f"ext_{tok}"
                assert tok not in token_list_dict
            token_list_dict[tok] = len(token_list_dict)

        logging.info(
            f"Add new modality: {modality_name} "
            f"from {token_bias[modality_name]} to {len(token_list_dict)}"
        )

    for task_name in args.additional_tasks:
        if task_name not in SPEECHLM_TASKS:
            raise ValueError(
                f"SpeechLM task {task_name} is not supported "
                f"Revise espnet2.speechlm.definitions if this is a new task "
            )
        if f"<{task_name}_task>" not in token_list_dict:
            for idx in range(64, 128):
                if f"<unused_token_{idx}>" in token_list_dict:
                    token_list_dict[f"<{task_name}_task>"] = token_list_dict[
                        f"<unused_token_{idx}>"
                    ]
                    del token_list_dict[f"<unused_token_{idx}>"]
                    logging.info(f"replace <unused_token_{idx}> by <{task_name}_task>")
                    break

    # save the new token list and token bias
    token_list = list(token_list_dict.keys())
    token_list.sort(key=lambda x: token_list_dict[x])
    config["token_list"] = token_list
    config["token_bias"] = token_bias

    args.output_token_list_dir.mkdir(parents=True, exist_ok=True)
    vocab_writer = open(args.output_token_list_dir / "token_list.json", "w")
    vocab_writer.write(json.dumps(token_list, indent=4))

    bias_writer = open(args.output_token_list_dir / "token_bias.json", "wb")
    bias_writer.write(
        json.dumps(token_bias, indent=4, ensure_ascii=False, sort_keys=False).encode(
            "utf_8"
        )
    )

    args.output_exp_dir.mkdir(parents=True, exist_ok=True)
    config_writer = open(args.output_exp_dir / "config.yaml", "w")
    yaml_no_alias_safe_dump(config, config_writer, indent=4, sort_keys=False)

    # (2) revise the embedding and lm_head
    num_new_tokens = sum([len(vocab) for modality_name, vocab in additional_vocabs])
    for tensor_name in [
        "corelm.emb.weight",
        "corelm.lm_head.weight",
        "corelm.criterion.lm_head.weight",
    ]:
        if tensor_name not in checkpoint:
            raise ValueError(
                f"Cannot find {tensor_name} in original checkpoint "
                f"Please revise this script if the embedding names have been changed"
            )
        old_tensor = checkpoint[tensor_name]
        new_tensor = torch.randn(
            (num_new_tokens, old_tensor.size(1)),
            device=old_tensor.device,
            dtype=old_tensor.dtype,
        )

        # force the deterministic behavior
        torch.random.manual_seed(0)
        std = torch.var(old_tensor, dim=None)
        torch.nn.init.normal_(new_tensor, mean=0, std=std)

        new_tensor = torch.cat([old_tensor, new_tensor], dim=0).contiguous()
        print("new tensor name and shape: ", tensor_name, new_tensor.size())
        checkpoint[tensor_name] = new_tensor

    torch.save(checkpoint, args.output_exp_dir / args.inference_model)


if __name__ == "__main__":
    main()
