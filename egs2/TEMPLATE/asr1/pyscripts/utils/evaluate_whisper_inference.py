#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
import whisper
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class Speech2Text:
    """Speech2Text class"""

    def __init__(
        self,
        model_tag: str = "base",
        device: str = "cpu",
    ):
        assert check_argument_types()

        self.model = whisper.load_model(model_tag).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, speech: str) -> Optional[str]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text

        """
        assert check_argument_types()

        # Input as audio signal
        result = self.model.transcribe(speech)

        return result["text"]


def inference(
    output_dir: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: str,
    key_file: Optional[str],
    model_tag: Optional[str],
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text = Speech2Text(
        model_tag=model_tag,
        device=device,
    )

    # 3. Build data-iterator
    info_list = []
    wavscp = open(data_path_and_name_and_type, "r", encoding="utf-8")
    for line in wavscp.readlines():
        info_list.append(line.split(maxsplit=1))

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for key, audio_file in info_list:
            # N-best list of (text, token, token_int, hyp_object)
            results = speech2text(os.path.abspath(audio_file.strip()))

            # Normal ASR
            ibest_writer = writer[f"1best_recog"]

            # Write the result to each file
            ibest_writer["text"][key] = results


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str,
        required=True,
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
