#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import whisper
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class Speech2Text:
    """Speech2Text class"""

    @typechecked
    def __init__(
        self,
        model_tag: str = "base",
        model_dir: str = "./models",
        device: str = "cpu",
    ):

        self.model = whisper.load_model(
            name=model_tag, download_root=model_dir, device=device
        )

    @torch.no_grad()
    @typechecked
    def __call__(self, speech: str, **decode_options) -> Optional[str]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text

        """

        # Input as audio signal
        result = self.model.transcribe(speech, **decode_options)

        return result["text"]


@typechecked
def inference(
    rank: int,
    nproc: int,
    output_dir: Path,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: str,
    key_file: Optional[str],
    model_tag: Optional[str],
    model_dir: Optional[str],
    allow_variable_data_keys: bool,
    decode_options: Dict,
):
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
    level=logging.INFO,
        format=f"[{rank}/{nproc}] "
               f"%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if torch.cuda.is_available() and ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text = Speech2Text(
        model_tag=model_tag,
        model_dir=model_dir,
        device=device,
    )

    # 3. Build data-iterator
    info_list = []
    wavscp = open(key_file, "r", encoding="utf-8")
    for line in wavscp.readlines():
        info_list.append(line.split(maxsplit=1))

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for iiter, (key, audio_file) in enumerate(info_list):
            if iiter % nproc != rank:
                continue

            # N-best list of (text, token, token_int, hyp_object)
            logging.info(
                f"decoding {key} ...",
            )
            results = speech2text(os.path.abspath(audio_file.strip()), **decode_options)

            # Normal ASR
            ibest_writer = writer[f"1best_recog"]

            # Write the result to each file
            ibest_writer[f"text_rank{rank}"][key] = results


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes on a single GPU",
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="global work rank",
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
        default="base",
        choices=[
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large",
            "large-v3",
            "large-v3-turbo",
        ],
        help="Model tag of the released whisper models.",
    )
    group.add_argument(
        "--model_dir",
        type=str_or_none,
        default="./models",
        help="The directory to download whisper models.",
    )

    group = parser.add_argument_group("Decoding options related")
    group.add_argument(
        "--decode_options",
        action=NestedDictAction,
        default=dict(),
        help="Decode options for whisper transcribe.",
    )
    return parser


def main(cmd=None):
    # (1) record the variables
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    print("kwargs: ", dict(kwargs))

    # (2) multiprocessing inference
    nproc = kwargs["nproc"]
    mp = torch.multiprocessing.get_context("spawn")
    processes = list()
    for rank in range(nproc):
        kwargs['rank'] = rank
        p = mp.Process(
            target=inference,
            kwargs=kwargs,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # (3) finally, merge results from all processes
    file_dict = dict()
    for file in args.output_dir.rglob('*_rank*'):
        name = file.parent / file.name.split('_rank')[0]
        if name not in file_dict:
            file_dict[name] = list()
        file_dict[name].append(file)
    
    for name, files in file_dict.items():
        writer = open(name, 'w')
        for file in files:
            for line in open(file):
                writer.write(line)


if __name__ == "__main__":
    main()
