#!/usr/bin/env python3
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import numpy as np
import torch
import yaml
from torch.nn.parallel import data_parallel
from torch.utils.data.dataloader import DataLoader
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.lm import LMTask
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.dataset import ESPnetDataset
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils.fileio import DatadirWriter
from espnet2.torch_utils.forward_adaptor import ForwardAdaptor
from espnet2.utils.types import float_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def calc_perplexity(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    log_base: Optional[float],
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # 2. Build LM
    with Path(train_config).open("r") as f:
        train_args = yaml.safe_load(f)
    train_args = argparse.Namespace(**train_args)
    model = LMTask.build_model(train_args)
    model.load_state_dict(torch.load(model_file, map_location=device))
    # Wrape model to make model.nll() data-parallel
    wrapped_model = ForwardAdaptor(model, "nll")
    wrapped_model.to(device=device, dtype=getattr(torch, dtype)).eval()

    # 3. Build data-iterator
    dataset = ESPnetDataset(
        data_path_and_name_and_type,
        float_dtype=dtype,
        preprocess=LMTask.build_preprocess_fn(train_args, False),
    )
    LMTask.check_task_requirements(dataset, allow_variable_data_keys, False)
    if key_file is None:
        key_file, _, _ = data_path_and_name_and_type[0]

    batch_sampler = ConstantBatchSampler(batch_size=batch_size, key_file=key_file,)

    logging.info(f"Model:\n{model}")
    logging.info(f"Batch sampler: {batch_sampler}")
    logging.info(f"dataset:\n{dataset}")
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=LMTask.build_collate_fn(train_args),
        num_workers=num_workers,
    )

    # 4. Start for-loop
    with DatadirWriter(output_dir) as writer:
        total_nll = 0.0
        total_ntokens = 0
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            with torch.no_grad():
                batch = to_device(batch, device)
                if ngpu <= 1:
                    # NOTE(kamo): data_parallel also should work with ngpu=1,
                    # but for debuggability it's better to keep this block.
                    nll, lengths = wrapped_model(**batch)
                else:
                    nll, lengths = data_parallel(
                        wrapped_model, (), range(ngpu), module_kwargs=batch
                    )

            assert _bs == len(nll) == len(lengths), (_bs, len(nll), len(lengths))
            # nll: (B, L) -> (B,)
            nll = nll.detach().cpu().numpy().sum(1)
            # lengths: (B,)
            lengths = lengths.detach().cpu().numpy()
            total_nll += nll.sum()
            total_ntokens += lengths.sum()

            for key, _nll, ntoken in zip(keys, nll, lengths):
                if log_base is None:
                    utt_ppl = np.exp(_nll / ntoken)
                else:
                    utt_ppl = log_base ** (_nll / ntoken / np.log(log_base))

                # Write PPL of each utts for debugging or analysis
                writer["utt2ppl"][key] = str(utt_ppl)
                writer["utt2ntokens"][key] = str(ntoken)

        if log_base is None:
            ppl = np.exp(total_nll / total_ntokens)
        else:
            ppl = log_base ** (total_nll / total_ntokens / np.log(log_base))

        with (Path(output_dir) / "ppl").open("w") as f:
            f.write(f"{ppl}\n")
        with (Path(output_dir) / "base").open("w") as f:
            if log_base is None:
                _log_base = np.e
            else:
                _log_base = log_base
            f.write(f"{_log_base}\n")
        logging.info(f"PPL={ppl}")


def get_parser():
    parser = configargparse.ArgumentParser(
        description="Calc perplexity",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )
    parser.add_argument(
        "--log_base",
        type=float_or_none,
        default=None,
        help="The base of logarithm for Perplexity. "
        "If None, napier's constant is used.",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    calc_perplexity(**kwargs)


if __name__ == "__main__":
    main()
