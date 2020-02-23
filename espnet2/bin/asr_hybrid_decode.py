#!/usr/bin/env python3
import logging
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import kaldiio
import numpy as np
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr_hybrid import ASRHybridTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def inference(
    wspecifier: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    pdfs: str,
    floor: float,
    prior_weight: float,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: str,
    model_file: str,
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
    set_all_random_seed(seed)

    # 2. Build ASR model
    asr_model, asr_train_args = ASRHybridTask.build_model_from_file(
        train_config, model_file, device
    )

    # Count the number of occurrences of each classes and deal the occurrence
    # probability as the "prior"
    prior = np.zeros(asr_model.num_targets)
    with open(pdfs) as f:
        for line in f:
            pdf = line.strip().split()[1:]
            pdf = list(map(int, pdf))
            prior += np.bincount(pdf, minlength=asr_model.num_targets)
    prior /= prior.sum()
    log_prior = np.log(np.maximum(prior, floor))

    # 3. Build data-iterator
    loader, _, _ = ASRHybridTask.build_non_sorted_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRHybridTask.build_preprocess_fn(asr_train_args, False),
        collate_fn=ASRHybridTask.build_collate_fn(asr_train_args),
        allow_variable_data_keys=allow_variable_data_keys,
    )

    # 4 .Start for-loop
    with kaldiio.WriteHelper(wspecifier) as f:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            with torch.no_grad():
                batch = to_device(batch, device)
                # Not applying log-softmax here, just forwarding nnet.
                posterior, lens = asr_model.nnet_forward(**batch)
            posterior = posterior.detach().cpu().numpy()
            for key, x, l in zip(keys, posterior, lens):
                # prior: p(s|x)
                # posterior: p(s|x)
                # -> likelihood: p(x|s) ~ p(s|x) - p(x) in log-domain
                #    (ignoring constant values because it doesn't affect decoding)
                f[key] = x[:l] - prior_weight * log_prior


def get_parser():
    parser = configargparse.ArgumentParser(
        description="ASR hybrid Decoding",
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
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )
    parser.add_argument("--pdfs", type=str, required=True)
    parser.add_argument("--floor", type=float, default=1e-20, help="Avoid log(0)")
    parser.add_argument(
        "--prior_weight", type=float, default=1.0, help="To weaken the effect of prior",
    )

    parser.add_argument(
        "--wspecifier",
        type=str,
        required=True,
        help="Same meaning as Kaldi's wspecifier. e.g. ark:-, ark:out.ark,scp:out.scp",
    )
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
    group.add_argument("--train_config", type=str, required=True)
    group.add_argument("--model_file", type=str, required=True)

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
