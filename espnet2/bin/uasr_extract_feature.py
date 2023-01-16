#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from torch.nn.parallel import data_parallel
from typeguard import check_argument_types

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.uasr import UASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.forward_adaptor import ForwardAdaptor
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="UASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    parser.add_argument(
        "--uasr_train_config",
        type=str,
        help="uasr training configuration",
    )
    parser.add_argument(
        "--uasr_model_file",
        type=str,
        help="uasr model parameter file",
    )
    parser.add_argument(
        "--key_file",
        type=str_or_none,
        help="key file",
    )
    parser.add_argument(
        "--allow_variable_data_keys",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for feature extraction",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--dset",
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    return parser


def extract_feature(
    uasr_train_config: Optional[str],
    uasr_model_file: Optional[str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    batch_size: int,
    dtype: str,
    num_workers: int,
    allow_variable_data_keys: bool,
    ngpu: int,
    output_dir: str,
    dset: str,
    log_level: Union[int, str],
):
    assert check_argument_types()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    output_dir_path = Path(output_dir)

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    uasr_model, uasr_train_args = UASRTask.build_model_from_file(
        uasr_train_config, uasr_model_file, device
    )

    test_iter = UASRTask.build_streaming_iterator(
        data_path_and_name_and_type=data_path_and_name_and_type,
        key_file=key_file,
        batch_size=batch_size,
        dtype=dtype,
        num_workers=num_workers,
        preprocess_fn=UASRTask.build_preprocess_fn(uasr_train_args, False),
        collate_fn=UASRTask.build_collate_fn(uasr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    npy_scp_writers = {}
    for keys, batch in test_iter:
        batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

        if ngpu <= 1:
            data = uasr_model.collect_feats(**batch)
        else:
            data = data_parallel(
                ForwardAdaptor(uasr_model, "collect_feats"),
                (),
                range(ngpu),
                module_kwargs=batch,
            )

        for key, v in data.items():
            for i, (uttid, seq) in enumerate(zip(keys, v.cpu().detach().numpy())):
                if f"{key}_lengths" in data:
                    length = data[f"{key}_lengths"][i]
                    seq = seq[:length]
                else:
                    seq = seq[None]

                if (key, dset) not in npy_scp_writers:
                    p = output_dir_path / dset / "collect_feats"
                    npy_scp_writers[(key, dset)] = NpyScpWriter(
                        p / f"data_{key}", p / f"{key}.scp"
                    )
                npy_scp_writers[(key, dset)][uttid] = seq


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    extract_feature(**kwargs)


if __name__ == "__main__":
    main()
