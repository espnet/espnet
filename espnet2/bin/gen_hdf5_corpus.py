#!/usr/bin/env python3
import argparse
import collections
from io import StringIO
import logging
from pathlib import Path

import h5py
import numpy as np

from espnet.utils.cli_utils import get_commandline_args
from espnet2.train.dataset import DATA_TYPES
from espnet2.train.dataset import ESPnetDataset
from espnet2.utils.types import str2triple_str


def get_parser():
    parser = argparse.ArgumentParser(
        description="Launch distributed process with appropriate options. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        action="append",
        default=[],
    )
    parser.add_argument("--shape_file", type=str, action="append", default=[])
    parser.add_argument("--out", type=str, help="Output HDF5 file name")
    return parser


def read_scp(data_path):
    with open(data_path) as f:
        for linenum, line in enumerate(f):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) != 2:
                raise RuntimeError(
                    f"must have two or more columns: " f"{line}({data_path}:{linenum})"
                )
            k, v = sps
            yield linenum, k, v


def main(cmd=None):
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = get_parser()
    args = parser.parse_args(cmd)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    """Generate 'HDF5 corpus'

    ESPnet2 supports two types of methods for data inputting:
      1. Separated files like feats.scp, text, etc.
      2. A HDF5 file created by combining them

    The HDF5 must have the following structure e.g.:
      - speech/type="sound"
      - speech/data
          - id1="/some/where/a.wav"
          - id2="/some/where/b.wav"
          - ...
      - text/type="text"
      - text/data
          - id1="abc def"
          - id2="hello world"
          - ...
      - shape_files/0
          - id1=(10000,)
          - id2=(14000,)
          - ...
      - shape_files/1
          - id1=(2,)
          - id2=(2,)
          - ...
    """
    with h5py.File(args.out, "w") as fout:
        for data_path, name, type in args.data_path_and_name_and_type:
            if type not in DATA_TYPES:
                raise RuntimeError(f"Must be one of {list(DATA_TYPES)}: {type}")

            # If scp file, insert the reference file path instead of ndarray
            # e.g. uttid_A /some/where/a.wav
            # => f["name/data/uttid_A"] = "/some/where/a.wav"
            if type in ("sound", "npy", "kaldi_ark", "pipe_wav"):
                fout[f"{name}/type"] = type
                for linenum, k, v in read_scp(data_path):
                    fout[f"{name}/data/{k}"] = v

            # The other case, set ndarray/str directly
            else:
                fout[f"{name}/type"] = "direct"
                loader = ESPnetDataset.build_loader(data_path, type)
                if isinstance(loader, collections.abc.Mapping):
                    for k in loader:
                        fout[f"{name}/data/{k}"] = loader[k]
                elif isinstance(loader, collections.abc.Collection):
                    for idx, v in enumerate(loader):
                        fout[f"{name}/data/{idx}"] = v
                else:
                    raise RuntimeError(f"{type} is not supported")

        for idx, shape_file in enumerate(args.shape_file):
            for linenum, k, v in read_scp(shape_file):
                v = np.loadtxt(StringIO(v), ndmin=1, dtype=np.long, delimiter=",")
                fout[f"shape_files/{idx}/{k}"] = v

        # Check having same keys set
        first_group = fout[args.data_path_and_name_and_type[0][1]]["data"]
        for data_path, name, type in args.data_path_and_name_and_type:
            if set(first_group) != set(fout[name]["data"]):
                raise RuntimeError(
                    f"Keys are mismatched between "
                    f"{args.data_path_and_name_and_type[0][0]} and {data_path}"
                )

        for idx, shape_file in enumerate(args.shape_file):
            if set(first_group) != set(fout["shape_files"][str(idx)]):
                raise RuntimeError(
                    f"Keys are mismatched between "
                    f"{args.data_path_and_name_and_type[0][0]} and {shape_file}"
                )


if __name__ == "__main__":
    main()
