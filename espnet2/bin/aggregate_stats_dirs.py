import argparse
import logging
import sys
from pathlib import Path
from typing import Union
from typing import Iterable

import numpy as np

from espnet.utils.cli_utils import get_commandline_args


def aggregate_stats_dirs(
    input_dir: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    log_level: str,
):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) "
               "%(levelname)s: %(message)s",
    )

    input_dirs = [Path(p) for p in input_dir]
    output_dir = Path(output_dir)

    for mode in ["train", "eval"]:
        with (input_dirs[0] / mode / "shape_keys").open("r") as f:
            shape_keys = [l.strip() for l in f if l.strip() != ""]
        with (input_dirs[0] / mode / "stats_keys").open("r") as f:
            stats_keys = [l.strip() for l in f if l.strip() != ""]
        (output_dir / mode).mkdir(parents=True, exist_ok=True)

        for key in shape_keys:
            with (output_dir / mode / f"{key}_shape").open("w") as fout:
                for idir in input_dirs:
                    with (idir / mode / f"{key}_shape").open("r") as fin:
                        for line in fin:
                            fout.write(line)

        for key in stats_keys:
            sum_stats = None
            for idir in input_dirs:
                stats = np.load(idir / mode / f"{key}_stats.npz")
                if sum_stats is None:
                    sum_stats = dict(**stats)
                else:
                    for k in stats:
                        sum_stats[k] += stats[k]

            np.savez(output_dir / mode / f"{key}_stats.npz", **sum_stats)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate statistics directories into one directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--input_dir", action="append",
                        help="Input directories")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    aggregate_stats_dirs(**kwargs)


if __name__ == "__main__":
    main()
