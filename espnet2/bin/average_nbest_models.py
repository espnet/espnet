import argparse
import logging
from pathlib import Path
import sys
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.train.reporter import Reporter
from espnet2.utils.types import str2triple_str


@torch.no_grad()
def average_nbest_models(
    output_dir: Path,
    criterion: Tuple[str, str, str],
    nbest: int,
    log_level: Union[int, str],
) -> None:
    assert check_argument_types()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) (levelname)s: %(message)s",
    )

    checkpoint = output_dir / "checkpoint.pth"
    if not checkpoint.exists():
        raise RuntimeError(f"{checkpoint} is not found.")

    reporter = Reporter()
    states = torch.load(checkpoint, map_location="cpu")
    reporter.load_state_dict(states["reporter"])

    # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
    ph, cr, mode = criterion
    epoch_and_values = reporter.sort_epochs_and_values(ph, cr, mode)[:nbest]
    op = output_dir / f"{ph}.{cr}.ave_{len(epoch_and_values)}best.pth"
    logging.info(
        f"Averaging {len(epoch_and_values)}best models: " f'criterion="{ph}.{cr}": {op}'
    )

    # 2. Averaging model
    avg = None
    for e, _ in epoch_and_values:
        states = torch.load(output_dir / f"{e}epoch.pth", map_location="cpu")
        if avg is None:
            avg = states
        else:
            # Accumulated
            for k in avg:
                avg[k] += states[k]
    for k in avg:
        avg[k] /= len(epoch_and_values)

    # 3. Save the ave model and create a symlink
    torch.save(avg, op)
    sym_op = output_dir / f"{ph}.{cr}.ave.pth"
    if sym_op.is_symlink() or sym_op.exists():
        sym_op.unlink()
    sym_op.symlink_to(op.name)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average n-best models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    parser.add_argument(
        "--nbest", type=int, default=100000, help="The number of models to be averaged",
    )
    parser.add_argument(
        "--criterion",
        type=str2triple_str,
        help="The evaluation criterion",
        default=("valid", "acc", "max"),
    )

    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    average_nbest_models(**kwargs)


if __name__ == "__main__":
    main()
