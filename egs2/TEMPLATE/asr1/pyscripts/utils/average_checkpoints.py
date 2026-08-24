#!/usr/bin/env python3

# Copyright 2025 Yifan Peng (CMU)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Average checkpoints.

Example:
    python3 average_checkpoints.py \
        --inputs model1.pth model2.pth model3.pth \
        --output averaged.pth
"""

import logging
from argparse import ArgumentParser
import pickle

import torch


def _is_weights_only_compat_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "weights only load failed" in message
        or "weights_only" in message
        or "unsupported global" in message
    )


def get_parser():
    parser = ArgumentParser(description="Average checkpoints.")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        help="Input checkpoint files to be averaged.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output averaged checkpoint file.",
    )
    return parser


def main(args):
    avg_state_dict = None
    for ckpt_path in args.inputs:
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError, TypeError) as e:
            if not isinstance(e, pickle.UnpicklingError) and not _is_weights_only_compat_error(e):
                raise
            logging.warning(
                "Loading %s with weights_only=False for compatibility: %s",
                ckpt_path,
                e,
            )
            state_dict = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )

        if avg_state_dict is None:
            avg_state_dict = state_dict
        else:
            for k in avg_state_dict:
                avg_state_dict[k] = avg_state_dict[k] + state_dict[k]

    for k in avg_state_dict:
        if str(avg_state_dict[k].dtype).startswith("torch.int"):
            # For int type, not averaged, but only accumulated.
            # e.g. BatchNorm.num_batches_tracked
            # (If there are any cases that requires averaging
            #  or the other reducing method, e.g. max/min, for integer type,
            #  please report.)
            # See also: espnet2/main_funcs/average_nbest_models.py
            pass
        else:
            avg_state_dict[k] = avg_state_dict[k] / len(args.inputs)

    torch.save(avg_state_dict, args.output)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
