#!/usr/bin/env python3
import argparse
import logging
from typing import Optional, Sequence, Tuple, Union

from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.ps2st import PS2STTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2triple_str


@typechecked
def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    model_file: Optional[str],
    train_config: Optional[str],
    decode_config_path: Optional[str],
    **kwargs,
):
    """Perform Qwen2-Audio inference using ESPnet2 framework"""

    # Initialize logging first
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    device = "cuda" if ngpu >= 1 else "cpu"
    set_all_random_seed(seed)

    args = argparse.Namespace()
    # Currently only Qwen2-Audio is available.
    args.model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    args.decode_config_path = decode_config_path
    model = PS2STTask.build_model(args)
    model.to(device).eval()
    train_args = None

    # Build data iterator
    loader = PS2STTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=PS2STTask.build_preprocess_fn(train_args, False),
        collate_fn=PS2STTask.build_collate_fn(train_args, False),
        allow_variable_data_keys=True,
        inference=True,
    )

    # Start inference
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            batch = to_device(batch, device)
            text_output = model.inference(**batch)

            writer["text"][keys[0]] = text_output


def get_parser():
    """Build argument parser"""
    parser = config_argparse.ArgumentParser(
        description="Qwen2-Audio inference using ESPnet2 framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", default="float32", choices=["float16", "float32", "float64"]
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)

    # Add data arguments
    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    parser.add_argument("--key_file", type=str)
    parser.add_argument("--train_config", type=str)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--decode_config_path", type=str, default=None)

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    return parser


def main(cmd=None):
    """Main function"""
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    inference(**kwargs)


if __name__ == "__main__":
    main()
