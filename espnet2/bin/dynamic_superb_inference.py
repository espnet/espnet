#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.dynamic_superb import DynamicSuperbTask
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
    **kwargs,
):
    """Perform Qwen2-Audio inference using ESPnet2 framework[36]"""
    print(type(data_path_and_name_and_type))
    print(len(data_path_and_name_and_type))
    print(data_path_and_name_and_type[0])
    
    # Initialize logging first
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    
    device = "cuda" if ngpu >= 1 else "cpu"
    set_all_random_seed(seed)

    # Build model using ESPnet2 task system[8]
    # model, train_args = DynamicSuperbTask.build_model_from_file(
    #     config_file=train_config, 
    #     model_file=model_file, 
    #     device=device
    # )
    args = argparse.Namespace()
    args.model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    model = DynamicSuperbTask.build_model(args)
    model.to(dtype=getattr(torch, dtype)).eval()
    train_args = None

    # Build data iterator[8]
    loader = DynamicSuperbTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=DynamicSuperbTask.build_preprocess_fn(train_args, False),
        collate_fn=DynamicSuperbTask.build_collate_fn(train_args, False),
        allow_variable_data_keys=True,
        inference=True,
    )

    # Start inference
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            batch = to_device(batch, device)

            for i, key in enumerate(keys):
                speech = batch["speech"][i : i + 1]
                instruction = batch.get("text", ["What does the person say?"])[0]
                
                if isinstance(instruction, torch.Tensor):
                    instruction = "What does the person say?"
                
                # Use the integrated Qwen2-Audio model
                prediction = model.inference(speech[0], instruction)
                
                writer["text"][key] = prediction

def get_parser():
    """Build argument parser[8]"""
    parser = config_argparse.ArgumentParser(
        description="Qwen2-Audio inference using ESPnet2 framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Add data arguments
    parser.add_argument("--data_path_and_name_and_type", type=str2triple_str, required=True, action="append")
    parser.add_argument("--key_file", type=str)
    parser.add_argument("--train_config", type=str)
    parser.add_argument("--model_file", type=str)
    
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
