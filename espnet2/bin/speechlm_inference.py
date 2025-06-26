#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from typeguard import typechecked

from espnet2.speechlm.espnet_model import ESPnetSpeechLMModel
from espnet2.speechlm.inference_utils import (
    ChatOrientedWriter,
    TaskOrientedWriter,
    build_inference_config,
    parse_sequence,
)
from espnet2.tasks.speechlm import SpeechLMTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args


class SpeechLM:
    """The Chat Interface of SpeechLM

    Args:
        train_config (str or Path): Path to the training configuration file
            from your training folder. Contains model architecture, tokenizer,
            and training hyperparameters.
        model_file (str or Path): Path to the trained model checkpoint file
            in your training folder containing the learned model weights.
        dtype (str): PyTorch data type for model inference. Must be one of:
            - "float16": Half precision (faster, less memory, potential accuracy loss)
            - "bfloat16": Half precision (faster, less memory, potential accuracy loss)
            - "float32": Single precision (default, good balance)
        device (str or torch.device): PyTorch device specification for model execution.
            Examples: "cpu", "cuda:0", "cuda:1", etc.
        inference_config (Dict[str, Any]): Dictionary mapping modality tokens to their
            respective inference configurations. Each modality (e.g., "speech", "text")
            has specific generation parameters like beam size, temperature, etc.
            See espnet2/speechlm/iference_utils.py for details.
        inference_mode (str): Inference operation mode. Must be one of:
            - "chat": Conversational mode with role tokens and turn-based interaction
            - "task": Task-oriented mode for structured input/output processing
        inference_last_segment (bool): Whether to perform inference only on the last
            segment of the input sequence. If False, processes all segments requiring
            generation.
        nbest (int): Number of hypothesis candidates to generate and return.
            Must be 1 for chat mode (batch inference not supported in chat).
            For task mode, can be > 1 to get multiple alternative outputs.
    """

    def __init__(
        self,
        # For model initialization
        train_config,
        model_file,
        dtype,
        device,
        # Inference parameters
        inference_config,
        inference_mode,
        inference_last_segment,
        nbest,
    ):
        # load model
        model, train_args = SpeechLMTask.build_model_from_file(
            train_config, model_file, device, dtype
        )

        # In case the model is trained with DPO
        if not isinstance(model, ESPnetSpeechLMModel):
            del model.reflm
            model.__class__ = ESPnetSpeechLMModel
            train_args.model = "espnet"

        self.model = model.corelm
        self.train_args = train_args

        # Inference configs
        self.inference_config = build_inference_config(
            train_args,
            inference_config,
            device,
            nbest,
        )
        self.inference_mode = inference_mode
        self.inference_last_segment = inference_last_segment
        self.nbest = nbest

        if self.inference_mode == "chat":
            assert self.nbest == 1, "Batch inference in chat mode is not supported."

    @torch.no_grad()
    def __call__(self, data):
        """Inference with the whole given sequence.

        This API is usually used for the given dataset, and working in segment-level
        teacher forcing.
        """
        dec_seq = data.get("dec_seq")

        # (1) Initialization
        self.model.decoders.reset()
        self.model.decoders.init()

        assert dec_seq.dim() == 3, dec_seq.size()
        assert dec_seq.size(0) == 1, dec_seq.size()
        if self.nbest > 1:
            dec_seq = dec_seq.expand(self.nbest, -1, -1)

        # (2) Parse the decoder sequence
        segments, is_prefills = parse_sequence(
            dec_seq,
            self.train_args.token_list,
            mode=self.inference_mode,
            inference_last_segment=self.inference_last_segment,
        )

        # (3) Inference on each segments
        prefill_buffer, all_segments = [], []
        for segment, is_prefill in zip(segments, is_prefills):
            # (3.1) cache the prefill
            if is_prefill:
                prefill_buffer.append(segment)
                all_segments.append(segment)
            else:
                # (3.2) add the known prefix of the target
                # if task mode, add the modality specifier
                # if chat mode, add the role token and modality specifier
                extra_prefill_len = 1 if self.inference_mode == "task" else 2
                prefill_buffer.append(segment[:, :extra_prefill_len, :])
                prefill = torch.cat(prefill_buffer, dim=1)
                prefill_buffer = []

                # (3.3) find the corresponding inference config
                modality_token = segment[0, extra_prefill_len - 1, 0].item()
                modality_token = self.train_args.token_list[modality_token]
                modality_token = modality_token.removeprefix("<").removesuffix(
                    "_start/end>"
                )
                inference_config = self.inference_config[modality_token]

                # (3.4) inference one segment
                inferred_segments = self.inference_one_segment(
                    prefill,
                    segment[:, extra_prefill_len:, :],
                    inference_config,
                )
                prefix = segment[0, :extra_prefill_len, :]
                inferred_segments_new = []
                for candidate in inferred_segments:
                    candidate = torch.cat([prefix, candidate], dim=0)
                    inferred_segments_new.append(candidate)

                all_segments.append(inferred_segments_new)

        return all_segments

    def inference_one_segment(self, prefill, reference, inference_config):
        """Inference one turn with the prefill until certain requirements are met.

        This API is used in both __call__ function and the user interactive interface.
        """
        inferred_segment, _ = self.model.inference(prefill, reference, inference_config)
        return inferred_segment


def get_parser():
    parser = argparse.ArgumentParser(
        description="SpeechLM inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of concurrent processes",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    # Model
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )

    # Inference configs
    parser.add_argument(
        "--inference_config_file",
        type=str,
        help="Inference configuration file",
    )
    parser.add_argument(
        "--inference_last_segment",
        type=str2bool,
        default=False,
        help="Inference configuration file",
    )
    parser.add_argument(
        "--nbest",
        type=int,
        help="Number of best hypotheses to return",
    )

    # Data
    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str,
        help="Data path and name and type",
    )

    return parser


@typechecked
def inference(
    # General
    rank: int,
    nproc: int,
    output_dir: Path,
    batch_size: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: str,
    # Model
    train_config: str,
    model_file: str,
    dtype: str,
    # Inference_configs
    inference_config: Dict,
    inference_last_segment: bool,
    nbest: int,
    # Data:
    data_path_and_name_and_type: str,
):
    # (1) General settings
    if batch_size > 1:
        raise ValueError("Batch size > 1 is not supported yet.")
    if ngpu > 1:
        raise ValueError("Multi-GPU is not supported yet.")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{rank}/{nproc}] "
        f"%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if torch.cuda.is_available() and ngpu > 0:
        device = "cuda:0"
    else:
        device = "cpu"

    data_path = data_path_and_name_and_type.split(",")[0]
    task_name = json.load(open(data_path))["task"]
    if "dialogue" in task_name:
        inference_mode = "chat"
    else:
        inference_mode = "task"

    # (2) Load model
    speechlm = SpeechLM(
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        device=device,
        inference_config=inference_config,
        inference_mode=inference_mode,
        inference_last_segment=inference_last_segment,
        nbest=nbest,
    )

    # (3) Load data
    loader = SpeechLMTask.build_streaming_iterator(
        [data_path_and_name_and_type.strip().split(",")],
        dtype=dtype,
        batch_size=batch_size,
        num_workers=num_workers,
        preprocess_fn=SpeechLMTask.build_preprocess_fn(speechlm.train_args, False),
        collate_fn=SpeechLMTask.build_collate_fn(speechlm.train_args, False),
        allow_variable_data_keys=False,
        inference=True,
        multi_task_dataset=True,
    )

    # (4) Build result writer
    if inference_mode == "task":
        writer = TaskOrientedWriter(
            train_args=speechlm.train_args,
            task=task_name,
            output_dir=output_dir,
            rank=rank,
            inference_config=speechlm.inference_config,
        )
    elif inference_mode == "chat":
        writer = ChatOrientedWriter(
            train_args=speechlm.train_args,
            task=task_name,
            output_dir=output_dir,
            rank=rank,
            inference_config=speechlm.inference_config,
        )
    # (5) Inference loop
    for iiter, (keys, batch) in enumerate(loader, 1):
        if iiter % nproc != rank:
            continue

        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert _bs == 1, _bs

        batch = to_device(batch, device=device)
        key = keys[0]

        # NOTE(Jinchuan): make each example independently rseproducible
        set_all_random_seed(seed)

        # model infernece
        logging.info(f"Inference on example: {key}")
        all_segments = speechlm(batch)

        # save results
        writer.write(key, all_segments)


def main(cmd=None):
    """Run SpeechLM model inference."""
    # (1) record the variables
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)

    inference_config = kwargs.pop("inference_config_file")
    inference_config = yaml.safe_load(open(inference_config, "r"))
    keys = list(inference_config.keys())
    for key in keys:
        if key in kwargs:
            kwargs[key] = inference_config.pop(key)
    kwargs["inference_config"] = inference_config
    print(f"kwargs: {kwargs}")

    # (2) multiprocessing inference
    nproc = kwargs["nproc"]
    mp = torch.multiprocessing.get_context("spawn")
    processes = list()
    for rank in range(nproc):
        kwargs["rank"] = rank
        p = mp.Process(
            target=inference,
            kwargs=kwargs,
        )
        p.start()
        processes.append(p)

    try:
        # Polling loop:
        # We repeatedly check if any process has exited and if so, whether it failed.
        # If a process fails, we terminate all remaining.
        while True:
            all_done = True

            for p in processes:
                # If the process is still alive, try joining for a short time (poll)
                if p.is_alive():
                    p.join(timeout=1)  # Non-blocking "poll" wait
                    # After the join with timeout, if p is still alive,
                    # we'll continue the loop. If it ended, we can check exitcode.

                # Now, if the process is NOT alive, it might have finished or failed
                if not p.is_alive():
                    # p.exitcode == 0 means success
                    # p.exitcode != 0 means error/exception
                    if p.exitcode is not None and p.exitcode != 0:
                        raise ChildProcessError(
                            f"Process {p.pid} terminated with exitcode={p.exitcode}"
                        )
                else:
                    # If it's still alive at this point, we're not all done yet
                    all_done = False

            if all_done:
                # Means all processes have finished successfully
                print("All processes finished successfully.")
                break

    except Exception as e:
        print(f"Error detected: {e}. Terminating all processes...", file=sys.stderr)

        # Terminate all running processes
        for p in processes:
            if p.is_alive():
                p.terminate()

        # Re-raise the original exception so it can be caught or exit accordingly
        raise

    # (3) finally, merge results from all processes
    file_dict = dict()
    for file in args.output_dir.rglob("rank*"):
        file_name = file.name
        match = re.match(r"rank(\d+)_(.+)", file_name)
        if match:
            rank = int(match.group(1))  # Extract rank as an integer
            file_name = match.group(2)

            if int(rank) >= nproc:
                continue
            if file_name.endswith(".ark"):
                continue

        else:
            continue

        merge_file_path = file.parent / file_name
        if merge_file_path not in file_dict:
            file_dict[merge_file_path] = list()
        file_dict[merge_file_path].append(file)

    for name, files in file_dict.items():
        writer = open(name, "w")
        for file in files:
            for line in open(file):
                writer.write(line)


if __name__ == "__main__":
    main()
