#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import shutil
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.tasks.speechlm import SpeechLMTask
from espnet2.speechlm.inference.inference import (
    SpeechLMInference,
    SpeechLMHypothesis,
)
from espnet2.speechlm.definitions import tasks as speechlm_tasks
from espnet2.speechlm.definitions import modalities as speechlm_modalities
from espnet2.tasks.speechlm import post_processor_choices

from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args

class SpeechLM:
    """ SpeechLM class.
    
    Examples: (TODO)
    """

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        # Inference parameters:
        search_type: str = "ar",
        search_algo: str = "sampling",
        nbest: int = 1,
        sampling_temperature: float = 1.0,
        top_k: int = 20,
        maxlenratio: float = 0.0,
        minlenratio: float = 10.0,
        modality: str = "codec",
    ):
        """Initialize Text2Speech module."""
        assert check_argument_types()

        # setup model
        model, train_args = SpeechLMTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.modality = modality

        self.inference_method = SpeechLMInference(
            corelm = model.corelm,
            predictor = model.predictor,
            emb = model.emb,
            device = device,
            token_list = train_args.token_list,
            token_bias = train_args.token_bias,
            search_type = search_type,
            search_algo = search_algo,
            nbest = nbest,
            sampling_temperature = sampling_temperature,
            top_k = top_k,
            maxlenratio=maxlenratio,
            minlenratio=minlenratio,
            modality=modality,
        )
    
    def __call__(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        enc_seq = kwargs.get("enc_seq", None)
        enc_seq_lengths = kwargs.get("enc_seq_lengths", None)
        if enc_seq is not None or enc_seq_lengths is not None:
            raise NotImplemented('encoder-decoder is not supported')
        
        prefix_len = kwargs["prefix_len"]
        prefix_len = prefix_len.cpu().tolist()[0][0]
        hypos = self.inference_method(
            dec_seq=dec_seq,
            enc_seq=enc_seq,
            prefix_len=prefix_len,
        )

        return hypos
    
    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        if model_tag is not None:
            raise ValueError("Model tag is not supported yet")
        
        return SpeechLM(**kwargs)


def inference(
    output_dir: str,
    batch_size: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    dtype,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    search_type: str = "ar",
    search_algo: str = "sampling",
    nbest: int = 1,
    sampling_temperature: float = 1.0,
    top_k: int = 20,
    minlenratio: float = 0.0,
    maxlenratio: float = 10.0,
    postprocessor: str = None,
    postprocessor_conf: dict = {},
):
    """Run text-to-speech inference."""
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
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

    # 2. parse task
    assert len(data_path_and_name_and_type) == 1, "Can only do inference for one json"
    task_name = json.load(open(data_path_and_name_and_type[0][0]))['task']
    task = speechlm_tasks[task_name]
    output_name, output_modality = task.decoder_entries[-1][:2]
    assert output_modality == postprocessor, f"Postprocessor should be {output_modality}"

    # 3. Build model
    speechlm_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        device=device,
        search_type=search_type,
        search_algo=search_algo,
        nbest=nbest,
        sampling_temperature=sampling_temperature,
        top_k=top_k,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        modality=output_modality,
    )

    speechlm = SpeechLM.from_pretrained(
        model_tag=model_tag,
        **speechlm_kwargs
    )

    post_processor_class = post_processor_choices.get_class(postprocessor)
    post_processor = post_processor_class(**postprocessor_conf).to(device)

    # 4. Build data-iterator
    loader = SpeechLMTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SpeechLMTask.build_preprocess_fn(speechlm.train_args, False),
        collate_fn=SpeechLMTask.build_collate_fn(speechlm.train_args, False),
        allow_variable_data_keys=False,
        inference=True,
        multi_task_dataset=True
    )

    # 5 Start for-loop
    output_dir = Path(output_dir)
    (output_dir / output_name).mkdir(parents=True, exist_ok=True)
    writer = open(output_dir / output_name / f"{output_name}.scp", 'w')
    # (output_dir / "token").mkdir(parents=True, exist_ok=True)
    # (output_dir / "score").mkdir(parents=True, exist_ok=True)

    for idx, (keys, batch) in enumerate(loader, 1):
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert _bs == 1, _bs

        batch = to_device(batch, device=device)
        hypos = speechlm(**batch)
        key = keys[0]

        for h_idx, hypo in enumerate(hypos):
            if output_modality == "codec":
                bias = speechlm.train_args.token_bias['codec']
                waveform = post_processor(hypo.generated - bias).cpu()

                print('waveform shape: ', waveform.size())

                wave_name = f"{key}_sample{h_idx}"
                wave_path = output_dir / output_name / f"{wave_name}.wav"
                writer.write(f"{wave_name} {str(wave_path)}\n")

                torchaudio.save(
                    wave_path, waveform, sample_rate=post_processor.sample_rate
                )

                logging.info(f"save audio {wave_name}: {wave_path}")


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="TTS inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
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
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=10.0,
        help="Maximum length ratio in decoding",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Minimum length ratio in decoding",
    )
    group.add_argument(
        "--search_type",
        type=str,
        default="ar",
        choices=["ar", "nar", "ar_nar"],
        help="the search style of SpeechLM",
    )
    group.add_argument(
        "--search_algo",
        type=str,
        default="sampling",
        choices=["sampling", "teacher_force", "beam_search", "greedy_search"],
        help="the search algorithm of SpeechLM",
    )
    group.add_argument(
        "--nbest",
        type=int,
        default=1,
        help="The number of hypotheses to return",
    )
    group.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="the temperature of softmax during sampling",
    )
    group.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="if positive, restrict the sampling to top-k tokens with highest probs."
    )

    group = parser.add_argument_group("Postprocessor related")
    post_processor_choices.add_arguments(group)

    # TODO: handle the post-processor
    return parser


def main(cmd=None):
    """Run SpeechLM model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()