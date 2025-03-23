#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from kaldiio import WriteHelper
from packaging.version import parse as V  # noqa
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.definitions import tasks as speechlm_tasks
from espnet2.tasks.speechlm import SpeechLMTask

# utilities
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none  # noqa
from espnet.utils.cli_utils import get_commandline_args


class SpeechLM:
    """SpeechLM class.

    Examples: TODO(Jinchuan): will finish this when the code is stable.
    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        search_algo: str = "sampling",
        inference_nq: int = None,
        nbest: int = 1,
        sampling_temperature: float = 1.0,
        top_k: int = 20,
        maxlenratio: float = 0.0,
        minlenratio: float = 10.0,
        modality: str = "codec",
        post_processor_conf: dict = {},
    ):
        """Initialize SpeechLM module."""

        # setup model
        model, train_args = SpeechLMTask.build_model_from_file(
            train_config, model_file, device
        )
        self.model = model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args

        # token_mask
        token_bias = train_args.token_bias
        token_list = train_args.token_list
        inference_nq = model.corelm.nq if inference_nq is None else inference_nq
        assert inference_nq <= model.corelm.nq
        valid_start = token_bias[modality]
        valid_end = min(
            [s for s in token_bias.values() if s > valid_start] + [len(token_list)]
        )

        masks = torch.ones(inference_nq, len(token_list)).to(device).bool()
        if modality == "codec":
            increment = (valid_end - valid_start) // train_args.codec_token_per_frame
            for l_idx in range(inference_nq):
                masks[
                    l_idx,
                    valid_start
                    + l_idx * increment : valid_start
                    + (l_idx + 1) * increment,
                ] = False
        else:
            masks[:, valid_start:valid_end] = False

        # inference options
        self.inference_opts = SpeechLMInferenceOptions(
            device=device,
            search_algo=search_algo,
            nbest=nbest,
            sampling_temperature=sampling_temperature,
            top_k=top_k,
            maxlenratio=maxlenratio,
            minlenratio=minlenratio,
            eos=train_args.token_list.index("<sos/eos>"),
            start=train_args.token_list.index(f"<{modality}_start/end>"),
            masks=masks,
            nq=inference_nq if inference_nq is not None else model.corelm.nq,
        )

        # post_processor: transform tokens to the target modality. E.g., speech, text.
        if modality in ["codec"]:
            self.bias = token_bias[modality]
        else:
            self.bias = 0

    @typechecked
    def __call__(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        prefix_len: torch.Tensor,
        **kwargs,
    ) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor]]:
        """Run SpeechLM inference"""

        enc_seq = kwargs.get("enc_seq", None)
        enc_seq_lengths = kwargs.get("enc_seq_lengths", None)
        if enc_seq is not None or enc_seq_lengths is not None:
            raise NotImplementedError("encoder-decoder is not supported yet.")

        # language model inference
        # Note(Jinchuan): the token dec_seq[prefix_len] is exactly
        # self.inference_opts.start and will be handled by the
        # inference algorithm. We discard it here.
        prefix_len = prefix_len.squeeze(1)
        gen_tokens, gen_scores = self.model.corelm.inference(
            prefix=dec_seq[:, :prefix_len],
            opts=self.inference_opts,
            enc_seq=None,
            suffix=dec_seq[:, prefix_len + 1 :],
        )

        if gen_tokens is None and gen_scores is None:
            return None, None, None

        # post-processing
        generated = []
        for gen_token in gen_tokens:
            gen_token = gen_token - self.bias
            generated.append(self.post_processor(gen_token))

        return generated, gen_tokens, gen_scores

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        if model_tag is not None:
            raise ValueError("Model tag is not supported yet")

        return SpeechLM(**kwargs)


@typechecked
def inference(
    # general
    output_dir: str,
    batch_size: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    dtype: str,
    log_level: Union[int, str],
    # data related
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    # model related
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    # inference related
    search_algo: str = "sampling",
    nbest: int = 1,
    sampling_temperature: float = 1.0,
    top_k: int = 20,
    minlenratio: float = 0.0,
    maxlenratio: float = 10.0,
    inference_nj: Optional[int] = 1,
    # post_processor related
    postprocessor: str = None,
    postprocessor_conf: dict = {},
):
    """Run SpeechLM inference."""
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
    task_name = json.load(open(data_path_and_name_and_type[0][0]))["task"]
    task = speechlm_tasks[task_name]
    output_name, output_modality = task.decoder_entries[-1][:2]
    assert (
        output_modality == postprocessor
    ), f"Postprocessor should be {output_modality}"

    # 3. Build model
    speechlm_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        device=device,
        search_algo=search_algo,
        inference_nq=inference_nj,
        nbest=nbest,
        sampling_temperature=sampling_temperature,
        top_k=top_k,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        modality=output_modality,
        post_processor_conf=postprocessor_conf,
    )

    speechlm = SpeechLM.from_pretrained(model_tag=model_tag, **speechlm_kwargs)

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
        multi_task_dataset=True,
    )

    # 5 Start for-loop
    output_dir = Path(output_dir)
    (output_dir / output_name).mkdir(parents=True, exist_ok=True)
    (output_dir / "token").mkdir(parents=True, exist_ok=True)
    (output_dir / "score").mkdir(parents=True, exist_ok=True)

    writer = open(output_dir / output_name / f"{output_name}_list", "w")
    token_writer = WriteHelper(f'ark:{str(output_dir / "token" / "token")}.ark')
    score_writer = WriteHelper(f'ark:{str(output_dir / "score" / "score")}.ark')

    for _, (keys, batch) in enumerate(loader, 1):
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert _bs == 1, _bs

        batch = to_device(batch, device=device)
        key = keys[0]
        logging.info(f"Inference on example: {key}")

        contents, tokens, scores = speechlm(**batch)
        if contents is None:
            logging.info(f"fail on example: {key}")
            continue

        for h_idx, (content, token, score) in enumerate(zip(contents, tokens, scores)):
            example_name = f"{key}_sample{h_idx}"

            if output_modality == "codec":
                wave_path = output_dir / output_name / f"{example_name}.wav"
                writer.write(f"{example_name} {str(wave_path)}\n")

                torchaudio.save(
                    wave_path,
                    content.cpu(),
                    sample_rate=speechlm.post_processor.sample_rate,
                )
                logging.info(f"save audio {example_name}: {wave_path}")

            else:
                raise NotImplementedError(
                    f"Output modality {output_modality} is not supported"
                )

            if isinstance(token, torch.Tensor):
                token = token.int().flatten().cpu().numpy()
            if isinstance(score, torch.Tensor):
                score = score.float().flatten().cpu().numpy()

            token_writer[example_name] = token
            score_writer[example_name] = score


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="SpeechLM inference",
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

    group = parser.add_argument_group("Infernece related")
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
        help="if positive, restrict the sampling to top-k tokens with highest probs.",
    )
    group.add_argument(
        "--inference_nj",
        type=int,
        default=None,
        help="nj used in inference, should be the same/smaller than the nq in train",
    )

    group = parser.add_argument_group("Postprocessor related")

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
