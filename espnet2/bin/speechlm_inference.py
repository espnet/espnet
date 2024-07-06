#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from kaldiio import WriteHelper
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.definitions import tasks as speechlm_tasks
from espnet2.tasks.speechlm import SpeechLMTask, tokenizer_choices

# utilities
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
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
        verbose: bool = False,
        search_algo: str = "topk_sampling",
        inference_nq: Optional[int] = None,
        nbest: int = 1,
        sampling_temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        maxlenratio: float = 0.0,
        minlenratio: float = 10.0,
        modality: str = "codec",
        tokenizer_conf: dict = {},
    ):
        """Initialize SpeechLM module."""

        # (1) setup model
        model, train_args = SpeechLMTask.build_model_from_file(
            train_config, model_file, device
        )
        self.model = model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.verbose = verbose
        self.dtype = dtype
        self.modality = modality
        self.train_args = train_args

        # (2) token_mask
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

        # (3) inference options
        self.inference_opts = SpeechLMInferenceOptions(
            device=device,
            search_algo=search_algo,
            nbest=nbest,
            sampling_temperature=sampling_temperature,
            top_k=top_k,
            top_p=top_p,
            maxlenratio=maxlenratio,
            minlenratio=minlenratio,
            eos=train_args.token_list.index("<sos/eos>"),
            start=train_args.token_list.index(f"<{modality}_start/end>"),
            masks=masks,
            nq=inference_nq if inference_nq is not None else model.corelm.nq,
        )

        # (4) tokenizer: detokenize speechlm tokens to the exact output, e.g. audio or text
        tokenizer_class = tokenizer_choices.get_class(modality)

        if modality == "text_bpe":
            tokenizer_conf.update(dict(
                token_list=train_args.token_list.copy(),
                model=train_args.bpemodel,
            ))
        self.tokenizer = tokenizer_class(**tokenizer_conf)
        try:
            self.tokenizer = self.tokenizer.to(device)
        except:
            logging.warning(f"cannot move tokenizer to device: {device}")
        
        if modality in ["codec"]:
            self.bias = token_bias[modality]
        else:
            self.bias = 0

    @typechecked
    @torch.no_grad()
    def __call__(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        prefix_len: torch.Tensor,
        **kwargs,
    ) -> Tuple[List[Any], List[Any]]:
        """Run SpeechLM inference"""

        enc_seq = kwargs.get("enc_seq", None)
        enc_seq_lengths = kwargs.get("enc_seq_lengths", None)
        if enc_seq is not None or enc_seq_lengths is not None:
            raise NotImplementedError("encoder-decoder is not supported yet.")

        # (1) language model inference
        # NOTE(Jinchuan): the token dec_seq[prefix_len] is exactly
        # self.inference_opts.start and will be handled by the
        # inference algorithm. We discard it here.
        prefix_len = prefix_len.squeeze(1)
        gen_tokens, gen_scores = self.model.corelm.inference(
            prefix=dec_seq[:, :prefix_len],
            opts=self.inference_opts,
            enc_seq=None,
            suffix=dec_seq[:, prefix_len + 1 :],
        )

        # (2) predicted tokens detokenization
        generated = []
        for token, score in zip(gen_tokens, gen_scores):
            if self.modality == "codec":
                token = token.view(-1) - self.bias
                score = score.view(-1)
            else:
                token = token[:, 0].view(-1) - self.bias
                score = score[:, 0].view(-1)
            content = self.tokenizer.detokenize(token.clone())
            
            generated.append(
                (token, score, content)
            )

        # (3) prefix tokens detokenization
        conditions = []
        if self.verbose:
            # [32, 64) is reserved for modality start. See:
            # espnet2.speechlm.definitions.py
            starts = (
                torch.logical_and(
                    dec_seq[0, :prefix_len, 0] >= 32,
                    dec_seq[0, :prefix_len, 0] < 64,
                )
                .nonzero(as_tuple=True)[0]
                .cpu()
                .tolist()
            )
            starts = starts + [prefix_len.cpu().item()]

            for idx in range(len(starts) - 1):
                start, end = starts[idx], starts[idx + 1]
                this_modality = self.train_args.token_list[dec_seq[0, start, 0].item()]
                this_modality = this_modality.lstrip("<").rstrip("_start/end>")
                token = dec_seq[0, start + 1 : end]
                detokenized = False

                # TODO(Jinchuan): support more detokenization options latre for other tasks
                if self.modality == "codec" and this_modality in ["codec", "spk"]:
                    token = (token - self.train_args.token_bias["codec"]).view(-1)
                    content = self.tokenizer.detokenize(token.clone())
                    detokenized = True

                elif this_modality in ["g2p"]:
                    token = token[:, 0]
                    content = " ".join([self.train_args.token_list[c] for c in token.cpu().tolist()])
                    detokenized = True
                
                else:
                    token = token.flatten()
                    content = None
                    detokenized = False

                conditions.append((token, content, this_modality, detokenized))

        return generated, conditions

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
    output_dir: Path,
    batch_size: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    dtype: str,
    log_level: Union[int, str],
    rank: int,
    verbose: bool,
    # data related
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    # model related
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    # inference related
    search_algo: str = "topk_sampling",
    nbest: int = 1,
    sampling_temperature: float = 1.0,
    top_k: int = 20,
    top_p: float = 0.8,
    minlenratio: float = 0.0,
    maxlenratio: float = 10.0,
    inference_nj: Optional[int] = 1,
    # tokenizer related
    tokenizer: str = "",
    tokenizer_conf: dict = {},
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

    if torch.cuda.is_available() and ngpu >= 1:
        if torch.cuda.device_count() > 1:
            device_id = rank % torch.cuda.device_count()
        else:
            device_id = 0
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    # 1. parse task
    assert len(data_path_and_name_and_type) == 1, "Can only do inference for one json"
    task_name = json.load(open(data_path_and_name_and_type[0][0]))["task"]
    task = speechlm_tasks[task_name]
    output_name, output_modality = task.decoder_entries[-1][:2]
    assert (
        output_modality == tokenizer
    ), f"Tokenizer should be {output_modality} for task: {task_name}"

    # 2. Build model
    speechlm_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        device=device,
        verbose=verbose,
        search_algo=search_algo,
        inference_nq=inference_nj,
        nbest=nbest,
        sampling_temperature=sampling_temperature,
        top_k=top_k,
        top_p=top_p,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        modality=output_modality,
        tokenizer_conf=tokenizer_conf,
    )

    speechlm = SpeechLM.from_pretrained(model_tag=model_tag, **speechlm_kwargs)

    # 3. Build data-iterator
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

    # 4 Start for-loop
    (output_dir / output_name).mkdir(parents=True, exist_ok=True)
    prefix_triplets = [
        triplet
        for triplet in task.encoder_entries + task.decoder_entries
        if triplet not in task.target_entries
    ]

    writer = open(output_dir / output_name / "example_list", "w")
    token_writer = WriteHelper(f'ark,scp:{str(output_dir / output_name / "token")}.ark,{str(output_dir / output_name / "token")}.scp')
    score_writer = WriteHelper(f'ark,scp:{str(output_dir / output_name / "score")}.ark,{str(output_dir / output_name / "score")}.scp')
    prefix_writers = [None for _ in prefix_triplets]
    prefix_token_writers = [None for _ in prefix_triplets]

    for _, (keys, batch) in enumerate(loader, 1):
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert _bs == 1, _bs

        batch = to_device(batch, device=device)
        key = keys[0]
        
        logging.info(f"Inference on example: {key}")

        # NOTE (Jinchuan): set random seed for each example so each result can be
        # reproduced independently.
        set_all_random_seed(seed)

        # (1) model infernece
        generated, conditions = speechlm(**batch)

        # (2) parse and save generated content
        for h_idx, (token, score, content) in enumerate(generated):
            example_name = f"{key}_sample{h_idx}"
            
            if output_modality == "codec":
                wave_path = output_dir / output_name / f"{example_name}.wav"
                writer.write(f"{example_name} {str(wave_path)}\n")

                torchaudio.save(
                    wave_path,
                    content.view(1, -1).cpu(),
                    sample_rate=speechlm.tokenizer.sample_rate,
                    bits_per_sample=16,
                )
                logging.info(f"save generated audio {example_name}: {wave_path}")
            
            elif output_modality == "text_bpe":
                writer.write(f"{example_name} {content[0]}\n")

            else:
                raise NotImplementedError(
                    f"Output modality {output_modality} is not supported"
                )

            if isinstance(token, torch.Tensor):
                token = token.int().cpu().numpy()
            if isinstance(score, torch.Tensor):
                score = score.float().cpu().numpy()

            token_writer[example_name] = token
            score_writer[example_name] = score

        # (3) parse and save conditon content
        if verbose:
            # NOTE(Jinchuan): remove task prefix so that the recorded prefix will have the same
            # format like other reference file. E.g., text in original dataset
            key = key.lstrip(f"{task_name}_")
            
            assert len(conditions) == len(prefix_triplets)
            for c_idx, (token, content, modality, detokenized) in enumerate(conditions):
                if not detokenized:
                    continue

                name, _modality, _ = prefix_triplets[c_idx]
                assert modality == _modality, (modality, _modality)

                # (3.1) save content
                if prefix_writers[c_idx] == None:
                    (output_dir / name).mkdir(parents=True, exist_ok=True)
                    prefix_writer = open(output_dir / name / "example_list", "w")
                    prefix_writers[c_idx] = prefix_writer
                    
                prefix_writer = prefix_writers[c_idx]
                if modality in ["codec", "spk"]:
                    content_path = output_dir / name / f"{key}.wav"
                    torchaudio.save(
                        content_path,
                        content.view(1, -1).cpu(),
                        sample_rate=speechlm.tokenizer.sample_rate,
                        bits_per_sample=16,
                        encoding="PCM_S",
                    )
                    prefix_writer.write(f"{key} {content_path}\n")
                    logging.info(f"save prefix audio {name} audio {key}: {content_path}")

                elif modality in ["g2p"]:
                    prefix_writer.write(f"{key} {content}\n")
                    
                    logging.info(f"prefix part {modality}: {content}")

                else:
                    raise ValueError(
                        f"save prefix in modality {modality} is not supported yet"
                    )
                
                if prefix_token_writers[c_idx] is None:
                    prefix_token_writer = WriteHelper(
                        f'ark,scp:{str(output_dir / name / "token")}.ark,{str(output_dir / name / "token")}.scp'
                    )
                    prefix_token_writers[c_idx] = prefix_token_writer
                prefix_token_writer = prefix_token_writers[c_idx]
                prefix_token_writer[key] = token.int().cpu().numpy()
        
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
        type=Path,
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
    parser.add_argument(
        "--rank", type=int, default=1, help="the job rank in decoding process"
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="If true, also dump the condition in the prefix (in the same modality only)",
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
        default="topk_sampling",
        choices=["topk_sampling", "topp_sampling", "teacher_force", "beam_search", "greedy_search"],
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
        default=30,
        help="if positive, restrict the sampling to top-k tokens with highest probs.",
    )
    group.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="if positive, restrict the sampling to tokens with top-p probs",
    )
    group.add_argument(
        "--inference_nj",
        type=int,
        default=None,
        help="nj used in inference, should be the same or smaller than the nq in training",
    )

    group = parser.add_argument_group("tokenizer related")
    tokenizer_choices.add_arguments(group)

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
