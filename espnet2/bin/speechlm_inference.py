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
import yaml
from kaldiio import WriteHelper
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.definitions import SPEECHLM_TASKS, SpeechLMTaskTemplate
from espnet2.tasks.speechlm import SpeechLMTask, tokenizer_choices

# utilities
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2triple_str, str_or_none, str2bool
from espnet.utils.cli_utils import get_commandline_args


class SpeechLM:
    """SpeechLM class.

    Examples: TODO(Jinchuan): will finish this when the code is stable.
    """

    @typechecked
    def __init__(
        self,
        task: Union[str, SpeechLMTaskTemplate],
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        search_algo: str = "topk_sampling",
        inference_nq: Optional[int] = None,
        nbest: int = 1,
        sampling_temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        maxlenratio: float = 0.0,
        minlenratio: float = 10.0,
        codec_ssl_predict_ssl: bool = True,
        codec_conf: dict = None,
    ):
        """Initialize SpeechLM module."""

        # (1) setup model
        model, train_args = SpeechLMTask.build_model_from_file(
            train_config, model_file, device
        )
        self.model = model.to(dtype=getattr(torch, dtype)).eval()
        self.preprocessor = SpeechLMTask.build_preprocess_fn(train_args, False)
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.task = SPEECHLM_TASKS[task] if isinstance(task, str) else task

        self.token_list = train_args.token_list
        self.token_bias = train_args.token_bias
        self.modalities = [triplet[1] for triplet in self.task.data_triplets]
        self.pad = self.token_list.index("<pad>")

        # (2) predict mask
        self.inference_nq = model.corelm.nq if inference_nq is None else inference_nq
        predict_masks = dict()
        all_boundaries = list(self.token_bias.values()) + [len(self.token_list)]
        for modality in set(self.modalities):
            if modality == "spk":  # never predict "spk" modality
                continue

            mask = torch.ones(self.inference_nq, len(self.token_list)).to(device).bool()
            start = self.token_bias[modality]
            end = min([b for b in all_boundaries if b > start])
            if modality == "codec":
                assert (end - start) % train_args.codec_token_per_frame == 0
                inc = (end - start) // train_args.codec_token_per_frame
                for n in range(self.inference_nq):
                    mask[n, start + n * inc : start + (n + 1) * inc] = False
                
                self.codec_start = start

            elif modality == "codec_ssl":
                ssl_start = self.token_list.index("<ssl_code1>")
                codec_start = self.token_list.index("<codec_layer0_code0>")
                
                if codec_ssl_predict_ssl:
                    ssl_end = end if ssl_start > codec_start else codec_start
                    mask[0, ssl_start: ssl_end] = False
                else:
                    mask[0, self.pad] = False

                codec_end = end if codec_start > ssl_start else ssl_start
                assert (codec_end - codec_start) % (train_args.codec_token_per_frame - 1) == 0
                inc = (codec_end - codec_start) // (train_args.codec_token_per_frame - 1)
                for n in range(self.inference_nq - 1):
                    mask[n + 1, codec_start + n * inc : codec_start + (n + 1) * inc] = False
                
                self.codec_start = codec_start
            
            elif modality == "text_bpe":
                mask[0, start: end] = False
                mask[1:, self.pad] = False 

            else:
                mask[:, start:end] = False
            
            # When more than one target, allow modality switch
            if len(self.task.targets) > 1:
                mask[0, 32:64] = False

            modality = f"<{modality}_start/end>"
            modality_idx = self.token_list.index(modality)
            predict_masks[modality_idx] = mask

        # (3) inference options
        start_modality = self.modalities[len(self.task.conditions)]
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
            start=train_args.token_list.index(f"<{start_modality}_start/end>"),
            masks=predict_masks,
            nq=self.inference_nq,
        )

        # (4) Only a limited number of modalities support detokenization
        # (4.1) offline tokenizers should be resumed from external config as they are
        #       not included in preprocessor.
        if any([m in self.modalities for m in ["codec", "codec_ssl", "spk"]]):
            self.codec_tokenizer = tokenizer_choices.get_class("codec")(**codec_conf)
            self.codec_tokenizer.to(self.device)
        else:
            self.codec_tokenizer = None

        # (4.2) online tokenizers should be from preprocessor.
        if "text_bpe" in self.modalities:
            self.text_bpe_tokenizer = self.preprocessor.bpe
        else:
            self.text_bpe_tokenizer = None

    @typechecked
    @torch.no_grad()
    def __call__(self, data: Dict) -> List:
        """Run SpeechLM inference"""

        if not "dec_seq" in data:
            data = self.preprocessor(data)

        dec_seq = data.get("dec_seq")
        prefix_len = data.get("prefix_len").squeeze(1)

        # (1) language model inference
        gen_tokens, _ = self.model.corelm.inference(
            prefix=dec_seq[:, :prefix_len],
            opts=self.inference_opts,
            enc_seq=None,
            suffix=dec_seq[:, prefix_len + 1 :],
        )

        # (2) record the prefix segments
        retval = [[] for _ in self.modalities]
        prefix = dec_seq[0, :prefix_len]

        segments = self.parse_sequence(prefix)
        if len(segments) != len(self.task.conditions):
            raise ValueError("Invalid Condition segments")

        for idx, segment in enumerate(segments):
            retval[idx].append(segment)

        # (3) record the generated segments
        start = self.inference_opts.start
        start = torch.Tensor([start] * self.inference_nq).view(1, -1).to(self.device)
        for gen_token in gen_tokens:
            gen_token = torch.cat([start.long(), gen_token], dim=0)
            segments = self.parse_sequence(gen_token)

            if len(segments) != len(self.task.targets):
                logging.warning(f"Invalid target segments. Skip")
                continue

            for idx2, segment in enumerate(segments, start=len(self.task.conditions)):
                retval[idx2].append(segment)

        return retval

    def parse_sequence(self, sequence):
        segment_starts = torch.logical_and(sequence[:, 0] >= 32, sequence[:, 0] < 64)
        segment_starts = segment_starts.nonzero(as_tuple=True)[0].cpu().tolist()
        segment_starts = segment_starts + [len(sequence)]

        retval = []
        for start, end in zip(segment_starts[:-1], segment_starts[1:]):
            modality_id = sequence[start, 0].int().item()
            modality = self.token_list[modality_id]
            modality = modality.removeprefix("<").removesuffix("_start/end>")
            segment = sequence[start + 1 : end]
            
            if modality in ["codec", "spk", "codec_ssl"]:
                if "codec_ssl" in self.token_bias:
                    segment = segment[:, 1:]
                    n_codebook = self.inference_nq - 1
                else:
                    n_codebook = self.inference_nq
                segment = segment[segment[:, 0] != self.pad]

                segment = segment.contiguous().view(-1) - self.codec_start
                detokenized = self.codec_tokenizer.detokenize(
                    segment.clone(),
                    n_codebook=n_codebook,
                )
            
            elif modality in ["text_bpe"]:
                segment = segment[segment[:, 0] != self.pad]
                segment = segment[:, 0]
                detokenized = self.text_bpe_tokenizer.tokens2text(
                    [self.token_list[tok] for tok in segment]
                )
                # sentencepiece will include "\n" but huggingface will not.
                # make it uniform
                detokenized = detokenized.strip() + "\n"

            else:
                segment = segment[:, 0] - self.token_bias[modality]
                detokenized = None

            retval.append((modality, segment, detokenized))

        return retval

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            hf_args = d.download_and_unpack(model_tag)

            if "inference_config" in hf_args:
                infer_args = yaml.safe_load(open(hf_args.pop("inference_config")))
                hf_args.update(infer_args)

            # NOTE(Jinchuan): do not override the external arguments
            for key, value in hf_args.items():
                if key not in kwargs:
                    kwargs[key] = value

        if "task" not in kwargs:
            raise ValueError("Please specify the task")

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
    # data related
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
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
    inference_nq: Optional[int] = 1,
    codec_ssl_corrupt_prob: float = 0.0,
    codec_ssl_predict_ssl: bool = True,
    # offline tokenizers
    codec_conf: dict = None,
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

    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        device = "cpu"

    # 1. parse task
    assert len(data_path_and_name_and_type) == 1, "Can only do inference for one json"
    task_name = json.load(open(data_path_and_name_and_type[0][0]))["task"]
    task = SPEECHLM_TASKS[task_name]

    # 2. Build model
    speechlm_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        device=device,
        search_algo=search_algo,
        inference_nq=inference_nq,
        nbest=nbest,
        sampling_temperature=sampling_temperature,
        top_k=top_k,
        top_p=top_p,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        task=task,
        codec_ssl_predict_ssl=codec_ssl_predict_ssl,
        codec_conf=codec_conf,
    )

    speechlm = SpeechLM.from_pretrained(model_tag=model_tag, **speechlm_kwargs)

    # 3. Build data-iterator
    if inference_nq is not None and speechlm.train_args.codec_token_in_use != inference_nq:
        logging.warning(
            f"The model is trained with nq={speechlm.train_args.codec_token_in_use} "
            f"While you are inference with {inference_nq}. "
        )
        speechlm.train_args.codec_token_in_use = inference_nq
    if codec_ssl_corrupt_prob != speechlm.train_args.codec_ssl_corrupt_prob:
        speechlm.train_args.codec_ssl_corrupt_prob = codec_ssl_corrupt_prob
    loader = SpeechLMTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        num_workers=num_workers,
        preprocess_fn=SpeechLMTask.build_preprocess_fn(speechlm.train_args, False),
        collate_fn=SpeechLMTask.build_collate_fn(speechlm.train_args, False),
        allow_variable_data_keys=False,
        inference=True,
        multi_task_dataset=True,
    )

    # 4. build writer
    writers, token_writers = dict(), dict()
    for triplet in task.data_triplets:
        name, modality, _ = triplet
        (output_dir / name).mkdir(parents=True, exist_ok=True)
        file_name = str(output_dir / name / ("token_" + name))
        token_writers[name] = WriteHelper(f"ark,scp:{file_name}.ark,{file_name}.scp")
        if modality in ["spk", "codec", "text_bpe", "codec_ssl"]:
            file_name = str(output_dir / name / name)
            writers[name] = open(file_name, "w")

    # 5. inference loop
    for _, (keys, batch) in enumerate(loader, 1):
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert _bs == 1, _bs

        batch = to_device(batch, device=device)
        key = keys[0]

        # NOTE(Jinchuan): make each example independently rseproducible
        set_all_random_seed(seed)

        # 5.1 model infernece
        logging.info(f"Inference on example: {key}")
        all_segments = speechlm(batch)

        for triplet, segments in zip(task.data_triplets, all_segments):
            name, _modality, _ = triplet
            generated = triplet in task.targets
            for idx, segment in enumerate(segments):
                modality, token, detokenized = segment
                assert modality == _modality, (modality, _modality)

                if generated:
                    example_name = f"{key}_sample{idx}"
                else:
                    example_name = key.removeprefix(f"{task_name}_")

                # 5.2 save token
                token_writers[name][example_name] = token.int().cpu().numpy()

                # 5.3 save tokenized results
                if detokenized is not None:
                    if modality in ["codec", "spk", "codec_ssl"]:
                        audio_path = output_dir / name / f"{example_name}.wav"
                        torchaudio.save(
                            str(audio_path),
                            detokenized.view(1, -1).cpu(),
                            sample_rate=speechlm.codec_tokenizer.sample_rate,
                            bits_per_sample=16,
                            encoding="PCM_S",
                        )
                        writers[name].write(f"{example_name} {audio_path}\n")
                        logging.info(f"Save audio: {audio_path}")

                    elif modality in ["text_bpe"]:
                        writers[name].write(f"{example_name} {detokenized}")
                        logging.info(f"Save text: {detokenized}")

                    else:
                        raise ValueError(
                            f"Modality {modality} is tokenized but no method to save."
                        )


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
        default=0,
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
        choices=[
            "topk_sampling",
            "topp_sampling",
            "teacher_force",
            "greedy_search",
        ],
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
        "--inference_nq",
        type=int,
        default=None,
        help="nq in inference stage",
    )
    group.add_argument(
        "--codec_ssl_corrupt_prob",
        type=float,
        default=0.0,
        help="the prob of corrputing ssl tokens in codec_ssl modality in sequence level "
          "1.0 means no ssl tokens in use; 0.0 means use ssl tokens "
          "This is only applied to the prefix sequence"
    )
    group.add_argument(
        "--codec_ssl_predict_ssl",
        type=str2bool,
        default=True,
        help="If true, allow to predict ssl token in codec_ssl modality prediction "
          "Otherwise, the first layer of codec_ssl is always paddings"
    )

    # Offline tokenizer configurations. The offline tokenizers are not used during
    # training and thus should be specified externally.
    for tokenizer in ["codec"]:
        tokenizer_class = tokenizer_choices.get_class(tokenizer)
        group.add_argument(
            f"--{tokenizer}_conf",
            action=NestedDictAction,
            default=get_default_kwargs(tokenizer_class),
            help=f"The keyword arguments for {tokenizer} class.",
        )

    return parser


def main(cmd=None):
    """Run SpeechLM model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    print("kwargs: ", dict(kwargs))
    inference(**kwargs)


if __name__ == "__main__":
    main()
