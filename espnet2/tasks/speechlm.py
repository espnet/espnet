import argparse
import json
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

# CoreLM
from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.core_lm.ar_multiscale import MultiScaleLM
from espnet2.speechlm.core_lm.valle import ValleLM
from espnet2.speechlm.espnet_model import ESPnetSpeechLMModel
from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize

# Top-level model
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import SpeechLMPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs  # noqa
from espnet2.utils.nested_dict_action import NestedDictAction  # noqa
from espnet2.utils.types import int_or_none, str2bool, str_or_none

corelm_choices = ClassChoices(
    "corelm",
    classes=dict(
        multiscale=MultiScaleLM,
        valle=ValleLM,
    ),
    type_check=AbsCoreLM,
    default="valle",
)

tokenizer_choices = ClassChoices(
    "tokenizer",
    classes=dict(
        codec=CodecTokenizer,
    ),
    type_check=AbsTokenizer,
    default=None,
)

model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetSpeechLMModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)


class SpeechLMTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --corelm and --corelm_conf
        corelm_choices,
        # --tokenizer and --tokenizer_conf
        tokenizer_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list", "token_bias"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--token_bias",
            type=str_or_none,
            default=None,
            help="A json file to specify the start index of each modality",
        )
        group.add_argument(
            "--encoder_decoder_format",
            type=str2bool,
            default=False,
            help="If true, work with encoder-decoder; otherwise decoder-only",
        )
        group.add_argument(
            "--speaker_prompt_length",
            type=int,
            default=150,
            help="the length of speaker prompt, in #frame",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file fo sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--codec_token_per_frame",
            type=int,
            default=1,
            help="Number of original codec codes for each frame",
        )
        group.add_argument(
            "--codec_token_in_use",
            type=int_or_none,
            default=None,
            help="Number of codec codes in exact use",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

        return parser

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        int_pad = args.token_list.index("<pad>")
        return CommonCollateFn(int_pad_value=int_pad)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:

        # (Jinchuan) SpeechLM task will always use the preprocess_fn
        retval = SpeechLMPreprocessor(
            token_list=args.token_list,
            token_bias=args.token_bias,
            encoder_decoder_format=args.encoder_decoder_format,
            bpemodel=args.bpemodel,
            non_linguistic_symbols=args.non_linguistic_symbols,
            text_cleaner=args.cleaner,
            g2p_type=args.g2p,
            codec_token_per_frame=args.codec_token_per_frame,
            codec_token_in_use=args.codec_token_in_use,
            speaker_prompt_length=args.speaker_prompt_length,
        )

        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("dec_seq",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("enc_seq", "prefix_len")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> Union[AbsESPnetModel]:

        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip("\n") for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or list")

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        if isinstance(args.token_bias, str):
            token_bias = json.load(open(args.token_bias))
            args.token_bias = token_bias
        elif isinstance(args.token_bias, Dict):
            token_bias = args.token_bias
        else:
            raise RuntimeError("token_list must be str or dict")
        logging.info(f"Token Bias: {token_bias}")

        # 1. Build CoreLM module
        corelm_class = corelm_choices.get_class(args.corelm)
        corelm = corelm_class(
            vocab_size=len(token_list), nq=args.codec_token_in_use, **args.corelm_conf
        )

        # 3. Build model
        model_class = model_choices.get_class(args.model)
        model = model_class(corelm=corelm, **args.model_conf)

        # 4. Initialize
        if args.init is not None:
            initialize(model, args.init)
        else:
            for m in model.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.Embedding):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        return model
