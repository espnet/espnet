#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.beats_encoder import BeatsEncoder, BeatsPretrainingPredictor
from espnet2.beats.espnet_model import BeatsPretrainModel, BeatsTokenizerPretrainModel
from espnet2.speechlm.tokenizer.beats_tokenizer import (
    BeatsTokenizer,
    BeatsTokenizerPretrainingPredictor,
)
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str_or_none

logger = logging.getLogger(__name__)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        beats=BeatsEncoder,
        beats_tokenizer=BeatsTokenizer,
    ),
    type_check=AbsEncoder,
    default="beats",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        beats=BeatsPretrainModel,
        beats_tokenizer=BeatsTokenizerPretrainModel,
    ),
    type_check=AbsESPnetModel,
    default="beats",
)


class BeatsTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --encoder and --encoder_conf
        encoder_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
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
                "normal",
                None,
            ],
        )
        group.add_argument(
            "--collate_fn_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for collate_fn class.",
        )
        group.add_argument(
            "--use_preprocessor", type=str2bool, default=True, help="Use preprocessor"
        )
        group.add_argument("--waveform_input", type=str2bool, default=False)

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type="word",
                token_list=args.token_list,
                text_name="target",
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("speech", "target")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("speech_lengths", "target_lengths")
        return retval

    @classmethod
    @typechecked
    def build_model(
        cls, args: argparse.Namespace
    ) -> Union[BeatsPretrainModel, BeatsTokenizerPretrainModel]:
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logger.info(f"Vocabulary size: {vocab_size }")
        n_codebook_vectors = vocab_size - 1

        assert (
            args.model == args.encoder
        ), f"Model and encoder choice should match {pretraining_model}"
        pretraining_model = args.model

        # 1. frontend
        input_size = 1  # model will extract features
        # 2. Encoder and Decoder
        encoder_class = encoder_choices.get_class(args.encoder)

        assert (
            pretraining_model == "beats"
        ), f"Model should be beats: {pretraining_model}"
        config = args.encoder_conf.pop("beats_config", None)
        encoder = encoder_class(
            input_size=input_size,
            beats_config=config,
            **args.encoder_conf,
        )
        predictor = BeatsPretrainingPredictor(beats_config=config)
        args.encoder_conf["beats_config"] = config  # put it back
        kwargs = {"encoder": encoder, "decoder": predictor}

        model = BeatsPretrainModel(
            **kwargs,
            **args.model_conf,
        )
        # Skip espnet intialization, beats does that internally.

        return model


class BeatsTokenizerTask(BeatsTask):

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        BeatsTask.add_task_arguments(parser)
        required = parser.get_default("required")
        required.remove("token_list")
        required += ["beats_teacher_ckpt_path"]

        group = parser.add_argument_group(description="Beats Tokenizer Task related")
        group.add_argument("--beats_teacher_ckpt_path", type=str, default=None)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            # only speech preprocessing
            retval = CommonPreprocessor(
                train=train,
                text_name="",
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("speech_lengths",)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> BeatsTokenizerPretrainModel:
        assert (
            args.beats_teacher_ckpt_path
        ), "Teacher checkpoint path is required for tokenizer pretraining."

        encoder = BeatsTokenizer(**args.encoder_conf)
        predictor = BeatsTokenizerPretrainingPredictor(
            tokenizer_config=args.encoder_conf.get("tokenizer_config", None)
        )
        teacher = BeatsEncoder(
            input_size=1, beats_ckpt_path=args.beats_teacher_ckpt_path
        )

        # Build model
        model = BeatsTokenizerPretrainModel(
            encoder=encoder,
            decoder=predictor,
            teacher=teacher,
            waveform_input=args.model_conf.get("waveform_input", False),
        )
        if args.init:
            initialize(model, args.init)
        return model
