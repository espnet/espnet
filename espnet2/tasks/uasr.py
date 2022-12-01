import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet import __version__
from espnet2.uasr.espnet_model import ESPnetUASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter
from espnet2.uasr.segmenter.join_segmenter import JoinSegmenter
from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.uasr.discriminator.conv_discriminator import ConvDiscriminator
from espnet2.uasr.generator.abs_generator import AbsGenerator
from espnet2.uasr.generator.conv_generator import ConvGenerator
from espnet2.uasr.loss.abs_loss import AbsUASRLoss

from espnet2.uasr.loss.discriminator_loss import UASRDiscriminatorLoss
from espnet2.uasr.loss.gradient_penalty import UASRGradientPenalty
from espnet2.uasr.loss.smoothness_penalty import UASRSmoothnessPenalty
from espnet2.uasr.loss.phoneme_diversity_loss import UASRPhonemeDiversityLoss
from espnet2.uasr.loss.pseudo_label_loss import UASRPseudoLabelLoss

from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.uasr_trainer import UASRTrainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
    ),
    type_check=AbsFrontend,
    default="default",
)
segmenter_choices = ClassChoices(
    name="segmenter",
    classes=dict(
        join=JoinSegmenter,
    ),
    type_check=AbsSegmenter,
    default=None,
    optional=True,
)
discriminator_choices = ClassChoices(
    name="discriminator",
    classes=dict(
        conv=ConvDiscriminator,
    ),
    type_check=AbsDiscriminator,
    default="conv",
)
generator_choices = ClassChoices(
    name="generator",
    classes=dict(
        conv=ConvGenerator,
    ),
    type_check=AbsGenerator,
    default="conv",
)
loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        discriminator_loss=UASRDiscriminatorLoss,
        gradient_penalty=UASRGradientPenalty,
        smoothness_penalty=UASRSmoothnessPenalty,
        phoneme_diversity_loss=UASRPhonemeDiversityLoss,
        pseudo_label_loss=UASRPseudoLabelLoss,
    ),
    type_check=AbsUASRLoss,
    default="discriminator_loss",
)


class UASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --segmenter and --segmenter_conf
        segmenter_choices,
        # --discriminator and --discriminator_conf
        discriminator_choices,
        # --generator and --generator_conf
        generator_choices,
        loss_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = UASRTrainer

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
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="phn",
            choices=["phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--losses",
            action=NestedDictAction,
            default=[
                {
                    "name": "discriminator_loss",
                    "conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
            # Loss format would be like:
            # losses:
            #   - name: loss1
            #     conf:
            #       weight: 1.0
            #       smoothed: false
            #   - name: loss2
            #     conf:
            #       weight: 0.1
            #       smoothed: false
        )
        group = parser.add_argument_group(description="Task related")
        group.add_argument(
            "--kenlm_path",
            type=str,
            help="path of n-gram kenlm for validation",
        )

        parser.add_argument(
            "--int_pad_value",
            type=int,
            default=0,
            help="Integer padding value for real token sequence",
        )

        parser.add_argument(
            "--fairseq_checkpoint",
            type=str,
            help="Fairseq checkpoint to initialize model",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=args.int_pad_value)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("pseudo_labels", "input_cluster_id")
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetUASRModel:
        assert check_argument_types()
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
        logging.info(f"Vocabulary size: {vocab_size}")

        # load from fairseq checkpoint
        load_fairseq_model = False
        cfg = None
        if args.fairseq_checkpoint is not None:
            load_fairseq_model = True
            ckpt = args.fairseq_checkpoint
            logging.info(f"Loading parameters from fairseq: {ckpt}")

            state_dict = torch.load(ckpt)
            if "cfg" in state_dict and state_dict["cfg"] is not None:
                model_cfg = state_dict["cfg"]["model"]
                logging.info(f"Building model from {model_cfg}")
            else:
                raise RuntimeError(f"Bad 'cfg' in state_dict of {ckpt}")

        # 1. frontend
        if args.write_collected_feats:
            # Extract features in the model
            # Note(jiatong): if we use write_collected_feats=True (we use
            #                pre-extracted feature for training): we still initial
            #                frontend to allow inference with raw speech signal
            #                but the frontend is not used in training
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            if args.input_size is None:
                input_size = frontend.output_size()
            else:
                input_size = args.input_size
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Segmenter
        if args.segmenter is not None:
            segmenter_class = segmenter_choices.get_class(args.segmenter)
            segmenter = segmenter_class(cfg=cfg, **args.segmenter_conf)
        else:
            segmenter = None

        # 3. Discriminator
        discriminator_class = discriminator_choices.get_class(args.discriminator)
        discriminator = discriminator_class(
            cfg=cfg, input_dim=vocab_size, **args.discriminator_conf
        )

        # 4. Generator
        generator_class = generator_choices.get_class(args.generator)
        generator = generator_class(
            cfg=cfg, input_dim=input_size, output_dim=vocab_size, **args.generator_conf
        )

        # 5. Loss definition
        losses = {}
        if getattr(args, "losses", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.losses:
                logging.info("initialize loss: {}".format(ctr["name"]))
                if ctr["name"] == "gradient_penalty":
                    loss = loss_choices.get_class(ctr["name"])(
                        discriminator=discriminator, **ctr["conf"]
                    )
                else:
                    loss = loss_choices.get_class(ctr["name"])(**ctr["conf"])
                losses[ctr["name"]] = loss

        # 6. Build model
        logging.info(f"kenlm_path is: {args.kenlm_path}")
        model = ESPnetUASRModel(
            cfg=cfg,
            frontend=frontend,
            segmenter=segmenter,
            discriminator=discriminator,
            generator=generator,
            losses=losses,
            kenlm_path=args.kenlm_path,
            token_list=args.token_list,
            max_epoch=args.max_epoch,
            vocab_size=vocab_size,
            use_collected_training_feats=args.write_collected_feats,
        )

        # FIXME(kamo): Should be done in model?
        # 7. Initialize
        if load_fairseq_model:
            logging.info(f"Initializing model from {ckpt}")
            model.load_state_dict(state_dict["model"], strict=False)
        else:
            if args.init is not None:
                initialize(model, args.init)

        assert check_return_type(model)
        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetUASRModel,
    ) -> List[torch.optim.Optimizer]:
        # check
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")

        generator_param_list = list(model.generator.parameters())
        discriminator_param_list = list(model.discriminator.parameters())

        # Add optional sets of model parameters
        if model.use_segmenter is not None:
            generator_param_list += list(model.segmenter.parameters())
        if (
            "pseudo_label_loss" in model.losses.keys()
            and model.losses["pseudo_label_loss"].weight > 0
        ):
            generator_param_list += list(
                model.losses["pseudo_label_loss"].decoder.parameters()
            )

        # define generator optimizer
        optim_generator_class = optim_classes.get(args.optim)
        if optim_generator_class is None:
            raise ValueError(
                f"must be one of {list(optim_classes)}: {args.optim_generator}"
            )
        optim_generator = optim_generator_class(
            generator_param_list,
            **args.optim_conf,
        )
        optimizers = [optim_generator]

        # define discriminator optimizer
        optim_discriminator_class = optim_classes.get(args.optim2)
        if optim_discriminator_class is None:
            raise ValueError(
                f"must be one of {list(optim_classes)}: {args.optim_discriminator}"
            )
        optim_discriminator = optim_discriminator_class(
            discriminator_param_list,
            **args.optim2_conf,
        )
        optimizers += [optim_discriminator]

        return optimizers
