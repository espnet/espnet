import argparse
from dataclasses import dataclass
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.null_decoder import NullDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.null_encoder import NullEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.cdiffuse_espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import CISDRLoss
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.criterions.time_domain import SNRLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter
from espnet2.enh.separator.conformer_separator import ConformerSeparator
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.transformer_separator import TransformerSeparator
from espnet2.iterators.abs_iter_factory import AbsIterFactory
# from espnet2.iterators.chunk_multiple_seq_iter_factory import ChunkIterFactory
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.enh import encoder_choices
from espnet2.tasks.enh import separator_choices
from espnet2.tasks.enh import decoder_choices
from espnet2.tasks.enh import loss_wrapper_choices
from espnet2.tasks.enh import criterion_choices
from espnet2.tasks.enh import MAX_REFERENCE_NUM
from espnet2.tasks.enh import EnhancementTask as Orig_EnhancementTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none







class EnhancementTask(Orig_EnhancementTask, AbsTask):

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

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
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)


    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementModel:
        assert check_argument_types()

        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **args.separator_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        loss_wrappers = []
        for ctr in args.criterions:
            criterion = criterion_choices.get_class(ctr["name"])(**ctr["conf"])
            loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                criterion=criterion, **ctr["wrapper_conf"]
            )
            loss_wrappers.append(loss_wrapper)

        # 1. Build model
        model = ESPnetEnhancementModel(
            encoder=encoder,
            separator=separator,
            decoder=decoder,
            loss_wrappers=loss_wrappers,
            **args.model_conf
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
