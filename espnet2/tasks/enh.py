import argparse
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
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter
from espnet2.enh.separator.conformer_separator import ConformerSeparator
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.transformer_separator import TransformerSeparator
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(stft=STFTEncoder, conv=ConvEncoder, same=NullEncoder),
    type_check=AbsEncoder,
    default="stft",
)

separator_choices = ClassChoices(
    name="separator",
    classes=dict(
        rnn=RNNSeparator,
        tcn=TCNSeparator,
        dprnn=DPRNNSeparator,
        transformer=TransformerSeparator,
        conformer=ConformerSeparator,
        wpe_beamformer=NeuralBeamformer,
        asteroid=AsteroidModel_Converter,
    ),
    type_check=AbsSeparator,
    default="rnn",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(stft=STFTDecoder, conv=ConvDecoder, same=NullDecoder),
    type_check=AbsDecoder,
    default="stft",
)

MAX_REFERENCE_NUM = 100


class EnhancementTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --encoder and --encoder_conf
        encoder_choices,
        # --separator and --separator_conf
        separator_choices,
        # --decoder and --decoder_conf
        decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

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
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_mix", "speech_ref1")
        else:
            # Recognition mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementModel:
        assert check_argument_types()

        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **args.separator_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        # 1. Build model
        model = ESPnetEnhancementModel(
            encoder=encoder, separator=separator, decoder=decoder, **args.model_conf
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
