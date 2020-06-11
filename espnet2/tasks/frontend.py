import argparse
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

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.enh_frontend import EnhFrontend
from espnet2.asr.frontend.espnet_model import ESPnetFrontendModel
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend),
    type_check=AbsFrontend,
    default="default",
)

normalize_choices = ClassChoices(
    "normalize",
    classes=dict(global_mvn=GlobalMVN, utterance_mvn=UtteranceMVN,),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=TransformerEncoder, vgg_rnn=VGGRNNEncoder, rnn=RNNEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(transformer=TransformerDecoder, rnn=RNNDecoder),
    type_check=AbsDecoder,
    default="rnn",
)


class FrontendTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1


    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")

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
            default=get_default_kwargs(ESPnetFrontendModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word"],
            help="The text will be tokenized " "in the specified level token",
        )


        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
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
    def required_data_names(cls, inference: bool = False) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_mix", "speech_ref1", "speech_ref2")
        else:
            # Recognition mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(cls, inference: bool = False) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetFrontendModel:
        assert check_argument_types()

        frontend = EnhFrontend()

        # 1. Build model
        model = ESPnetFrontendModel(
            frontend=frontend
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
