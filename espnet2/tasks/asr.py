import argparse
import logging
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import configargparse
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.e2e import ASRE2E
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.abs_task import BaseChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.initialize import initialize
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


class FrontendChoices(BaseChoices):
    def __init__(self):
        super().__init__(
            name="frontend",
            classes=dict(default=DefaultFrontend)
        )


class NormalizeChoices(BaseChoices):
    def __init__(self):
        super().__init__(
            "normalize",
            classes=dict(
                global_mvn=GlobalMVN,
                utterance_mvn=UtteranceMVN,
            ),
            default="utterance_mvn",
            optional=True
        )


class EncoderChoices(BaseChoices):
    def __init__(self):
        super().__init__(
            "encoder",
            classes=dict(
                transformer=TransformerEncoder,
                vgg_rnn=VGGRNNEncoder,
                rnn=RNNEncoder,
            ),
            default="rnn"
        )


class DecoderChoices(BaseChoices):
    def __init__(self):
        super().__init__(
            "decoder",
            classes=dict(
                transformer=TransformerDecoder,
                rnn=RNNDecoder
            ),
            default="rnn"
        )


frontend_choices = FrontendChoices()
normalize_choices = NormalizeChoices()
encoder_choices = EncoderChoices()
decoder_choices = DecoderChoices()


class ASRTask(AbsTask):
    class_choices_list = (
        AbsTask.class_choices_list +
        [frontend_choices,
         normalize_choices,
         encoder_choices,
         decoder_choices]
    )

    @classmethod
    def add_arguments(
            cls, parser: configargparse.ArgumentParser = None
    ) -> configargparse.ArgumentParser:
        assert check_argument_types()
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        if parser is None:
            parser = configargparse.ArgumentParser(
                description="Train ASR",
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            )

        AbsTask.add_arguments(parser)
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
            choices=cls.init_choices(),
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--e2e_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for E2E class.",
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
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        return parser

    @classmethod
    def get_task_config(cls) -> Dict[str, Any]:
        # This method is used only for --print_config
        ctc_conf = get_default_kwargs(CTC)
        e2e_conf = get_default_kwargs(ASRE2E)
        config = dict(
            ctc_conf=ctc_conf,
            e2e_conf=e2e_conf,
        )
        assert check_return_type(config)
        return config

    @classmethod
    def init_choices(cls) -> Tuple[Optional[str], ...]:
        choices = (
            "chainer",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            None,
        )
        return choices

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                  Tuple[List[str], Dict[str, torch.Tensor]]]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

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
                bpemodel=args.bpemodel)
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        if train:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ASRE2E:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list) as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 3. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 4. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # 4. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_sizse=encoder.output_size(), **args.ctc_conf
        )

        # 5. RNN-T Decoder (Not implemented)
        rnnt_decoder = None

        # 6. Build model
        model = ASRE2E(
            vocab_size=vocab_size,
            frontend=frontend,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=rnnt_decoder,
            token_list=token_list,
            **args.e2e_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 7. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
