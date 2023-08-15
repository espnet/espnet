"""ASR Transducer Task."""

import argparse
import logging
import os
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.decoder.mega_decoder import MEGADecoder
from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.rwkv_decoder import RWKVDecoder
from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.espnet_transducer_model import ESPnetASRTransducerModel
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    "specaug",
    classes=dict(
        specaug=SpecAug,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        mega=MEGADecoder,
        rnn=RNNDecoder,
        rwkv=RWKVDecoder,
        stateless=StatelessDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)


class ASRTransducerTask(AbsTask):
    """ASR Transducer Task definition."""

    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        decoder_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """Add Transducer task arguments.

        Args:
            cls: ASRTransducerTask object.
            parser: Transducer arguments parser.

        """
        group = parser.add_argument_group(description="Task related.")

        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="Integer-string mapper for tokens.",
        )
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of dimensions for input features.",
        )
        group.add_argument(
            "--init",
            type=str_or_none,
            default=None,
            help="Type of model initialization to use.",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRTransducerModel),
            help="The keyword arguments for the model class.",
        )
        group.add_argument(
            "--encoder_conf",
            action=NestedDictAction,
            default={},
            help="The keyword arguments for the encoder class.",
        )
        group.add_argument(
            "--joint_network_conf",
            action=NestedDictAction,
            default={},
            help="The keyword arguments for the joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related.")

        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Whether to apply preprocessing to input data.",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The type of tokens to use during tokenization.",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The path of the sentencepiece model.",
        )
        group.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="The 'non_linguistic_symbols' file path.",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Text cleaner to use.",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="g2p method to use if --token_type=phn.",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Normalization value for maximum amplitude scaling.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The RIR SCP file path.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="The probability of the applied RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The path of noise SCP file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability of the applied noise addition.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of the noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --decoder and --decoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        """Build collate function.

        Args:
            cls: ASRTransducerTask object.
            args: Task arguments.
            train: Training mode.

        Return:
            : Callable collate function.

        """
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """Build pre-processing function.

        Args:
            cls: ASRTransducerTask object.
            args: Task arguments.
            train: Training mode.

        Return:
            : Callable pre-processing function.

        """
        assert check_argument_types()

        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
            )
        else:
            retval = None

        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Required data depending on task mode.

        Args:
            cls: ASRTransducerTask object.
            train: Training mode.
            inference: Inference mode.

        Return:
            retval: Required task data.

        """
        if not inference:
            retval = ("speech", "text")
        else:
            retval = ("speech",)

        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Optional data depending on task mode.

        Args:
            cls: ASRTransducerTask object.
            train: Training mode.
            inference: Inference mode.

        Return:
            retval: Optional task data.

        """
        retval = ()
        assert check_return_type(retval)

        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRTransducerModel:
        """Required data depending on task mode.

        Args:
            cls: ASRTransducerTask object.
            args: Task arguments.

        Return:
            model: ASR Transducer model.

        """
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

        if hasattr(args, "scheduler_conf"):
            args.model_conf["warmup_steps"] = args.scheduler_conf.get(
                "warmup_steps", 25000
            )

        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Encoder
        encoder = Encoder(input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size,
            **args.decoder_conf,
        )
        decoder_output_size = decoder.output_size

        # 6. Joint Network
        joint_network = JointNetwork(
            vocab_size,
            encoder_output_size,
            decoder_output_size,
            **args.joint_network_conf,
        )

        # 7. Build model
        model = ESPnetASRTransducerModel(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            joint_network=joint_network,
            **args.model_conf,
        )

        # 8. Initialize model
        if args.init is not None:
            raise NotImplementedError(
                "Currently not supported.",
                "Initialization part will be reworked in a short future.",
            )

        assert check_return_type(model)

        return model
