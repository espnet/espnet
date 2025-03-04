import argparse
import logging
from inspect import signature
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.beats_encoder import BeatsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.hugging_face_transformers_encoder import (
    HuggingFaceTransformersEncoder,
)
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.cls.decoder.abs_decoder import AbsDecoder
from espnet2.cls.decoder.linear_decoder import LinearDecoder
from espnet2.cls.espnet_model import ESPnetClassificationModel
from espnet2.cls.layers.sequence_embedding_fusion import (
    AbsEmbeddingFusion,
    AudioTextAttnFusion,
    AudioTextConcat,
)
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    CommonPreprocessor,
    MutliTokenizerCommonPreprocessor,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

logger = logging.getLogger("cls")

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
specaug_choices = ClassChoices(
    name="specaug",
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
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        beats=BeatsEncoder,
        transformer=TransformerEncoder,
        conformer=ConformerEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)
text_encoder_choices = ClassChoices(
    "text_encoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersEncoder,
        # clap=CLAPTextEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
    optional=True,
)
embedding_fusion_choices = ClassChoices(
    "embedding_fusion",
    classes=dict(
        attention=AudioTextAttnFusion,
        concatenate=AudioTextConcat,
    ),
    type_check=AbsEmbeddingFusion,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        linear=LinearDecoder,
    ),
    type_check=AbsDecoder,
    default="linear",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetClassificationModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)


class CLSTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --text_encoder and --text_encoder_conf
        text_encoder_choices,
        # --embedding_fusion and --embedding_fusion_conf
        embedding_fusion_choices,
        # --decoder and --decoder_conf
        decoder_choices,
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
            "--text_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--text_bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
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
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        # NOTE(shikhar): Please ensure pad value is -1 for integer values.
        # This is important for generating one-hot vectors for
        # multi-label classification.
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            # valid_args = signature(CommonPreprocessor.__init__).parameters
            # filtered_args = {k: v for k, v in vars(args).items() if k in valid_args}
            # token_type
            # retval = CommonPreprocessor(train=train, text_name="label", **filtered_args)
            retval = MutliTokenizerCommonPreprocessor(
                train=train,
                text_name=["label", "text"],
                token_type=["word", "hugging_face" if args.text_token_list else None],
                token_list=[args.token_list, args.text_token_list],
                bpemodel=[None, args.text_bpemodel],
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "label")
        else:
            # Inference mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("speech_lengths", "label_lengths", "text", "text_lengths")
        return retval

    @classmethod
    def load_token_list(self, token_list):
        """Load token list from a file or validate list input."""
        if isinstance(token_list, str):
            with open(token_list, encoding="utf-8") as f:
                return [line.rstrip() for line in f]
        elif isinstance(token_list, (tuple, list)):
            return list(token_list)
        else:
            raise RuntimeError(
                "Token list must be a str or a list, recheck all token lists."
            )

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetClassificationModel:
        args.token_list = cls.load_token_list(args.token_list)
        if args.text_token_list is not None:
            args.text_token_list = cls.load_token_list(args.text_token_list)

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
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

        # 4. Pre-encoder input block
        if args.preencoder is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Optional text encoder and embedding fusion
        text_encoder = None
        embedding_fusion = None
        if args.text_encoder is not None:
            text_encoder_class = text_encoder_choices.get_class(args.text_encoder)
            text_encoder = text_encoder_class(**args.text_encoder_conf)
            embedding_fusion_class = embedding_fusion_choices.get_class(
                args.embedding_fusion
            )
            assert (
                embedding_fusion_class is not None
            ), "embedding_fusion is required when text_encoder is used"
            embedding_fusion = embedding_fusion_class(**args.embedding_fusion_conf)

        # 6. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        n_classes = len(args.token_list) - 2  # -2 for <unk>, <blank>

        encoder_output_size = (
            encoder.output_size()
            if embedding_fusion is None
            else embedding_fusion.output_size()
        )
        decoder = decoder_class(
            n_classes,
            encoder_output_size=encoder_output_size,
            **args.decoder_conf,
        )

        # 7. Build model
        model = ESPnetClassificationModel(
            vocab_size=n_classes,
            token_list=args.token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            decoder=decoder,
            text_encoder=text_encoder,
            embedding_fusion=embedding_fusion,
            **args.model_conf,
        )

        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
