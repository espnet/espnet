import argparse
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
)
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.cnn import CNNFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.ser.espnet_model import ESPnetSERModel
from espnet2.ser.loss.cross_entropy_loss import Xnt
from espnet2.ser.pooling.abs_pooling import AbsPooling
from espnet2.ser.pooling.mean_pooling import MeanPooling
from espnet2.ser.projector.abs_projector import AbsProjector
from espnet2.ser.projector.linear_projector import LinearProjector
from espnet2.slu.postencoder.conformer_postencoder import ConformerPostEncoder
from espnet2.slu.postencoder.transformer_postencoder import TransformerPostEncoder
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.preprocessor import SERPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        cnn=CNNFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetSERModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
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
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        longformer=LongformerEncoder,
        branchformer=BranchformerEncoder,
    ),
    type_check=AbsEncoder,
    default="None",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
        conformer=ConformerPostEncoder,
        transformer=TransformerPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
pooling_choices = ClassChoices(
    name="pooling",
    classes=dict(
        mean=MeanPooling,
    ),
    type_check=AbsPooling,
    default="mean",
)

projector_choices = ClassChoices(
    name="projector",
    classes=dict(
        linear=LinearProjector,
    ),
    type_check=AbsProjector,
    default="linear",
)
loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        cross_entropy=Xnt,
    ),
    default="cross_entropy",
)


class SERTask(ASRTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --model and --model_conf
        model_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        pooling_choices,
        projector_choices,
        loss_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        group.add_argument(
            "--pre_postencoder_norm",
            type=str2bool,
            default=False,
            help="pre_postencoder_norm",
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
            "--emotions",
            type=str,
            default="A S H U F D C N O X",
            help="Space-separated list of emotion labels to use",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            retval = SERPreprocessor(
                train=train,
                emotions=getattr(args, "emotions", None),
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=getattr(args, "rir_scp", None),
                rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
                noise_scp=getattr(args, "noise_scp", None),
                noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
                noise_db_range=getattr(args, "noise_db_range", "13_15"),
                short_noise_thres=getattr(args, "short_noise_thres", 0.5),
                speech_volume_normalize=getattr(args, "rir_scp", None),
                **getattr(args, "preprocessor_conf", {}),
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "emo")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("transcript",)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSERModel:

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

        # 3. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        if getattr(args, "encoder", None) is not None:
            encoder_class = encoder_choices.get_class(args.encoder)
            encoder = encoder_class(input_size=input_size, **args.encoder_conf)

            # 5. Post-encoder block
            # NOTE(kan-bayashi): Use getattr to keep the compatibility
            encoder_output_size = encoder.output_size()
            if getattr(args, "postencoder", None) is not None:
                postencoder_class = postencoder_choices.get_class(args.postencoder)
                postencoder = postencoder_class(
                    input_size=encoder_output_size, **args.postencoder_conf
                )
                encoder_output_size = postencoder.output_size()
            else:
                postencoder = None

        # 6. Pooling
        pooling_class = pooling_choices.get_class(args.pooling)
        if preencoder is not None:
            pooling = pooling_class(
                input_size=preencoder.output_size(), **args.pooling_conf
            )
            pooling_output_size = pooling.output_size()
        # 7. Projector
        projector_class = projector_choices.get_class(args.projector)
        projector = projector_class(
            input_size=pooling_output_size, **args.projector_conf
        )
        projector_output_size = projector.output_size()

        loss_class = loss_choices.get_class(args.loss)
        loss = loss_class(
            nout=projector_output_size, nclasses=projector_output_size, **args.loss_conf
        )

        # 8. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        model = model_class(
            frontend=frontend,
            specaug=specaug,
            preencoder=preencoder,
            pooling=pooling,
            projector=projector,
            loss=loss,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
