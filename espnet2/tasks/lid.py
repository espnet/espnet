import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.asteroid_frontend import AsteroidFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.melspec_torch import MelSpectrogramTorch
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.lid.espnet_model import ESPnetLIDModel
from espnet2.spk.encoder.conformer_encoder import MfaConformerEncoder
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.spk.encoder.identity_encoder import IdentityEncoder
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.spk.encoder.ska_tdnn_encoder import SkaTdnnEncoder
from espnet2.spk.encoder.xvector_encoder import XvectorEncoder
from espnet2.spk.loss.aamsoftmax import AAMSoftmax
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)
from espnet2.spk.loss.abs_loss import AbsLoss
from espnet2.spk.loss.softmax import Softmax
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.pooling.mean_pooling import MeanPooling
from espnet2.spk.pooling.stat_pooling import StatsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector
from espnet2.spk.projector.ska_tdnn_projector import SkaTdnnProjector
from espnet2.spk.projector.xvector_projector import XvectorProjector
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.lid_trainer import LIDTrainer
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    LIDPreprocessor,
)
from espnet2.utils.types import int_or_none, str2bool, str_or_none

model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetLIDModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)

# Check and understand
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        asteroid_frontend=AsteroidFrontend,
        default=DefaultFrontend,
        fused=FusedFrontends,
        melspec_torch=MelSpectrogramTorch,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default=None,
    optional=True,
)

specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)

normalize_choices = ClassChoices(
    name="normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(
        ecapa_tdnn=EcapaTdnnEncoder,
        identity=IdentityEncoder,
        mfaconformer=MfaConformerEncoder,
        rawnet3=RawNet3Encoder,
        ska_tdnn=SkaTdnnEncoder,
        xvector=XvectorEncoder,
    ),
    type_check=AbsEncoder,
    default="rawnet3",
)

pooling_choices = ClassChoices(
    name="pooling",
    classes=dict(
        chn_attn_stat=ChnAttnStatPooling,
        mean=MeanPooling,
        stats=StatsPooling,
    ),
    type_check=AbsPooling,
    default="chn_attn_stat",
)

projector_choices = ClassChoices(
    name="projector",
    classes=dict(
        rawnet3=RawNet3Projector,
        ska_tdnn=SkaTdnnProjector,
        xvector=XvectorProjector,
    ),
    type_check=AbsProjector,
    default="rawnet3",
)

encoder_condition_choices = ClassChoices(
    name="encoder_condition",
    classes=dict(
        ecapa_tdnn=EcapaTdnnEncoder,
        identity=IdentityEncoder,
        mfaconformer=MfaConformerEncoder,
        rawnet3=RawNet3Encoder,
        ska_tdnn=SkaTdnnEncoder,
        xvector=XvectorEncoder,
    ),
    type_check=AbsEncoder,
    default="rawnet3",
)

pooling_condition_choices = ClassChoices(
    name="pooling_condition",
    classes=dict(
        chn_attn_stat=ChnAttnStatPooling,
        mean=MeanPooling,
        stats=StatsPooling,
    ),
    type_check=AbsPooling,
    default="chn_attn_stat",
)

projector_condition_choices = ClassChoices(
    name="projector_condition",
    classes=dict(
        rawnet3=RawNet3Projector,
        ska_tdnn=SkaTdnnProjector,
        xvector=XvectorProjector,
    ),
    type_check=AbsProjector,
    default="rawnet3",
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        common=CommonPreprocessor,
        lid=LIDPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="lid",
)

loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        aamsoftmax=AAMSoftmax,
        aamsoftmax_sc_topk=ArcMarginProduct_intertopk_subcenter,
        softmax=Softmax,
    ),
    type_check=AbsLoss,
    default="aamsoftmax",
)


class LIDTask(AbsTask):
    num_optimizers: int = 1

    trainer = LIDTrainer

    class_choices_list = [
        model_choices,
        frontend_choices,
        specaug_choices,
        normalize_choices,
        encoder_choices,
        pooling_choices,
        projector_choices,
        encoder_condition_choices,
        pooling_condition_choices,
        projector_condition_choices,
        preprocessor_choices,
        loss_choices,
    ]

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

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
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--target_duration",
            type=float,
            default=3.0,
            help="Duration (in seconds) of samples in a minibatch",
        )

        group.add_argument(
            "--lang2utt",
            type=str,
            default="",
            help="Directory of the train lang2utt file to be used in label mapping"
            "Note that both train and validation use the same lang2utt file, since"
            "we can only support the same categories during validation",
        )

        group.add_argument(
            "--lang_num",
            type=int,
            default=None,
            help="specify the number of languages during training",
        )

        group.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            help="Sampling rate",
        )

        group.add_argument(
            "--num_eval",
            type=int,
            default=10,
            help="Number of segments to make from one utterance in the "
            "inference phase",
        )

        group.add_argument(
            "--rir_scp",
            type=str,
            default="",
            help="Directory of the rir data to be augmented",
        )

        for class_choices in cls.class_choices_list:
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return CommonCollateFn(
            not_sequence=["lid_labels"],
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            retval = preprocessor_choices.get_class(args.preprocessor)(
                lang2utt=args.lang2utt,
                train=train,
                **args.preprocessor_conf,
            )

        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if train:
            # train
            retval = ("speech", "lid_labels")
        elif not train and not inference:
            # validation or plot tsne or collect statistics
            retval = ("speech",)
        else:
            # inference
            retval = ("speech",)

        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not train and not inference:
            # validation or plot tsne
            # not required for collect statistics
            retval = ("lid_labels",)
        else:
            # train or inference
            retval = ()

        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetLIDModel:

        if args.frontend is not None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()

        else:
            # Give features from data-loader (e.g., precompute features).
            frontend = None
            input_size = args.input_size

        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size()

        pooling_class = pooling_choices.get_class(args.pooling)
        pooling = pooling_class(input_size=encoder_output_size, **args.pooling_conf)
        pooling_output_size = pooling.output_size()

        projector_class = projector_choices.get_class(args.projector)
        projector = projector_class(
            input_size=pooling_output_size, **args.projector_conf
        )
        projector_output_size = projector.output_size()

        loss_class = loss_choices.get_class(args.loss)
        loss = loss_class(
            nout=projector_output_size, nclasses=args.lang_num, **args.loss_conf
        )

        model_arg = getattr(args, "model", None)
        if model_arg is None:
            model_arg = "espnet"
        model_class = model_choices.get_class(model_arg)

        model = model_class(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            pooling=pooling,
            projector=projector,
            loss=loss,
            **args.model_conf,
        )

        if args.init is not None:
            initialize(model, args.init)

        return model
