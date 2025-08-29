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
from espnet2.spk.encoder.conformer_encoder import MfaConformerEncoder
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.spk.encoder.identity_encoder import IdentityEncoder
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.spk.encoder.ska_tdnn_encoder import SkaTdnnEncoder
from espnet2.spk.encoder.xvector_encoder import XvectorEncoder
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector
from espnet2.spk.projector.ska_tdnn_projector import SkaTdnnProjector
from espnet2.spk.projector.xvector_projector import XvectorProjector
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    LIDPreprocessor,
)
from espnet2.utils.types import int_or_none, str2bool, str_or_none

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


class LIDTask(AbsTask):
    num_optimizers: int = 1

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
            # validation or plot tsne
            retval = ("speech", "lid_labels")
        else:
            # inference
            retval = ("speech",)

        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()

        return retval
