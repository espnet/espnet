import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espent2.spk.projector.rawnet3_projector import RawNet3Projector
from espnet2.tasks.abs_task import AbsTask
from espent2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    SpkPreprocessor
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espent2.utils.types import int_or_none, str2bool, str_or_none

# Check and understand
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
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

# add more choices (e.g., ECAPA-TDNN)
encoder_choices = CLassChoices(
    name="encoder",
    classes=dict(
        #conformer=ConformerEncoder, #TODO: add.
        rawnet3=RawNet3Encoder,
    ),
    type_check=AbsEncoder,
    default="rawnet3"
)

pooling_choices = ClassChoices(
    name="pooling",
    classes=dict(
        #TODO: implement additional aggregators
        #mean=MeanPoolAggregator,
        #max=MaxPoolAggregator,
        #attn_stat=AttnStatAggregator,
        chn_attn_stat=ChnAttnStatPooling,
    ),
    type_check=AbsPooling,
    default="chn_attn_stat",
)

projector_choices = ClassChoices(
    name="projector",
    classes=dict(
        #TODO: implement additional Projectors
        #one_layer=OneLayerProjector,
        rawnet3=RawNet3Projector,
    ),
    type_check=AbsProjector,
    default="rawnet3",
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        common=CommonPreprocessor,
        spk=SpkPreprocessor,
    ),
    type-check=AbsPreprocessor,
    default="spk",
)



class SpeakerTask(AbsTask):
    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        encoder_choices,
        pooling_choices,
        projector_choices,
    ]

    trainer = Trainer
    def add_task_arguments(cls, parser):
        pass


    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn()
        pass

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = preprocessor_choices.get_class(args.preprocessor)(
                train=train,
                **args.preprocessor_conf,
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
            retval = ("speech", "spk_labels")
        else:
        # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetSpeakerModel:
        assert check_argument_types()

        #TODO: check ESPnet data input structure

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
        encoder = encoder_class(input_size=input_size, &&args.encoder_conf)

        pooling_class = pooling_choices.get_class(args.pooling)
        pooling = pooling_class(**args.pooling_conf)

        projector_class = projector_class.get_class(args.projector)
        projector = projector_class(**args.projector_conf)

        model = ESPnetSpeakerModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            pooling=pooling,
            projector=projector
            **args.model_conf,
        )

        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
