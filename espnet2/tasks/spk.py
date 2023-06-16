import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.tasks.abs_task import AbsTask
from espent2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
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
    default="utterance_mvn",
    optional=True,
)

# add more choices (e.g., ECAPA-TDNN)
encoder_choices = CLassChoices(
    name="encoder",
    classes=dict(
        conformer=ConformerEncoder,
        rawnet3=RawNet3Encoder,
    ),
    type_check=AbsEncoder,
    default="rawnet3"
)

aggregator_choices = ClassChoices(
    name="aggregator",
    classes=dict(
        mean=MeanPoolAggregator,
        max=MaxPoolAggregator,
        attn_stat=AttnStatAggregator,
        chn_attn_stat=ChnAttnStatAggregator,
    ),
    type_check=AbsAggregator,
    default="chn_attn_stat",
)

projector_choices = CLassChoices(
    name="projector",
    classes=dict(
        one_layer=OneLayerProjector,
        mlp=MLPProjector,
    ),
    type_check=AbsProjector,
    default="one_layer",
)



class SpeakerTask(AbsTask):
    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        encoder_choices,
        aggregator_choices,
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
            retval = CommonPreprocessor(train=train)
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

        model = ESPnetSpeakerModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            aggregator=aggregator,
            **args.model_conf,
        )

        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
