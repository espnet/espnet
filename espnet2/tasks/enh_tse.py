import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.enh.espnet_model_tse import ESPnetExtractionModel
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.extractor.td_speakerbeam_extractor import TDSpeakerBeamExtractor
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.enh import (
    criterion_choices,
    decoder_choices,
    encoder_choices,
    loss_wrapper_choices,
)
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import AbsPreprocessor, TSEPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

extractor_choices = ClassChoices(
    name="extractor",
    classes=dict(
        td_speakerbeam=TDSpeakerBeamExtractor,
    ),
    type_check=AbsExtractor,
    default="td_speakerbeam",
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        tse=TSEPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="tse",
)

MAX_REFERENCE_NUM = 100


class TargetSpeakerExtractionTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --encoder and --encoder_conf
        encoder_choices,
        # --extractor and --extractor_conf
        extractor_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
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
            default=get_default_kwargs(ESPnetExtractionModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--train_spk2enroll",
            type=str_or_none,
            default=None,
            help="The scp file containing the mapping from speakerID to enrollment\n"
            "(This is used to sample the target-speaker enrollment signal)",
        )
        group.add_argument(
            "--enroll_segment",
            type=int_or_none,
            default=None,
            help="Truncate the enrollment audio to the specified length if not None",
        )
        group.add_argument(
            "--load_spk_embedding",
            type=str2bool,
            default=False,
            help="Whether to load speaker embeddings instead of enrollments",
        )
        group.add_argument(
            "--load_all_speakers",
            type=str2bool,
            default=False,
            help="Whether to load target-speaker for all speakers in each sample",
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
        retval = TSEPreprocessor(
            train=train,
            train_spk2enroll=args.train_spk2enroll,
            enroll_segment=getattr(args, "enroll_segment", None),
            load_spk_embedding=getattr(args, "load_spk_embedding", False),
            load_all_speakers=getattr(args, "load_all_speakers", False),
        )
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_mix", "enroll_ref1", "speech_ref1")
        else:
            # Inference mode
            retval = ("speech_mix", "enroll_ref1")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["enroll_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        if "speech_ref1" in retval:
            retval += [
                "speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)
            ]
        else:
            retval += [
                "speech_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)
            ]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetExtractionModel:
        assert check_argument_types()

        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        extractor = extractor_choices.get_class(args.extractor)(
            encoder.output_dim, **args.extractor_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        loss_wrappers = []

        if getattr(args, "criterions", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.criterions:
                criterion_conf = ctr.get("conf", {})
                criterion = criterion_choices.get_class(ctr["name"])(**criterion_conf)
                loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                    criterion=criterion, **ctr["wrapper_conf"]
                )
                loss_wrappers.append(loss_wrapper)

        # 1. Build model
        model = ESPnetExtractionModel(
            encoder=encoder,
            extractor=extractor,
            decoder=decoder,
            loss_wrappers=loss_wrappers,
            **args.model_conf
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
