import argparse
import copy
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
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.diar.espnet_model import ESPnetDiarModel
from espnet2.diar.layers.abs_mask import AbsMask
from espnet2.diar.layers.multi_mask import MultiMask
from espnet2.diar.loss.wrappers.pit_solver_2 import PITSolver2
from espnet2.diar.separator.abs_separator import AbsSeparator
from espnet2.diar.separator.tcn_separator_nomask import TCNSeparator
from espnet2.enh.espnet_enh_s2t_model import ESPnetEnhS2TModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr import decoder_choices as asr_decoder_choices_
from espnet2.tasks.asr import encoder_choices as asr_encoder_choices_
from espnet2.tasks.asr import frontend_choices
from espnet2.tasks.asr import normalize_choices
from espnet2.tasks.asr import postencoder_choices as asr_postencoder_choices_
from espnet2.tasks.asr import preencoder_choices as asr_preencoder_choices_
from espnet2.tasks.asr import specaug_choices
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.diar import attractor_choices
from espnet2.tasks.diar import decoder_choices as diar_decoder_choices_
from espnet2.tasks.diar import encoder_choices as diar_encoder_choices_
from espnet2.tasks.diar import label_aggregator_choices
from espnet2.tasks.enh import decoder_choices as enh_decoder_choices_
from espnet2.tasks.enh import encoder_choices as enh_encoder_choices_
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh import separator_choices as enh_separator_choices_
from espnet2.tasks.st import decoder_choices as st_decoder_choices_
from espnet2.tasks.st import encoder_choices as st_encoder_choices_
from espnet2.tasks.st import extra_asr_decoder_choices as st_extra_asr_decoder_choices_
from espnet2.tasks.st import extra_mt_decoder_choices as st_extra_mt_decoder_choices_
from espnet2.tasks.st import postencoder_choices as st_postencoder_choices_
from espnet2.tasks.st import preencoder_choices as st_preencoder_choices_
from espnet2.tasks.st import STTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.preprocessor import CommonPreprocessor_multi
from espnet2.train.preprocessor import MutliTokenizerCommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


# Enhancement
enh_encoder_choices = copy.deepcopy(enh_encoder_choices_)
enh_encoder_choices.name = "enh_encoder"
enh_decoder_choices = copy.deepcopy(enh_decoder_choices_)
enh_decoder_choices.name = "enh_decoder"
enh_separator_choices = copy.deepcopy(enh_separator_choices_)
enh_separator_choices.name = "enh_separator"

# ASR (also SLU)
asr_preencoder_choices = copy.deepcopy(asr_preencoder_choices_)
asr_preencoder_choices.name = "asr_preencoder"
asr_encoder_choices = copy.deepcopy(asr_encoder_choices_)
asr_encoder_choices.name = "asr_encoder"
asr_postencoder_choices = copy.deepcopy(asr_postencoder_choices_)
asr_postencoder_choices.name = "asr_postencoder"
asr_decoder_choices = copy.deepcopy(asr_decoder_choices_)
asr_decoder_choices.name = "asr_decoder"

# ST
st_preencoder_choices = copy.deepcopy(st_preencoder_choices_)
st_preencoder_choices.name = "st_preencoder"
st_encoder_choices = copy.deepcopy(st_encoder_choices_)
st_encoder_choices.name = "st_encoder"
st_postencoder_choices = copy.deepcopy(st_postencoder_choices_)
st_postencoder_choices.name = "st_postencoder"
st_decoder_choices = copy.deepcopy(st_decoder_choices_)
st_decoder_choices.name = "st_decoder"
st_extra_asr_decoder_choices = copy.deepcopy(st_extra_asr_decoder_choices_)
st_extra_asr_decoder_choices.name = "st_extra_asr_decoder"
st_extra_mt_decoder_choices = copy.deepcopy(st_extra_mt_decoder_choices_)
st_extra_mt_decoder_choices.name = "st_extra_mt_decoder"

# Diarization
diar_frontend_choices = ClassChoices(
    name="diar_frontend",
    classes=dict(
        default=DefaultFrontend,
    ),
    type_check=AbsFrontend,
    default=None,
    optional=True,
)
diar_specaug_choices = copy.deepcopy(specaug_choices)
diar_specaug_choices.name = "diar_specaug"
diar_normalize_choices = copy.deepcopy(normalize_choices)
diar_normalize_choices.name = "diar_normalize"
diar_encoder_choices = copy.deepcopy(diar_encoder_choices_)
diar_encoder_choices.name = "diar_encoder"
diar_decoder_choices = copy.deepcopy(diar_decoder_choices_)
diar_decoder_choices.name = "diar_decoder"

# Separation (using "sep" to differenciate from "enh")
sep_encoder_choices = copy.deepcopy(enh_encoder_choices_)
sep_encoder_choices.name = "sep_encoder"
sep_decoder_choices = copy.deepcopy(enh_decoder_choices_)
sep_decoder_choices.name = "sep_decoder"
sep_separator_choices = ClassChoices(
    name="sep_separator",
    classes=dict(
        tcn=TCNSeparator,
    ),
    type_check=AbsSeparator,
    default="tcn",
)
sep_mask_module_choices = ClassChoices(
    name="sep_mask_module",
    classes=dict(multi_mask=MultiMask),
    type_check=AbsMask,
    default="multi_mask",
)
sep_loss_wrapper_choices = ClassChoices(
    name="sep_loss_wrappers",
    classes=dict(pit=PITSolver, fixed_order=FixedOrderSolver, pit2=PITSolver2),
    type_check=AbsLossWrapper,
    default=None,
)

MAX_REFERENCE_NUM = 100

name2task = dict(
    enh=EnhancementTask,
    asr=ASRTask,
    st=STTask,
)

# More can be added to the following attributes
enh_attributes = [
    "encoder",
    "encoder_conf",
    "separator",
    "separator_conf",
    "decoder",
    "decoder_conf",
    "criterions",
]

asr_attributes = [
    "token_list",
    "input_size",
    "frontend",
    "frontend_conf",
    "specaug",
    "specaug_conf",
    "normalize",
    "normalize_conf",
    "preencoder",
    "preencoder_conf",
    "encoder",
    "encoder_conf",
    "postencoder",
    "postencoder_conf",
    "decoder",
    "decoder_conf",
    "ctc_conf",
]

st_attributes = [
    "token_list",
    "src_token_list",
    "input_size",
    "frontend",
    "frontend_conf",
    "specaug",
    "specaug_conf",
    "normalize",
    "normalize_conf",
    "preencoder",
    "preencoder_conf",
    "encoder",
    "encoder_conf",
    "postencoder",
    "postencoder_conf",
    "decoder",
    "decoder_conf",
    "ctc_conf",
    "extra_asr_decoder",
    "extra_asr_decoder_conf",
    "extra_mt_decoder",
    "extra_mt_decoder_conf",
]

diar_attributes = [
    "num_spk",
    "input_size",
    "frontend",
    "frontend_conf",
    "specaug",
    "specaug_conf",
    "normalize",
    "normalize_conf",
    "label_aggregator",
    "label_aggregator_choices",
    "encoder",
    "encoder_conf",
    "decoder",
    "decoder_conf",
    "attractor",
    "attractor_conf",
]

sep_attributes = [
    "encoder",
    "encoder_conf",
    "separator",
    "separator_conf",
    "mask_module",
    "mask_module_conf",
    "decoder",
    "decoder_conf",
    "criterions",
]


class EnhS2TTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --enh_encoder and --enh_encoder_conf
        enh_encoder_choices,
        # --enh_separator and --enh_separator_conf
        enh_separator_choices,
        # --enh_decoder and --enh_decoder_conf
        enh_decoder_choices,
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --asr_preencoder and --asr_preencoder_conf
        asr_preencoder_choices,
        # --asr_encoder and --asr_encoder_conf
        asr_encoder_choices,
        # --asr_postencoder and --asr_postencoder_conf
        asr_postencoder_choices,
        # --asr_decoder and --asr_decoder_conf
        asr_decoder_choices,
        # --st_preencoder and --st_preencoder_conf
        st_preencoder_choices,
        # --st_encoder and --st_encoder_conf
        st_encoder_choices,
        # --st_postencoder and --st_postencoder_conf
        st_postencoder_choices,
        # --st_decoder and --st_decoder_conf
        st_decoder_choices,
        # --st_extra_asr_decoder and --st_extra_asr_decoder_conf
        st_extra_asr_decoder_choices,
        # --st_extra_mt_decoder and --st_extra_mt_decoder_conf
        st_extra_mt_decoder_choices,
        # --diar_frontend and --diar_frontend_conf
        diar_frontend_choices,
        # --diar_specaug and --diar_specaug_conf
        diar_specaug_choices,
        # --diar_normalize and --diar_normalize_conf
        diar_normalize_choices,
        # --diar_encoder and --diar_encoder_conf
        diar_encoder_choices,
        # --diar_decoder and --diar_decoder_conf
        diar_decoder_choices,
        # --label_aggregator and --label_aggregator_conf
        label_aggregator_choices,
        # --attractor and --attractor_conf
        attractor_choices,
        # --sep_encoder and --sep_encoder_conf
        sep_encoder_choices,
        # --sep_separator and --sep_separator_conf
        sep_separator_choices,
        # --sep_mask_module and --sep_mask_module_conf
        sep_mask_module_choices,
        # --sep_decoder and --sep_decoder_conf
        sep_decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--src_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for source language)",
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

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )

        group.add_argument(
            "--enh_criterions",
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

        group.add_argument(
            "--sep_criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the sep loss wrappers.",
        )

        group.add_argument(
            "--num_spk",
            type=int_or_none,
            default=None,
            help="The number of speakers (for each recording) used in diar and sep",
        )

        group.add_argument(
            "--enh_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for enh submodel class.",
        )

        group.add_argument(
            "--asr_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRModel),
            help="The keyword arguments for asr submodel class.",
        )

        group.add_argument(
            "--st_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for st submodel class.",
        )

        group.add_argument(
            "--sep_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for sep submodel class.",
        )

        group.add_argument(
            "--diar_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetDiarModel),
            help="The keyword arguments for diar submodel class.",
        )

        group.add_argument(
            "--subtask_series",
            type=str,
            nargs="+",
            default=("enh", "asr"),
            choices=["enh", "asr", "st", "sep", "diar"],
            help="The series of subtasks in the pipeline.",
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhS2TModel),
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
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        group.add_argument(
            "--src_token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--src_bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for source language)",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
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
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            if "st" in args.subtask_series:
                retval = MutliTokenizerCommonPreprocessor(
                    train=train,
                    token_type=[args.token_type, args.src_token_type],
                    token_list=[args.token_list, args.src_token_list],
                    bpemodel=[args.bpemodel, args.src_bpemodel],
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                    # NOTE(kamo): Check attribute existence for backward compatibility
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
                    if hasattr(args, "speech_volume_normalize")
                    else None,
                    speech_name="speech",
                    text_name=["text", "src_text"],
                )
            elif "diar" in args.subtask_series:
                retval = CommonPreprocessor(train=train)
            else:
                retval = CommonPreprocessor_multi(
                    train=train,
                    token_type=args.token_type,
                    token_list=args.token_list,
                    bpemodel=args.bpemodel,
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_name=["text"],
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
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
            retval = ("speech", "speech_ref1")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["text"]  # for enh_{asr,st}
        retval += ["spk_labels"]  # for sep_diar
        retval += ["dereverb_ref1"]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["src_text"]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhS2TModel:
        assert check_argument_types()

        # Build submodels in the order of subtask_series
        model_conf = args.model_conf.copy()
        for _, subtask in enumerate(args.subtask_series):
            subtask_conf = dict(
                init=None, model_conf=eval(f"args.{subtask}_model_conf")
            )

            for attr in eval(f"{subtask}_attributes"):
                subtask_conf[attr] = (
                    getattr(args, subtask + "_" + attr, None)
                    if getattr(args, subtask + "_" + attr, None) is not None
                    else getattr(args, attr, None)
                )

            if subtask in ["asr", "st"]:
                m_subtask = "s2t"
            elif subtask in ["enh", "sep", "diar"]:
                m_subtask = subtask
            else:
                raise ValueError(f"{subtask} not supported.")

            logging.info(f"Building {subtask} task model, using config: {subtask_conf}")

            if m_subtask in ["diar", "sep"]:
                model_conf[f"{m_subtask}_model"] = eval(f"{subtask}_build_model")(
                    argparse.Namespace(**subtask_conf)
                )
            else:  # this is for "s2t" and "enh"
                model_conf[f"{m_subtask}_model"] = name2task[subtask].build_model(
                    argparse.Namespace(**subtask_conf)
                )

        # 8. Build model
        model = ESPnetEnhS2TModel(**model_conf)

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
