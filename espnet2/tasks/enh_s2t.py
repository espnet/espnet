import argparse
import copy
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from espnet2.enh.espnet_enh_s2t_model import ESPnetEnhS2TModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr import decoder_choices as asr_decoder_choices_
from espnet2.tasks.asr import encoder_choices as asr_encoder_choices_
from espnet2.tasks.asr import frontend_choices, normalize_choices
from espnet2.tasks.asr import postencoder_choices as asr_postencoder_choices_
from espnet2.tasks.asr import preencoder_choices as asr_preencoder_choices_
from espnet2.tasks.asr import specaug_choices
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.diar import attractor_choices as diar_attractor_choices_
from espnet2.tasks.diar import decoder_choices as diar_decoder_choices_
from espnet2.tasks.diar import encoder_choices as diar_encoder_choices_
from espnet2.tasks.diar import frontend_choices as diar_front_end_choices_
from espnet2.tasks.diar import label_aggregator_choices
from espnet2.tasks.diar import normalize_choices as diar_normalize_choices_
from espnet2.tasks.diar import specaug_choices as diar_specaug_choices_
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh import decoder_choices as enh_decoder_choices_
from espnet2.tasks.enh import encoder_choices as enh_encoder_choices_
from espnet2.tasks.enh import mask_module_choices as enh_mask_module_choices_
from espnet2.tasks.enh import separator_choices as enh_separator_choices_
from espnet2.tasks.st import STTask
from espnet2.tasks.st import decoder_choices as st_decoder_choices_
from espnet2.tasks.st import encoder_choices as st_encoder_choices_
from espnet2.tasks.st import extra_asr_decoder_choices as st_extra_asr_decoder_choices_
from espnet2.tasks.st import extra_mt_decoder_choices as st_extra_mt_decoder_choices_
from espnet2.tasks.st import postencoder_choices as st_postencoder_choices_
from espnet2.tasks.st import preencoder_choices as st_preencoder_choices_
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    CommonPreprocessor,
    CommonPreprocessor_multi,
    MutliTokenizerCommonPreprocessor,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

# Enhancement
enh_encoder_choices = copy.deepcopy(enh_encoder_choices_)
enh_encoder_choices.name = "enh_encoder"
enh_decoder_choices = copy.deepcopy(enh_decoder_choices_)
enh_decoder_choices.name = "enh_decoder"
enh_separator_choices = copy.deepcopy(enh_separator_choices_)
enh_separator_choices.name = "enh_separator"
enh_mask_module_choices = copy.deepcopy(enh_mask_module_choices_)
enh_mask_module_choices.name = "enh_mask_module"

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

# DIAR
diar_frontend_choices = copy.deepcopy(diar_front_end_choices_)
diar_frontend_choices.name = "diar_frontend"
diar_specaug_choices = copy.deepcopy(diar_specaug_choices_)
diar_specaug_choices.name = "diar_specaug"
diar_normalize_choices = copy.deepcopy(diar_normalize_choices_)
diar_normalize_choices.name = "diar_normalize"
diar_encoder_choices = copy.deepcopy(diar_encoder_choices_)
diar_encoder_choices.name = "diar_encoder"
diar_decoder_choices = copy.deepcopy(diar_decoder_choices_)
diar_decoder_choices.name = "diar_decoder"
diar_attractor_choices = copy.deepcopy(diar_attractor_choices_)
diar_attractor_choices.name = "diar_attractor"


MAX_REFERENCE_NUM = 100

name2task = dict(
    enh=EnhancementTask,
    asr=ASRTask,
    st=STTask,
    diar=DiarizationTask,
)

# More can be added to the following attributes
enh_attributes = [
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
    "input_size",
    "num_spk",
    "frontend",
    "frontend_conf",
    "specaug",
    "specaug_conf",
    "normalize",
    "normalize_conf",
    "encoder",
    "encoder_conf",
    "decoder",
    "decoder_conf",
    "attractor",
    "attractor_conf",
    "label_aggregator",
    "label_aggregator_conf",
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
        # --enh_mask_module and --enh_mask_module_conf
        enh_mask_module_choices,
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
        # --diar_attractor and --diar_attractor_conf
        diar_attractor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

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
            "--diar_num_spk",
            type=int_or_none,
            default=None,
            help="The number of speakers (for each recording) for diar submodel class",
        )

        group.add_argument(
            "--diar_input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
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
            "--diar_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetDiarizationModel),
            help="The keyword arguments for diar submodel class.",
        )

        group.add_argument(
            "--subtask_series",
            type=str,
            nargs="+",
            default=("enh", "asr"),
            choices=["enh", "asr", "st", "diar"],
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
        group.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
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
                    short_noise_thres=args.short_noise_thres
                    if hasattr(args, "short_noise_thres")
                    else 0.5,
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
            retval = ("speech", "speech_ref1", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref1"]
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

            if subtask in ["asr", "st", "diar"]:
                m_subtask = "s2t"
            elif subtask in ["enh"]:
                m_subtask = subtask
            else:
                raise ValueError(f"{subtask} not supported.")

            logging.info(f"Building {subtask} task model, using config: {subtask_conf}")

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
