import argparse
import copy
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

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
    """
    A task class for enhancing speech-to-text (S2T) tasks, which includes
    functionalities for speech enhancement, automatic speech recognition (ASR),
    and potentially other subtasks such as diarization and speech translation.

    This class inherits from AbsTask and provides methods for adding task
    arguments, building collate functions, preprocessing data, and creating
    models for the various subtasks involved in speech enhancement and
    recognition.

    Attributes:
        num_optimizers (int): Number of optimizers used for training.
        class_choices_list (List): A list of class choices for model components
            including encoders, decoders, and other task-specific components.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser): Adds task-specific
            arguments to the provided argument parser.
        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Builds a collate function for data loading.
        build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
            Builds a preprocessing function based on provided arguments.
        required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of required data for training or inference.
        optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of optional data for training or inference.
        build_model(args: argparse.Namespace) -> ESPnetEnhS2TModel:
            Constructs the ESPnetEnhS2TModel using the provided configuration.

    Examples:
        # To add task arguments
        parser = argparse.ArgumentParser()
        EnhS2TTask.add_task_arguments(parser)

        # To build a model
        args = parser.parse_args()
        model = EnhS2TTask.build_model(args)

    Note:
        The class can be extended or modified to include additional
        functionalities or support for more subtasks as needed.

    Todo:
        - Add more detailed logging for model building steps.
        - Implement unit tests for each method to ensure functionality.
    """

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
        """
            Adds task-specific arguments to the provided argument parser.

        This method configures the argument parser to accept various command-line
        arguments related to the enhancement, ASR, speech translation, and
        diarization tasks. It groups the arguments into task-related and
        preprocess-related categories.

        Args:
            cls: The class type, typically `EnhS2TTask`.
            parser (argparse.ArgumentParser): The argument parser to which the
                task arguments will be added.

        Examples:
            To add task arguments to an argument parser, you can use the method
            as follows:

            ```python
            import argparse
            from your_package import EnhS2TTask

            parser = argparse.ArgumentParser()
            EnhS2TTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            The method defines several argument groups, allowing for organized
            command-line interface (CLI) management of task configurations.
            Ensure that the required libraries and dependencies are installed
            for argument parsing and type checking.

        Raises:
            ValueError: If an invalid argument is provided that does not match
                the expected types or choices.
        """
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
        group.add_argument(
            "--text_name",
            nargs="+",
            default=["text"],
            type=str,
            help="Specify the text_name attribute used in the preprocessor",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        """
        Builds a collate function for preparing batches of data during training or
    evaluation.

    This method creates a collate function that takes a collection of tuples
    containing the data samples and their associated metadata. The function
    will pad the input data appropriately based on the specified padding values.

    Args:
        args (argparse.Namespace): The command-line arguments containing
            configuration settings for the task.
        train (bool): A flag indicating whether the collate function is being
            built for training (True) or evaluation (False).

    Returns:
        Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                 Tuple[List[str], Dict[str, torch.Tensor]]]:
            A collate function that processes input data and prepares it for
            batching.

    Note:
        The padding values are set such that float values are padded with
        `0.0` and integer values are padded with `-1`. The integer value `0`
        is reserved for the CTC-blank symbol.

    Examples:
        >>> collate_fn = EnhS2TTask.build_collate_fn(args, train=True)
        >>> batch_data = collate_fn([("id1", {"feature": np.array([1, 2])}),
                                      ("id2", {"feature": np.array([3])})])
        >>> print(batch_data)
        (['id1', 'id2'], {'feature': tensor([[1, 2], [3, -1]])})
        """[
            Collection[Tuple[str, Dict[str, np.ndarray]]]
        ],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
        Builds a preprocessing function based on the provided arguments and training
        state.

        This method creates a callable function that will preprocess the input data
        according to the specified subtask series. If the `use_preprocessor` argument
        is set to True, the appropriate preprocessing strategy is chosen based on
        the presence of subtasks like "st" (speech translation) or "diar"
        (diarization).

        Args:
            args (argparse.Namespace): Command line arguments that specify various
                configurations for the preprocessing.
            train (bool): A flag indicating whether the preprocessing function is
                being built for training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A callable preprocessing function that processes the input data,
                or None if preprocessing is not enabled.

        Examples:
            >>> args = argparse.Namespace()
            >>> args.use_preprocessor = True
            >>> args.subtask_series = ['enh', 'asr']
            >>> preprocess_fn = EnhS2TTask.build_preprocess_fn(args, train=True)
            >>> result = preprocess_fn('input_file.wav', {'key': np.array([1, 2, 3])})

        Note:
            The returned function will vary depending on the specified
            subtask series and preprocessing configurations.
        """
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
                    rir_scp=getattr(args, "rir_scp", None),
                    rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
                    noise_scp=getattr(args, "noise_scp", None),
                    noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
                    noise_db_range=getattr(args, "noise_db_range", "13_15"),
                    short_noise_thres=getattr(args, "short_noise_thres", 0.5),
                    speech_volume_normalize=getattr(
                        args, "speech_volume_normalize", None
                    ),
                    speech_name="speech",
                    text_name=["text", "src_text"],
                    **getattr(args, "preprocessor_conf", {}),
                )
            elif "diar" in args.subtask_series:
                retval = CommonPreprocessor(
                    train=train, **getattr(args, "preprocessor_conf", {})
                )
            else:
                retval = CommonPreprocessor_multi(
                    train=train,
                    token_type=args.token_type,
                    token_list=args.token_list,
                    bpemodel=args.bpemodel,
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_name=getattr(args, "text_name", ["text"]),
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                    **getattr(args, "preprocessor_conf", {}),
                )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns the required data names for the task.

        The required data names depend on whether the task is in inference mode.
        If not in inference mode, the required data names include the speech
        input and the first reference speech. In inference mode, only the speech
        input is required.

        Args:
            train (bool): A flag indicating if the task is in training mode.
                Default is True.
            inference (bool): A flag indicating if the task is in inference mode.
                Default is False.

        Returns:
            Tuple[str, ...]: A tuple of required data names.

        Examples:
            >>> required_data_names()
            ('speech', 'speech_ref1')

            >>> required_data_names(inference=True)
            ('speech',)
        """
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
        """
            Returns a tuple of optional data names that can be used in the task.

        The method constructs a list of optional data names based on the training
        and inference flags. It includes standard references such as "text",
        "dereverb_ref1", and dynamically adds multiple references for speech,
        noise, and speaker text based on the defined maximum number of references.

        Args:
            train (bool): A flag indicating whether the task is in training mode.
                Defaults to True.
            inference (bool): A flag indicating whether the task is in inference
                mode. Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of optional data
            attributes.

        Examples:
            >>> optional_names = EnhS2TTask.optional_data_names(train=True)
            >>> print(optional_names)
            ('text', 'dereverb_ref1', 'speech_ref2', 'speech_ref3', ...)

            >>> optional_names = EnhS2TTask.optional_data_names(inference=True)
            >>> print(optional_names)
            ('text', 'dereverb_ref1', 'speech_ref1', ...)

        Note:
            The maximum number of speech and noise references is defined by
            the constant `MAX_REFERENCE_NUM`, which is set to 100.
        """
        retval = ["text", "dereverb_ref1"]
        st = 2 if "speech_ref1" in retval else 1
        retval += ["speech_ref{}".format(n) for n in range(st, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["text_spk{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["src_text"]
        retval = tuple(retval)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhS2TModel:
        """
                Builds the enhancement speech-to-text (S2T) model based on the provided
        configuration arguments. This method constructs submodels for each specified
        subtask in the order of `subtask_series`, and initializes the main model.

        Args:
            args (argparse.Namespace): The argument namespace containing model
                configuration and other necessary parameters.

        Returns:
            ESPnetEnhS2TModel: An instance of the enhancement S2T model configured
                according to the specified arguments.

        Raises:
            ValueError: If a subtask specified in `subtask_series` is not supported.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     model_conf={'some_model_param': 'value'},
            ...     subtask_series=['enh', 'asr'],
            ...     enh_model_conf={'enh_param': 'value'},
            ...     asr_model_conf={'asr_param': 'value'},
            ...     # ... other required arguments
            ... )
            >>> model = EnhS2TTask.build_model(args)
            >>> print(model)

        Note:
            Ensure that all required arguments are provided in `args` to avoid
            initialization errors.

        Todo:
            - Implement better error handling for missing parameters.
        """

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

        return model
