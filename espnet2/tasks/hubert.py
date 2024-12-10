#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert
import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import humanfriendly
import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.hubert_encoder import (  # noqa: H301
    FairseqHubertPretrainEncoder,
    TorchAudioHuBERTPretrainEncoder,
)
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.hubert.espnet_model import (
    HubertPretrainModel,
    TorchAudioHubertPretrainModel,
)
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import HuBERTCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend, sliding_window=SlidingWindow),
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
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        hubert_pretrain=FairseqHubertPretrainEncoder,
        torchaudio_hubert=TorchAudioHuBERTPretrainEncoder,
    ),
    type_check=AbsEncoder,
    default="hubert_pretrain",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        fairseq=HubertPretrainModel,
        torchaudio=TorchAudioHubertPretrainModel,
    ),
    type_check=AbsESPnetModel,
    default="fairseq",
)


class HubertTask(AbsTask):
    """
    HubertTask is a task class for training and evaluating HuBERT models in
    speech processing tasks. It inherits from AbsTask and provides methods
    to add task-specific arguments, build collate functions, preprocess data,
    and construct the model architecture based on the given configurations.

    This class supports various components, including frontends,
    spec augmentation, normalization, pre-encoders, and encoders. It allows
    customization through command-line arguments and facilitates easy
    integration with different model architectures.

    Attributes:
        num_optimizers (int): The number of optimizers used in the task.
        class_choices_list (list): A list of available class choices for
            task components (frontend, specaug, normalize, preencoder,
            encoder, model).
        trainer (Trainer): The trainer class used for training and evaluation.

    Args:
        parser (argparse.ArgumentParser): The argument parser instance to
            which task-specific arguments will be added.

    Returns:
        None

    Yields:
        None

    Raises:
        RuntimeError: If the token list is not a string or a list.

    Examples:
        To use HubertTask, you can initialize it and add task arguments as follows:

        ```python
        import argparse
        from espnet2.tasks.hubert import HubertTask

        parser = argparse.ArgumentParser(description="HuBERT Task Example")
        HubertTask.add_task_arguments(parser)
        args = parser.parse_args()
        ```

        You can then build the model based on the parsed arguments:

        ```python
        model = HubertTask.build_model(args)
        ```

    Note:
        The original HuBERT work is detailed in the paper:
        https://arxiv.org/pdf/2106.07447.pdf
        The implementation can be found in Fairseq:
        https://github.com/pytorch/fairseq/tree/master/examples/hubert

    Todo:
        - Add support for additional model architectures.
        - Enhance error handling for argument parsing.
    """

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
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
            Adds task-specific arguments to the provided argument parser.

        This method defines and adds various command-line arguments related to
        the task and preprocessing. It organizes the arguments into groups for
        clarity and ease of use. The added arguments include configurations for
        token lists, initialization methods, input sizes, and various preprocessing
        options.

        Args:
            cls: The class itself (HubertTask).
            parser (argparse.ArgumentParser): The argument parser to which the
                task-related arguments will be added.

        Examples:
            To add task arguments to a parser, you can do the following:

            ```python
            import argparse
            from hubert_task import HubertTask

            parser = argparse.ArgumentParser()
            HubertTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            The method modifies the parser in place and does not return a value.
            It is expected to be called as a class method.
        """
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
            "--collate_fn_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for collate_fn class.",
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--num_classes",
            type=int,
            default=None,
            help="The number of classes in hubert",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
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
        parser.add_argument(
            "--pred_masked_weight",
            type=float,
            default=1.0,
            help="weight for predictive loss for masked frames",
        )
        parser.add_argument(
            "--pred_nomask_weight",
            type=float,
            default=0.0,
            help="weight for predictive loss for unmasked frames",
        )
        parser.add_argument(
            "--loss_weights",
            type=float,
            default=0.0,
            help="weights for additional loss terms (not first one)",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        """
        Builds a collate function for processing batches of data during training or 
    evaluation.

    This method constructs a callable that will be used to collate a batch of 
    data into a format suitable for the model. It takes into account various 
    configurations related to the data, such as sampling rate and padding.

    Args:
        args (argparse.Namespace): The command-line arguments containing various 
            configurations, including `frontend_conf`, `collate_fn_conf`, and 
            `encoder_conf`.
        train (bool): A flag indicating whether the collate function is being 
            built for training or evaluation.

    Returns:
        Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]], 
                 Tuple[List[str], Dict[str, torch.Tensor]]]:
            A callable that collates a batch of data.

    Examples:
        >>> from argparse import Namespace
        >>> args = Namespace(
        ...     frontend_conf={"fs": 16000},
        ...     encoder_conf={},
        ...     collate_fn_conf={"label_downsampling": 1, "pad": True}
        ... )
        >>> collate_fn = HubertTask.build_collate_fn(args, train=True)
        >>> batch_data = collate_fn([("sample1", {"feature": np.array([1.0])}),
        ...                            ("sample2", {"feature": np.array([2.0])})])
        >>> print(batch_data)

    Note:
        The function assumes a default sampling rate of 16000 Hz if not 
        specified in the `frontend_conf`.

    Todo:
        - Consider adding support for additional collate configurations.
        """[
            Collection[Tuple[str, Dict[str, np.ndarray]]]
        ],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:

        # default sampling rate is 16000
        fs = args.frontend_conf.get("fs", 16000)
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        sample_rate = fs / 1000

        if args.encoder_conf.get("extractor_conv_layer_config", None) is None:
            # corresponding to default conv extractor
            # refer to espnet2/asr/encoder/hubert_encoder.py
            reception_field = 400
            stride_field = 320
        else:
            stride_field, reception_field = 1, 1
            for conv_config in args.encoder_conf["extractor_conv_layer_config"][::-1]:
                _, kernel, stride = conv_config
                stride_field *= stride
                reception_field = stride * (reception_field - 1) + kernel

        window_size = reception_field / sample_rate
        window_shift = stride_field / sample_rate
        return HuBERTCollateFn(
            float_pad_value=0.0,
            int_pad_value=-1,
            label_downsampling=args.collate_fn_conf.get("label_downsampling", 1),
            pad=args.collate_fn_conf.get("pad", False),
            rand_crop=args.collate_fn_conf.get("rand_crop", True),
            crop_audio=not args.collect_stats,
            window_size=window_size,
            window_shift=window_shift,
            sample_rate=sample_rate,
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
                Builds a preprocessing function for the HubertTask.

        This function returns a callable that preprocesses the input data
        according to the specified configuration. If preprocessing is not
        enabled, it returns None.

        Args:
            args (argparse.Namespace): The arguments namespace containing the
                preprocessing configurations.
            train (bool): A flag indicating whether the preprocessing is for
                training or not.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A preprocessing function that takes a string and a dictionary
                of numpy arrays and returns a dictionary of numpy arrays, or
                None if preprocessing is not enabled.

        Examples:
            # To build a preprocessing function for training
            preprocess_fn = HubertTask.build_preprocess_fn(args, train=True)

            # To use the preprocessing function
            processed_data = preprocess_fn("input_file.wav", {"key": np.array([])})

        Note:
            The preprocessing function applies various transformations such as
            tokenization, text cleaning, and volume normalization based on the
            provided arguments.
        """
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
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
        """
            Returns the names of the required data for the Hubert task.

        This method determines the required data names based on whether the task
        is in training or inference mode. In training mode, both 'speech' and
        'text' data are required. In inference mode, only 'speech' data is
        needed.

        Args:
            train (bool): A flag indicating if the task is in training mode.
                Default is True.
            inference (bool): A flag indicating if the task is in inference mode.
                Default is False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of the required data.
                If not in inference mode, returns ("speech", "text"). If in
                inference mode, returns ("speech",).

        Examples:
            >>> HubertTask.required_data_names(train=True, inference=False)
            ('speech', 'text')

            >>> HubertTask.required_data_names(train=False, inference=True)
            ('speech',)
        """
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            A task for training and evaluating Hubert models, inheriting from AbsTask.

        This class encapsulates methods for adding task-specific arguments, building
        data collators and preprocessors, and constructing the model. It also defines
        the required and optional data names needed for training and inference.

        Attributes:
            num_optimizers (int): The number of optimizers to be used for training.
            class_choices_list (list): List of class choices for different components
                of the model, such as frontend, specaug, normalize, preencoder,
                encoder, and model.
            trainer (Trainer): The class used for training and evaluation procedures.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which task
                related arguments will be added.

        Returns:
            None

        Yields:
            None

        Raises:
            RuntimeError: If the token list is neither a string nor a list.

        Examples:
            To add task arguments to a parser:
            ```python
            parser = argparse.ArgumentParser()
            HubertTask.add_task_arguments(parser)
            ```

            To build a collate function for training:
            ```python
            collate_fn = HubertTask.build_collate_fn(args, train=True)
            ```

            To build a preprocessor function:
            ```python
            preprocess_fn = HubertTask.build_preprocess_fn(args, train=True)
            ```

            To construct a model:
            ```python
            model = HubertTask.build_model(args)
            ```

        Note:
            The class supports multiple configurations for frontend, specaug,
            normalization, preencoder, encoder, and model, which can be specified
            through command line arguments.

        Todo:
            - Implement additional validation for input parameters.
            - Add more detailed logging for model building process.
        """
        retval = ()
        return retval

    @classmethod
    @typechecked
    def build_model(
        cls, args: argparse.Namespace
    ) -> Union[HubertPretrainModel, TorchAudioHubertPretrainModel]:
        """
        Builds and initializes the model based on provided arguments.

        This method constructs a Hubert model, including various components
        such as the frontend, data augmentation, normalization, pre-encoder,
        and encoder, using the configuration specified in the input arguments.

        Args:
            args (argparse.Namespace): The namespace containing configuration
                parameters for building the model.

        Returns:
            Union[HubertPretrainModel, TorchAudioHubertPretrainModel]:
                An instance of the Hubert model specified in the configuration.

        Raises:
            RuntimeError: If the `token_list` argument is not of type str or list.

        Examples:
            >>> args = argparse.Namespace(
            ...     token_list="path/to/token_list.txt",
            ...     input_size=None,
            ...     num_classes=10,
            ...     frontend="default",
            ...     frontend_conf={"fs": 16000},
            ...     specaug=None,
            ...     normalize="utterance_mvn",
            ...     preencoder=None,
            ...     encoder="hubert_pretrain",
            ...     model="fairseq",
            ...     model_conf={},
            ...     init="xavier_uniform"
            ... )
            >>> model = HubertTask.build_model(args)
            >>> print(model)
        """
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {**args.frontend_conf}
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
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(
            input_size=input_size,
            num_classes=args.num_classes,
            **args.encoder_conf,
        )

        # 8. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("fairseq")
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            token_list=token_list,
            **args.model_conf,
        )

        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
