"""ASR Transducer Task."""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.decoder.mega_decoder import MEGADecoder
from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.rwkv_decoder import RWKVDecoder
from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.espnet_transducer_model import ESPnetASRTransducerModel
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    "specaug",
    classes=dict(
        specaug=SpecAug,
    ),
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
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        mega=MEGADecoder,
        rnn=RNNDecoder,
        rwkv=RWKVDecoder,
        stateless=StatelessDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)


class ASRTransducerTask(AbsTask):
    """
        ASR Transducer Task definition.

    This class implements the ASR (Automatic Speech Recognition) Transducer task,
    which includes functionalities for building models, processing data, and
    handling task-specific arguments.

    Attributes:
        num_optimizers (int): Number of optimizers used for training.
        class_choices_list (List[ClassChoices]): List of available class choices
            for frontend, specaug, normalization, and decoder.
        trainer (Trainer): Trainer class used for managing training processes.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser): Adds ASR Transducer
            task arguments to the provided argument parser.
        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Builds a collate function for batching data.
        build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
            Builds a pre-processing function for input data.
        required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the required data names based on task mode.
        optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the optional data names based on task mode.
        build_model(args: argparse.Namespace) -> ESPnetASRTransducerModel:
            Builds and returns the ASR Transducer model based on the provided
            arguments.

    Examples:
        To add task arguments:
            parser = argparse.ArgumentParser()
            ASRTransducerTask.add_task_arguments(parser)

        To build a model:
            args = parser.parse_args()
            model = ASRTransducerTask.build_model(args)

    Raises:
        RuntimeError: If the token_list is not of type str or list.
        NotImplementedError: If the initialization is not supported.

    Note:
        The class relies on various components like frontend, specaug,
        normalization, and decoders, which can be customized through the
        task arguments.
    """

    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        decoder_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
                Add Transducer task arguments.

        This method is responsible for adding command-line arguments specific to the
        ASR Transducer task to the provided argument parser. The arguments include
        configuration options for the model, encoder, joint network, preprocessing,
        and data augmentation.

        Args:
            cls: ASRTransducerTask object.
            parser: Transducer arguments parser.

        Examples:
            To add task arguments to an argument parser, you can use:

            ```python
            import argparse
            from your_module import ASRTransducerTask

            parser = argparse.ArgumentParser()
            ASRTransducerTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            The method modifies the parser in-place by adding a group of arguments
            related to the ASR Transducer task, such as `--token_list`, `--input_size`,
            `--init`, and others.
        """
        group = parser.add_argument_group(description="Task related.")

        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="Integer-string mapper for tokens.",
        )
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of dimensions for input features.",
        )
        group.add_argument(
            "--init",
            type=str_or_none,
            default=None,
            help="Type of model initialization to use.",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRTransducerModel),
            help="The keyword arguments for the model class.",
        )
        group.add_argument(
            "--encoder_conf",
            action=NestedDictAction,
            default={},
            help="The keyword arguments for the encoder class.",
        )
        group.add_argument(
            "--joint_network_conf",
            action=NestedDictAction,
            default={},
            help="The keyword arguments for the joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related.")

        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Whether to apply preprocessing to input data.",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The type of tokens to use during tokenization.",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The path of the sentencepiece model.",
        )
        group.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="The 'non_linguistic_symbols' file path.",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Text cleaner to use.",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="g2p method to use if --token_type=phn.",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Normalization value for maximum amplitude scaling.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The RIR SCP file path.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="The probability of the applied RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The path of noise SCP file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability of the applied noise addition.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of the noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --decoder and --decoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        """
        Build collate function.

        This method constructs a collate function that is used to combine
        multiple samples into a mini-batch during training or evaluation.

        Args:
            cls: ASRTransducerTask object.
            args: Task arguments containing configurations for the collate
                function.
            train: A boolean indicating whether the function is for training
                mode or not.

        Returns:
            Callable: A collate function that takes a collection of tuples,
            where each tuple contains a string and a dictionary of NumPy
            arrays, and returns a tuple containing a list of strings and a
            dictionary of PyTorch tensors.

        Examples:
            >>> collate_fn = ASRTransducerTask.build_collate_fn(args, train=True)
            >>> batch = collate_fn(data)
            >>> print(batch)
            (['example1', 'example2'], {'features': tensor(...), ...})
        """

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
            Build pre-processing function.

        This method constructs a pre-processing function based on the provided
        arguments. If preprocessing is enabled, it utilizes the `CommonPreprocessor`
        to handle various preprocessing tasks such as tokenization, noise
        application, and volume normalization.

        Args:
            cls: ASRTransducerTask object.
            args: Task arguments containing configurations for preprocessing.
            train: A boolean indicating whether the function is for training mode.

        Returns:
            A callable pre-processing function that takes a string and a dictionary
            of numpy arrays as input and returns a dictionary of numpy arrays, or
            None if preprocessing is not enabled.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(use_preprocessor=True, token_type='bpe', ...)
            >>> preprocess_fn = ASRTransducerTask.build_preprocess_fn(args, train=True)
            >>> result = preprocess_fn("sample text", {"feature": np.array([1, 2, 3])})

        Note:
            This function is intended for use in preparing input data for the ASR
            model during training or inference.
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
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=(
                    args.rir_apply_prob if hasattr(args, "rir_apply_prob") else 1.0
                ),
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=(
                    args.noise_apply_prob if hasattr(args, "noise_apply_prob") else 1.0
                ),
                noise_db_range=(
                    args.noise_db_range if hasattr(args, "noise_db_range") else "13_15"
                ),
                speech_volume_normalize=(
                    args.speech_volume_normalize if hasattr(args, "rir_scp") else None
                ),
            )
        else:
            retval = None

        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
        Required data depending on task mode.

        This method returns the names of the required data based on whether the
        task is in training or inference mode.

        Args:
            cls: ASRTransducerTask object.
            train: A boolean indicating if the task is in training mode.
            inference: A boolean indicating if the task is in inference mode.

        Returns:
            Tuple[str, ...]: A tuple containing the required task data names.
                - If not in inference mode, returns ("speech", "text").
                - If in inference mode, returns ("speech",).

        Examples:
            >>> ASRTransducerTask.required_data_names(train=True, inference=False)
            ('speech', 'text')
            >>> ASRTransducerTask.required_data_names(train=False, inference=True)
            ('speech',)
        """
        if not inference:
            retval = ("speech", "text")
        else:
            retval = ("speech",)

        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
        Optional data depending on task mode.

        This method returns a tuple of optional data names that may be used
        during the training or inference modes of the ASR Transducer Task.

        Args:
            cls: ASRTransducerTask object.
            train: A boolean indicating whether the task is in training mode.
            inference: A boolean indicating whether the task is in inference mode.

        Returns:
            retval: A tuple containing the optional task data names.

        Examples:
            >>> ASRTransducerTask.optional_data_names(train=True)
            ()
            >>> ASRTransducerTask.optional_data_names(inference=True)
            ()

        Note:
            The default implementation returns an empty tuple, indicating that
            there are no optional data names. Subclasses may override this
            method to provide specific optional data names.
        """
        retval = ()

        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRTransducerModel:
        """
        Builds the ASR Transducer model based on provided arguments.

        This method constructs an instance of the `ESPnetASRTransducerModel` by
        configuring the frontend, data augmentation, normalization, encoder,
        decoder, and joint network based on the parameters specified in the
        `args` argument.

        Args:
            cls: ASRTransducerTask object.
            args: Task arguments containing configurations for model components.

        Returns:
            model: An instance of the ASR Transducer model configured as per
            the specified arguments.

        Raises:
            RuntimeError: If `token_list` is neither a string nor a list.
            NotImplementedError: If model initialization is requested but not
            currently supported.

        Examples:
            >>> args = argparse.Namespace()
            >>> args.token_list = "path/to/token_list.txt"
            >>> args.input_size = None
            >>> args.specaug = "specaug"
            >>> model = ASRTransducerTask.build_model(args)
            >>> print(model)

        Note:
            The `token_list` is read from a file if provided as a string. If
            it's a list, it is used directly. The method logs the vocabulary
            size and initializes various components of the model as specified
            in the arguments.
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

        if hasattr(args, "scheduler_conf"):
            args.model_conf["warmup_steps"] = args.scheduler_conf.get(
                "warmup_steps", 25000
            )

        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
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

        # 4. Encoder
        encoder = Encoder(input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size,
            **args.decoder_conf,
        )
        decoder_output_size = decoder.output_size

        # 6. Joint Network
        joint_network = JointNetwork(
            vocab_size,
            encoder_output_size,
            decoder_output_size,
            **args.joint_network_conf,
        )

        # 7. Build model
        model = ESPnetASRTransducerModel(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            joint_network=joint_network,
            **args.model_conf,
        )

        # 8. Initialize model
        if args.init is not None:
            raise NotImplementedError(
                "Currently not supported.",
                "Initialization part will be reworked in a short future.",
            )

        return model
