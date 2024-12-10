# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based text-to-speech task."""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.espnet_model import ESPnetGANTTSModel
from espnet2.gan_tts.jets import JETS
from espnet2.gan_tts.joint import JointText2Wav
from espnet2.gan_tts.vits import VITS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.gan_trainer import GANTrainer
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

feats_extractor_choices = ClassChoices(
    "feats_extract",
    classes=dict(
        fbank=LogMelFbank,
        log_spectrogram=LogSpectrogram,
        linear_spectrogram=LinearSpectrogram,
    ),
    type_check=AbsFeatsExtract,
    default="linear_spectrogram",
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
tts_choices = ClassChoices(
    "tts",
    classes=dict(
        vits=VITS,
        joint_text2wav=JointText2Wav,
        jets=JETS,
    ),
    type_check=AbsGANTTS,
    default="vits",
)
pitch_extractor_choices = ClassChoices(
    "pitch_extract",
    classes=dict(dio=Dio),
    type_check=AbsFeatsExtract,
    default=None,
    optional=True,
)
energy_extractor_choices = ClassChoices(
    "energy_extract",
    classes=dict(energy=Energy),
    type_check=AbsFeatsExtract,
    default=None,
    optional=True,
)
pitch_normalize_choices = ClassChoices(
    "pitch_normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
energy_normalize_choices = ClassChoices(
    "energy_normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)


class GANTTSTask(AbsTask):
    """
    GAN-based text-to-speech task.

    This class implements a GAN-based approach for text-to-speech (TTS)
    synthesis. It includes functionalities for building models, managing
    data, and processing input features, all tailored for generative
    adversarial networks.

    Attributes:
        num_optimizers (int): Number of optimizers required by GAN (default is 2).
        class_choices_list (list): A list of class choices for various
            components such as feature extraction, normalization, and TTS models.
        trainer (type): Specifies the trainer class to be used (GANTrainer).

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser):
            Adds task-specific arguments to the argument parser.

        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Builds a collate function for batching data.

        build_preprocess_fn(args: argparse.Namespace, train: bool) ->
            Optional[Callable]:
            Constructs a preprocessing function based on the input arguments.

        required_data_names(train: bool = True, inference: bool = False) ->
            Tuple[str, ...]:
            Returns a tuple of required data names for training or inference.

        optional_data_names(train: bool = True, inference: bool = False) ->
            Tuple[str, ...]:
            Returns a tuple of optional data names for training or inference.

        build_model(args: argparse.Namespace) -> ESPnetGANTTSModel:
            Builds the ESPnet GAN TTS model based on the provided arguments.

        build_optimizers(args: argparse.Namespace, model: ESPnetGANTTSModel) ->
            List[torch.optim.Optimizer]:
            Constructs the optimizers for the model.

    Examples:
        To add task arguments:
            parser = argparse.ArgumentParser()
            GANTTSTask.add_task_arguments(parser)

        To build the model:
            args = parser.parse_args()
            model = GANTTSTask.build_model(args)

        To build optimizers:
            optimizers = GANTTSTask.build_optimizers(args, model)

    Note:
        The class uses a combination of various components including feature
        extractors, normalization layers, and different TTS models to create
        a complete TTS pipeline.

    Todo:
        - Extend the class to support additional TTS models and features.
    """

    # GAN requires two optimizers
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --feats_extractor and --feats_extractor_conf
        feats_extractor_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --tts and --tts_conf
        tts_choices,
        # --pitch_extract and --pitch_extract_conf
        pitch_extractor_choices,
        # --pitch_normalize and --pitch_normalize_conf
        pitch_normalize_choices,
        # --energy_extract and --energy_extract_conf
        energy_extractor_choices,
        # --energy_normalize and --energy_normalize_conf
        energy_normalize_choices,
    ]

    # Use GANTrainer instead of Trainer
    trainer = GANTrainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
            Adds task-related arguments to the argument parser for the GANTTSTask.

        This method is responsible for defining the command line arguments that
        are specific to the GANTTSTask, including configurations for feature
        extraction, normalization, and text-to-speech models. It organizes the
        arguments into groups for better clarity and structure.

        Args:
            cls: The class itself (used for class method).
            parser (argparse.ArgumentParser): The argument parser to which the
                task-related arguments will be added.

        Note:
            The function modifies the parser directly and adds several required
            and optional arguments that are essential for the GANTTSTask.

        Examples:
            To use the arguments added by this method, you might do:

            ```python
            import argparse
            from gantts_task import GANTTSTask

            parser = argparse.ArgumentParser()
            GANTTSTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Raises:
            ValueError: If the argument parsing fails or if required arguments
                are not provided.
        """
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
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
            "--odim",
            type=int_or_none,
            default=None,
            help="The number of dimension of output feature",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetGANTTSModel),
            help="The keyword arguments for model class.",
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
            default="phn",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese", "korean_cleaner"],
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
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        """
        Builds a collate function for batching input data.

        This method creates a callable that can be used to collate
        input data into batches for training or evaluation. It
        handles padding for different types of input data.

        Args:
            args (argparse.Namespace): The arguments parsed from
                the command line, including configuration options.
            train (bool): A flag indicating whether the function
                is being built for training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]]:
                A callable that takes a collection of input data
                tuples and returns a batch of input data.

        Examples:
            >>> collate_fn = GANTTSTask.build_collate_fn(args, train=True)
            >>> batch = collate_fn(data)

        Note:
            The collate function will pad the sequences to the
            maximum length within the batch. It will ignore
            padding for specified fields, such as "spembs",
            "sids", and "lids".
        """
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=0,
            not_sequence=["spembs", "sids", "lids"],
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
        Builds a preprocessing function based on the provided arguments.

        This function returns a callable that can be used to preprocess input
        data before it is fed into the model. If preprocessing is disabled,
        it returns None.

        Args:
            args (argparse.Namespace): The parsed arguments containing
                configurations for preprocessing.
            train (bool): A flag indicating whether the function is for
                training or inference.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]],
            Dict[str, np.ndarray]]]: A preprocessing function if
            `args.use_preprocessor` is True, otherwise None.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(use_preprocessor=True, token_type='phn',
            ... token_list='path/to/token_list', bpemodel=None,
            ... non_linguistic_symbols=None, cleaner=None, g2p=None)
            >>> preprocess_fn = GANTTSTask.build_preprocess_fn(args, train=True)
            >>> processed_data = preprocess_fn("sample_text",
            ... {"additional_data": np.array([1, 2, 3])})

        Note:
            This method relies on the `CommonPreprocessor` for the actual
            preprocessing logic. Ensure that the necessary configurations
            are provided in the `args`.
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
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Get the required data names for the GANTTS task.

        This method returns a tuple of required data names based on whether the task
        is in training or inference mode. When in training mode, both "text" and
        "speech" data are required. In inference mode, only "text" data is required.

        Args:
            train (bool, optional): Indicates if the task is in training mode. Defaults
                to True.
            inference (bool, optional): Indicates if the task is in inference mode.
                Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the required data names.

        Examples:
            >>> GANTTSTask.required_data_names(train=True, inference=False)
            ('text', 'speech')

            >>> GANTTSTask.required_data_names(train=False, inference=True)
            ('text',)
        """
        if not inference:
            retval = ("text", "speech")
        else:
            # Inference mode
            retval = ("text",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            GAN-based text-to-speech task.

        This class defines a GANTTSTask for generating speech from text using GANs.
        It includes methods for argument parsing, building models, optimizers, and
        processing data for training and inference.

        Attributes:
            num_optimizers (int): The number of optimizers required for GAN training.
            class_choices_list (List[ClassChoices]): A list of class choices for
                feature extraction, normalization, and TTS methods.
            trainer (Type[GANTrainer]): The trainer class used for training.

        Args:
            parser (argparse.ArgumentParser): The argument parser for the task.

        Returns:
            None

        Yields:
            None

        Raises:
            RuntimeError: If `token_list` is not of type str or dict.
            ValueError: If optimizer type is not recognized.

        Examples:
            To add task arguments to the parser:

            ```python
            parser = argparse.ArgumentParser()
            GANTTSTask.add_task_arguments(parser)
            ```

            To build the model with specific arguments:

            ```python
            model = GANTTSTask.build_model(args)
            ```

            To create optimizers for the model:

            ```python
            optimizers = GANTTSTask.build_optimizers(args, model)
            ```

        Note:
            The task uses two optimizers for the generator and discriminator.

        Todo:
            - Implement additional features for enhanced preprocessing.
        """
        if not inference:
            retval = (
                "spembs",
                "durations",
                "pitch",
                "energy",
                "sids",
                "lids",
            )
        else:
            # Inference mode
            retval = (
                "spembs",
                "speech",
                "durations",
                "pitch",
                "energy",
                "sids",
                "lids",
            )
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetGANTTSModel:
        """
            Build the model for the GAN-based text-to-speech task.

        This method constructs the ESPnetGANTTSModel using the provided arguments.
        It initializes various components of the model such as feature extractors,
        normalization layers, and the TTS module based on the input arguments.

        Args:
            args (argparse.Namespace): The parsed command-line arguments containing
                configuration options for building the model.

        Returns:
            ESPnetGANTTSModel: An instance of the ESPnetGANTTSModel constructed
                with the specified configuration.

        Raises:
            RuntimeError: If the token_list argument is not of type str or dict.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(
            ...     token_list="path/to/token_list.txt",
            ...     odim=None,
            ...     feats_extract="log_spectrogram",
            ...     feats_extract_conf={},
            ...     normalize="global_mvn",
            ...     normalize_conf={},
            ...     tts="vits",
            ...     tts_conf={},
            ...     pitch_extract=None,
            ...     energy_extract=None,
            ...     pitch_normalize=None,
            ...     energy_normalize=None,
            ...     model_conf={}
            ... )
            >>> model = GANTTSTask.build_model(args)
            >>> print(model)
        """
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line[0] + line[1:].rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. feats_extract
        if args.odim is None:
            # Extract features in the model
            feats_extract_class = feats_extractor_choices.get_class(args.feats_extract)
            feats_extract = feats_extract_class(**args.feats_extract_conf)
            odim = feats_extract.output_size()
        else:
            # Give features from data-loader
            args.feats_extract = None
            args.feats_extract_conf = None
            feats_extract = None
            odim = args.odim

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 3. TTS
        tts_class = tts_choices.get_class(args.tts)
        tts = tts_class(idim=vocab_size, odim=odim, **args.tts_conf)

        # 4. Extra components
        pitch_extract = None
        energy_extract = None
        pitch_normalize = None
        energy_normalize = None
        if getattr(args, "pitch_extract", None) is not None:
            pitch_extract_class = pitch_extractor_choices.get_class(
                args.pitch_extract,
            )
            pitch_extract = pitch_extract_class(
                **args.pitch_extract_conf,
            )
        if getattr(args, "energy_extract", None) is not None:
            energy_extract_class = energy_extractor_choices.get_class(
                args.energy_extract,
            )
            energy_extract = energy_extract_class(
                **args.energy_extract_conf,
            )
        if getattr(args, "pitch_normalize", None) is not None:
            pitch_normalize_class = pitch_normalize_choices.get_class(
                args.pitch_normalize,
            )
            pitch_normalize = pitch_normalize_class(
                **args.pitch_normalize_conf,
            )
        if getattr(args, "energy_normalize", None) is not None:
            energy_normalize_class = energy_normalize_choices.get_class(
                args.energy_normalize,
            )
            energy_normalize = energy_normalize_class(
                **args.energy_normalize_conf,
            )

        # 5. Build model
        model = ESPnetGANTTSModel(
            feats_extract=feats_extract,
            normalize=normalize,
            pitch_extract=pitch_extract,
            pitch_normalize=pitch_normalize,
            energy_extract=energy_extract,
            energy_normalize=energy_normalize,
            tts=tts,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetGANTTSModel,
    ) -> List[torch.optim.Optimizer]:
        """
        Builds the optimizers for the GAN-based text-to-speech model.

        This method initializes two optimizers: one for the generator and
        one for the discriminator. The optimizers are configured based on
        the arguments provided in `args` and the model structure.

        Args:
            args (argparse.Namespace): The arguments namespace containing
                configuration settings, such as optimizer types and their
                respective configurations.
            model (ESPnetGANTTSModel): The GAN-based text-to-speech model
                which contains both the generator and discriminator.

        Returns:
            List[torch.optim.Optimizer]: A list containing the initialized
                optimizers for the generator and discriminator.

        Raises:
            ValueError: If the specified optimizer class is not found in the
                available optimizer classes.
            RuntimeError: If `fairscale` is required but not installed.

        Examples:
            >>> from espnet2.train.class_choices import ClassChoices
            >>> args = argparse.Namespace(
            ...     optim='adam',
            ...     optim_conf={'lr': 0.001},
            ...     optim2='sgd',
            ...     optim2_conf={'lr': 0.01},
            ...     sharded_ddp=False,
            ... )
            >>> model = ESPnetGANTTSModel(...)
            >>> optimizers = GANTTSTask.build_optimizers(args, model)
            >>> assert len(optimizers) == 2  # One for generator and one for discriminator
        """
        # check
        assert hasattr(model.tts, "generator")
        assert hasattr(model.tts, "discriminator")

        # define generator optimizer
        optim_g_class = optim_classes.get(args.optim)
        if optim_g_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_g = fairscale.optim.oss.OSS(
                params=model.tts.generator.parameters(),
                optim=optim_g_class,
                **args.optim_conf,
            )
        else:
            optim_g = optim_g_class(
                model.tts.generator.parameters(),
                **args.optim_conf,
            )
        optimizers = [optim_g]

        # define discriminator optimizer
        optim_d_class = optim_classes.get(args.optim2)
        if optim_d_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_d = fairscale.optim.oss.OSS(
                params=model.tts.discriminator.parameters(),
                optim=optim_d_class,
                **args.optim2_conf,
            )
        else:
            optim_d = optim_d_class(
                model.tts.discriminator.parameters(),
                **args.optim2_conf,
            )
        optimizers += [optim_d]

        return optimizers
