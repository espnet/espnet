"""Text-to-speech task."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import typechecked

from espnet2.tasks.abs_task import AbsTask

# TTS continuous feature extraction operators
from espnet2.tasks.tts import (
    energy_extractor_choices,
    energy_normalize_choices,
    pitch_extractor_choices,
    pitch_normalize_choices,
)
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.tts2.abs_tts2 import AbsTTS2
from espnet2.tts2.espnet_model import ESPnetTTS2Model
from espnet2.tts2.fastspeech2 import FastSpeech2Discrete
from espnet2.tts2.feats_extract.abs_feats_extract import AbsFeatsExtractDiscrete
from espnet2.tts2.feats_extract.identity import IdentityFeatureExtract
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

discrete_feats_extractor_choices = ClassChoices(
    "discrete_feats_extract",
    classes=dict(
        identity=IdentityFeatureExtract,
    ),
    type_check=AbsFeatsExtractDiscrete,
    default="identity",
)
tts_choices = ClassChoices(
    "tts",
    classes=dict(
        fastspeech2=FastSpeech2Discrete,
    ),
    type_check=AbsTTS2,
    default="fastspeech2",
)


class TTS2Task(AbsTask):
    """
        Text-to-speech (TTS) task class for ESPnet2.

    This class is responsible for managing the TTS task, including setting up
    arguments, building models, and handling data processing. It extends the
    abstract base task class `AbsTask`.

    Attributes:
        num_optimizers (int): Number of optimizers to use. Default is 1.
        class_choices_list (List[ClassChoices]): List of class choices for
            various components in the TTS task.
        trainer (Trainer): Trainer class used for training procedures.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser):
            Adds command-line arguments specific to the TTS task.

        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Builds a collate function for batching data.

        build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
            Builds a preprocessing function for input data.

        required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of the required data for the task.

        optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of the optional data for the task.

        build_model(args: argparse.Namespace) -> ESPnetTTS2Model:
            Constructs the TTS model based on the provided arguments.

        build_vocoder_from_file(
            vocoder_config_file: Union[Path, str] = None,
            vocoder_file: Union[Path, str] = None,
            model: Optional[ESPnetTTS2Model] = None,
            device: str = "cpu"
        ):
            Builds a vocoder from the specified configuration and model files.

    Examples:
        To add task-specific arguments to a parser:
            parser = argparse.ArgumentParser()
            TTS2Task.add_task_arguments(parser)

        To build a model:
            args = ...  # Namespace with necessary arguments
            model = TTS2Task.build_model(args)

    Note:
        Ensure that the necessary configuration files and data are available
        when using this class.

    Todo:
        - Implement additional features for enhanced flexibility.
    """

    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --discrete_feats_extractor and --discrete_feats_extractor_conf
        discrete_feats_extractor_choices,
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

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
        Add task-related arguments to the argument parser.

        This method defines and adds various arguments related to the TTS task
        that are necessary for model configuration and data preprocessing. The
        arguments include options for specifying source and target token lists,
        model configurations, and preprocessing preferences.

        Args:
            parser (argparse.ArgumentParser): The argument parser instance to
                which the task-related arguments will be added.

        Note:
            Use '_' instead of '-' to avoid confusion in argument names.

        Examples:
            To use this method, you can create an argument parser and call
            `add_task_arguments`:

            ```python
            import argparse
            parser = argparse.ArgumentParser()
            TTS2Task.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Raises:
            RuntimeError: If there is an issue with argument parsing or required
                arguments are missing.
        """
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["src_token_list", "tgt_token_list"]

        group.add_argument(
            "--src_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--tgt_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to target speech token",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetTTS2Model),
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
            "--src_token_type",
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
            default=None,
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
                Build a collate function for batching data during training or evaluation.

        This method constructs a collate function that is used to combine multiple
        data samples into a single batch. The collate function will pad sequences
        to the maximum length in the batch and convert the data into appropriate
        tensor formats.

        Args:
            args (argparse.Namespace): The command-line arguments parsed into a
                namespace object.
            train (bool): A flag indicating whether the collate function is being
                built for training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]:
                A callable that takes a collection of data samples and returns a
                tuple containing a list of keys and a dictionary of tensorized
                data.

        Examples:
            >>> collate_fn = TTS2Task.build_collate_fn(args, train=True)
            >>> batch = collate_fn(data_samples)
            >>> print(batch)

        Note:
            The function uses the `CommonCollateFn` for the actual implementation,
            which handles padding and tensor conversion.
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
                Build a preprocessing function based on the provided arguments.

        This function constructs a preprocessing function that will be used to
        process the input data for the TTS task. It leverages the `CommonPreprocessor`
        if preprocessing is enabled through the arguments.

        Args:
            args (argparse.Namespace): The parsed arguments containing configurations
                for preprocessing.
            train (bool): A flag indicating whether the function is being built for
                training or not.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A preprocessing function that takes a string and a dictionary of
                numpy arrays as input and returns a dictionary of numpy arrays.
                Returns None if preprocessing is not enabled.

        Examples:
            To create a preprocessing function for training:

            >>> args = argparse.Namespace(use_preprocessor=True, src_token_type='phn', ...)
            >>> preprocess_fn = TTS2Task.build_preprocess_fn(args, train=True)

            To create a preprocessing function for evaluation:

            >>> args = argparse.Namespace(use_preprocessor=False, ...)
            >>> preprocess_fn = TTS2Task.build_preprocess_fn(args, train=False)

        Note:
            Ensure that the `use_preprocessor` argument is set to True to enable
            preprocessing; otherwise, the function will return None.
        """
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.src_token_type,
                token_list=args.src_token_list,
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
                Defines the required data names for the TTS2Task class.

        This method specifies the necessary data names that are required for
        training and inference. The data names differ based on whether the
        function is called in training mode or inference mode.

        Args:
            train (bool): Indicates if the function is called in training mode.
                Default is True.
            inference (bool): Indicates if the function is called in inference
                mode. Default is False.

        Returns:
            Tuple[str, ...]: A tuple of required data names. The data names
            include:
                - For training (inference=False):
                    ("text", "speech", "discrete_speech")
                - For inference (inference=True):
                    ("text",)

        Note:
            - The "speech" data is used for on-the-fly feature extraction like
            pitch and energy.
            - The "discrete_speech" is mainly used for predicting the target.

        Examples:
            >>> required_data_names()
            ('text', 'speech', 'discrete_speech')

            >>> required_data_names(inference=True)
            ('text',)
        """
        # Note (Jinchuan): We need both speech and discrete_speech
        # speech is for on-the-fly feature extraction like pitch & energy
        # discrete_speech is mainly for the predicting target.
        # We can later make the speech optional so that the non-text info
        # can be injected though reference speech clips.
        if not inference:
            retval = ("text", "speech", "discrete_speech")
        else:
            # Inference mode
            retval = ("text",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
                Text-to-speech task.

        This class implements the TTS2Task for handling text-to-speech tasks,
        including argument parsing, data processing, and model building.

        Attributes:
            num_optimizers (int): Number of optimizers to use.
            class_choices_list (list): List of class choices for various components.
            trainer (Trainer): The trainer class used for training and evaluation.

        Args:
            parser (argparse.ArgumentParser): Argument parser to add task-specific
                arguments.

        Returns:
            Callable: A function that collates data for training or evaluation.

        Yields:
            Optional[Callable]: A function that preprocesses data based on the
                provided arguments.

        Raises:
            RuntimeError: If the source or target token list is not a string or
                dictionary.

        Examples:
            # Adding task arguments
            parser = argparse.ArgumentParser()
            TTS2Task.add_task_arguments(parser)

            # Building a collate function
            collate_fn = TTS2Task.build_collate_fn(args, train=True)

            # Building a preprocess function
            preprocess_fn = TTS2Task.build_preprocess_fn(args, train=True)

            # Required data names
            required_data = TTS2Task.required_data_names(train=True)

            # Optional data names
            optional_data = TTS2Task.optional_data_names(train=True)

            # Building the model
            model = TTS2Task.build_model(args)

            # Building a vocoder from a file
            vocoder = TTS2Task.build_vocoder_from_file(vocoder_config_file='path/to/config',
                                                        vocoder_file='path/to/vocoder')
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
    def build_model(cls, args: argparse.Namespace) -> ESPnetTTS2Model:
        """
                Builds and returns an instance of the ESPnetTTS2Model using the provided
        arguments.

        This method constructs a text-to-speech model by first processing the
        source and target token lists, extracting discrete features, and
        configuring various components like pitch and energy extraction. The
        resulting model is an instance of ESPnetTTS2Model.

        Args:
            args (argparse.Namespace): The parsed command-line arguments containing
                configurations for model building, including paths to token lists
                and various extraction settings.

        Returns:
            ESPnetTTS2Model: An instance of the ESPnetTTS2Model configured with
                the specified parameters.

        Raises:
            RuntimeError: If the source or target token lists are not provided in
                the expected format (must be str or dict).

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument("--src_token_list", type=str, default="src_tokens.txt")
            >>> parser.add_argument("--tgt_token_list", type=str, default="tgt_tokens.txt")
            >>> args = parser.parse_args()
            >>> model = TTS2Task.build_model(args)

        Note:
            Ensure that the token lists specified in the arguments exist and are
            properly formatted to avoid runtime errors.

        Todo:
            - Extend functionality to support additional feature extraction methods.
        """
        if isinstance(args.src_token_list, str):
            with open(args.src_token_list, encoding="utf-8") as f:
                src_token_list = [line[0] + line[1:].rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.src_token_list = src_token_list.copy()
        elif isinstance(args.src_token_list, (tuple, list)):
            src_token_list = args.src_token_list.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        if isinstance(args.tgt_token_list, str):
            with open(args.tgt_token_list, encoding="utf-8") as f:
                tgt_token_list = [line[0] + line[1:].rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.tgt_token_list = tgt_token_list.copy()
        elif isinstance(args.tgt_token_list, (tuple, list)):
            tgt_token_list = args.tgt_token_list.copy()
        else:
            raise RuntimeError("tgt_token_list must be str or dict")

        vocab_size = len(src_token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
        tgt_vocab_size = len(tgt_token_list)
        logging.info(f"Target Vocabulary size: {tgt_vocab_size}")

        # 1. discrete feature extraction
        discrete_feats_extract_class = discrete_feats_extractor_choices.get_class(
            args.discrete_feats_extract
        )
        discrete_feats_extract = discrete_feats_extract_class(
            **args.discrete_feats_extract_conf
        )

        # 3. TTS
        tts_class = tts_choices.get_class(args.tts)
        tts = tts_class(idim=vocab_size, odim=tgt_vocab_size, **args.tts_conf)

        # 4. Extra components
        pitch_extract = None
        energy_extract = None
        pitch_normalize = None
        energy_normalize = None
        if getattr(args, "pitch_extract", None) is not None:
            pitch_extract_class = pitch_extractor_choices.get_class(args.pitch_extract)
            if args.pitch_extract_conf.get("reduction_factor", None) is not None:
                assert args.pitch_extract_conf.get(
                    "reduction_factor", None
                ) == args.tts_conf.get("reduction_factor", 1)
            else:
                args.pitch_extract_conf["reduction_factor"] = args.tts_conf.get(
                    "reduction_factor", 1
                )
            pitch_extract = pitch_extract_class(**args.pitch_extract_conf)
        if getattr(args, "energy_extract", None) is not None:
            if args.energy_extract_conf.get("reduction_factor", None) is not None:
                assert args.energy_extract_conf.get(
                    "reduction_factor", None
                ) == args.tts_conf.get("reduction_factor", 1)
            else:
                args.energy_extract_conf["reduction_factor"] = args.tts_conf.get(
                    "reduction_factor", 1
                )
            energy_extract_class = energy_extractor_choices.get_class(
                args.energy_extract
            )
            energy_extract = energy_extract_class(**args.energy_extract_conf)
        if getattr(args, "pitch_normalize", None) is not None:
            pitch_normalize_class = pitch_normalize_choices.get_class(
                args.pitch_normalize
            )
            pitch_normalize = pitch_normalize_class(**args.pitch_normalize_conf)
        if getattr(args, "energy_normalize", None) is not None:
            energy_normalize_class = energy_normalize_choices.get_class(
                args.energy_normalize
            )
            energy_normalize = energy_normalize_class(**args.energy_normalize_conf)

        # 5. Build model
        model = ESPnetTTS2Model(
            discrete_feats_extract=discrete_feats_extract,
            pitch_extract=pitch_extract,
            energy_extract=energy_extract,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            tts=tts,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_vocoder_from_file(
        cls,
        vocoder_config_file: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        model: Optional[ESPnetTTS2Model] = None,
        device: str = "cpu",
    ):
        """
                Builds a vocoder from a given configuration file and model file.

        This method is responsible for constructing a vocoder instance based on the
        provided vocoder configuration and model file. It currently supports vocoder
        models trained with Parallel WaveGAN.

        Args:
            vocoder_config_file (Union[Path, str], optional): Path to the vocoder
                configuration file. If not provided, defaults to None.
            vocoder_file (Union[Path, str]): Path to the vocoder model file. This
                argument is required and must not be None.
            model (Optional[ESPnetTTS2Model], optional): An instance of the TTS2 model
                to be used with the vocoder. Defaults to None.
            device (str, optional): The device to which the vocoder should be moved
                (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            ParallelWaveGANPretrainedVocoder: An instance of the vocoder model
            ready for inference.

        Raises:
            AssertionError: If `vocoder_file` is None.
            ValueError: If the file format of `vocoder_file` is not supported.

        Examples:
            # Building a vocoder using a vocoder config and model file
            vocoder = TTS2Task.build_vocoder_from_file(
                vocoder_config_file='path/to/vocoder_config.yaml',
                vocoder_file='path/to/vocoder_model.pkl'
            )

            # Building a vocoder with a specified device
            vocoder = TTS2Task.build_vocoder_from_file(
                vocoder_config_file='path/to/vocoder_config.yaml',
                vocoder_file='path/to/vocoder_model.pkl',
                device='cuda'
            )
        """
        # Build vocoder
        assert vocoder_file is not None, "TTS2 model must have a vocoder."

        if str(vocoder_file).endswith(".pkl"):
            # If the extension is ".pkl", the model is trained with parallel_wavegan
            vocoder = ParallelWaveGANPretrainedVocoder(
                vocoder_file, vocoder_config_file
            )
            return vocoder.to(device)

        else:
            raise ValueError(f"{vocoder_file} is not supported format.")
