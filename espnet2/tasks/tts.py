"""Text-to-speech task."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import typechecked

from espnet2.gan_tts.jets import JETS
from espnet2.gan_tts.joint import JointText2Wav
from espnet2.gan_tts.vits import VITS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.espnet_model import ESPnetTTSModel
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.tts.prodiff import ProDiff
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

feats_extractor_choices = ClassChoices(
    "feats_extract",
    classes=dict(
        fbank=LogMelFbank,
        spectrogram=LogSpectrogram,
        linear_spectrogram=LinearSpectrogram,
    ),
    type_check=AbsFeatsExtract,
    default="fbank",
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
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(global_mvn=GlobalMVN),
    type_check=AbsNormalize,
    default="global_mvn",
    optional=True,
)
pitch_normalize_choices = ClassChoices(
    "pitch_normalize",
    classes=dict(global_mvn=GlobalMVN),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
energy_normalize_choices = ClassChoices(
    "energy_normalize",
    classes=dict(global_mvn=GlobalMVN),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
tts_choices = ClassChoices(
    "tts",
    classes=dict(
        tacotron2=Tacotron2,
        transformer=Transformer,
        fastspeech=FastSpeech,
        fastspeech2=FastSpeech2,
        prodiff=ProDiff,
        # NOTE(kan-bayashi): available only for inference
        vits=VITS,
        joint_text2wav=JointText2Wav,
        jets=JETS,
    ),
    type_check=AbsTTS,
    default="tacotron2",
)


class TTSTask(AbsTask):
    """
        TTSTask is a class that implements a text-to-speech (TTS) task for training
    and evaluating TTS models.

    This class provides methods to handle TTS-specific functionalities such as
    building models, processing data, and defining the required and optional
    data names for training and inference.

    Attributes:
        num_optimizers (int): The number of optimizers to be used in the task.
        class_choices_list (list): A list of choices for various components
            like feature extraction, normalization, and TTS models.
        trainer (Trainer): The trainer class used for training the TTS model.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add task
            related arguments.

    Returns:
        Callable: A callable function that collates the input data.

    Raises:
        RuntimeError: If the token list is not of the expected type.

    Examples:
        # To add task-specific arguments
        TTSTask.add_task_arguments(parser)

        # To build a model
        model = TTSTask.build_model(args)

        # To get required data names for training
        required_names = TTSTask.required_data_names(train=True)

        # To build vocoder from a file
        vocoder = TTSTask.build_vocoder_from_file(vocoder_config_file='path/to/config.yaml')

    Note:
        This class inherits from AbsTask and requires the implementation of
        abstract methods defined in the parent class.

    Todo:
        - Expand the functionality to support more advanced TTS models.
    """

    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

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

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
                Adds task-related arguments to the argument parser for the TTS task.

        This method configures the argument parser with the necessary arguments
        for the text-to-speech task. It defines both task-specific and preprocessing
        arguments, which are essential for training and evaluating TTS models.

        Args:
            cls: The class itself, typically used in class methods.
            parser (argparse.ArgumentParser): The argument parser to which the
                arguments will be added.

        Examples:
            parser = argparse.ArgumentParser()
            TTSTask.add_task_arguments(parser)

            # This will add the following arguments to the parser:
            # --token_list
            # --odim
            # --model_conf
            # --use_preprocessor
            # --token_type
            # --bpemodel
            # --non_linguistic_symbols
            # --cleaner
            # --g2p
            # and others defined in class_choices_list.

        Note:
            The arguments for feature extraction, normalization, and TTS models
            are added from the `class_choices_list`.

        Raises:
            None: This method does not raise any exceptions.
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
            default=get_default_kwargs(ESPnetTTSModel),
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
        Build a collate function for batching data.

        This method constructs a collate function that is used to process
        batches of data. The collate function will handle padding and
        formatting of input data into a form suitable for training or
        evaluation.

        Args:
            args (argparse.Namespace): The arguments parsed from the command line.
            train (bool): Indicates whether the function is being used for
                training or evaluation.

        Returns:
            Callable: A collate function that takes a collection of data
            samples and returns a tuple containing a list of keys and a
            dictionary of tensors.

        Examples:
            >>> collate_fn = TTSTask.build_collate_fn(args, train=True)
            >>> batch = collate_fn([("sample1", {"feature": np.array([1, 2])}),
            ...                      ("sample2", {"feature": np.array([3, 4])})])
            >>> print(batch)
            (['sample1', 'sample2'], {'feature': tensor([[1, 2], [3, 4]])})

        Note:
            This function uses CommonCollateFn to handle padding and
            data organization.
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
        Builds a preprocessing function based on the given arguments.

        This function returns a callable that applies preprocessing to input text
        and its corresponding feature dictionary if preprocessing is enabled. If
        preprocessing is not required, it returns None.

        Args:
            cls: The class type for method resolution.
            args (argparse.Namespace): The parsed arguments containing
                preprocessing options.
            train (bool): Indicates whether the function is for training or
                inference.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A function that takes a string and a dictionary of features,
                applying the specified preprocessing steps, or None if
                preprocessing is disabled.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(use_preprocessor=True, token_type='phn', ...)
            >>> preprocess_fn = TTSTask.build_preprocess_fn(args, train=True)
            >>> features = preprocess_fn("sample text", {"feature_key": np.array([1, 2])})

        Note:
            The preprocessing function may include tokenization, text cleaning,
            and conversion to phonemes based on the specified arguments.

        Todo:
            Consider adding more preprocessing options in future iterations.
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
                Text-to-speech task.

        This class provides functionalities to manage the text-to-speech (TTS) task,
        including argument parsing, data handling, and model building.

        Attributes:
            num_optimizers (int): The number of optimizers used for training.
            class_choices_list (List[ClassChoices]): A list of available class choices
                for features extraction, normalization, TTS, pitch extraction, and
                energy extraction.

        Args:
            train (bool): Indicates whether the task is in training mode. Defaults to
                True.
            inference (bool): Indicates whether the task is in inference mode.
                Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple of required data names based on the mode.

        Examples:
            >>> TTSTask.required_data_names(train=True, inference=False)
            ('text', 'speech')

            >>> TTSTask.required_data_names(train=True, inference=True)
            ('text',)

        Note:
            In inference mode, only 'text' is required.

        Todo:
            - Expand the functionality to include more data types as needed.
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
            Returns a tuple of optional data names used in the text-to-speech task.

        The optional data names vary based on whether the task is in training or
        inference mode. During training, the following optional data names can be
        utilized: 'spembs', 'durations', 'pitch', 'energy', 'sids', 'lids'. In
        inference mode, 'spembs', 'speech', 'durations', 'pitch', 'energy',
        'sids', and 'lids' can be used.

        Args:
            train (bool): Indicates if the task is in training mode. Defaults to
                True.
            inference (bool): Indicates if the task is in inference mode.
                Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of optional data.

        Examples:
            >>> TTSTask.optional_data_names(train=True, inference=False)
            ('spembs', 'durations', 'pitch', 'energy', 'sids', 'lids')

            >>> TTSTask.optional_data_names(train=True, inference=True)
            ('spembs', 'speech', 'durations', 'pitch', 'energy', 'sids', 'lids')
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
    def build_model(cls, args: argparse.Namespace) -> ESPnetTTSModel:
        """
                Builds and returns an instance of the ESPnetTTSModel based on the provided
        arguments.

        This method configures various components of the text-to-speech (TTS) model,
        including feature extraction, normalization, and the TTS architecture itself.
        It also handles the token list for input processing.

        Args:
            args (argparse.Namespace): A namespace object containing the necessary
                configurations for building the model, including:
                - token_list (str or list): Path to a file or a list of tokens mapping
                  int-id to token.
                - odim (int, optional): The number of dimensions of the output feature.
                - feats_extract (str): Type of feature extractor to use.
                - feats_extract_conf (dict): Configuration for the feature extractor.
                - normalize (str, optional): Type of normalization to apply.
                - normalize_conf (dict, optional): Configuration for normalization.
                - tts (str): Type of TTS model to use.
                - tts_conf (dict): Configuration for the TTS model.
                - pitch_extract (str, optional): Type of pitch extractor to use.
                - pitch_extract_conf (dict, optional): Configuration for pitch extractor.
                - energy_extract (str, optional): Type of energy extractor to use.
                - energy_extract_conf (dict, optional): Configuration for energy extractor.
                - pitch_normalize (str, optional): Type of pitch normalization to apply.
                - pitch_normalize_conf (dict, optional): Configuration for pitch
                  normalization.
                - energy_normalize (str, optional): Type of energy normalization to apply.
                - energy_normalize_conf (dict, optional): Configuration for energy
                  normalization.
                - model_conf (dict, optional): Additional keyword arguments for the
                  model class.

        Returns:
            ESPnetTTSModel: An instance of the configured ESPnetTTSModel.

        Raises:
            RuntimeError: If token_list is not a string or a list.

        Examples:
            # Building a model with a specified token list file
            args = argparse.Namespace(
                token_list='path/to/token_list.txt',
                odim=None,
                feats_extract='fbank',
                feats_extract_conf={'param1': value1},
                normalize='global_mvn',
                tts='tacotron2',
                tts_conf={'param2': value2},
            )
            model = TTSTask.build_model(args)

            # Building a model with a predefined list of tokens
            args = argparse.Namespace(
                token_list=['a', 'b', 'c'],
                odim=80,
                feats_extract='spectrogram',
                feats_extract_conf={'param1': value1},
                normalize=None,
                tts='fastspeech2',
                tts_conf={'param2': value2},
            )
            model = TTSTask.build_model(args)
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
        model = ESPnetTTSModel(
            feats_extract=feats_extract,
            pitch_extract=pitch_extract,
            energy_extract=energy_extract,
            normalize=normalize,
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
        model: Optional[ESPnetTTSModel] = None,
        device: str = "cpu",
    ):
        """
                Builds a vocoder from the specified configuration file or file path.

        This method constructs a vocoder based on the provided vocoder configuration
        file and vocoder model file. If no vocoder model file is provided, it defaults
        to using the Griffin-Lim vocoder. If the vocoder model file has a ".pkl"
        extension, it is assumed to be a trained Parallel WaveGAN model.

        Attributes:
            vocoder_config_file (Union[Path, str]): Path to the vocoder configuration file.
            vocoder_file (Union[Path, str]): Path to the vocoder model file.
            model (Optional[ESPnetTTSModel]): The TTS model used for extracting features.
            device (str): The device to load the model onto (default: "cpu").

        Args:
            vocoder_config_file (Union[Path, str]): The path to the vocoder config file.
            vocoder_file (Union[Path, str]): The path to the vocoder model file.
            model (Optional[ESPnetTTSModel]): The TTS model to extract features from.
            device (str): The device to load the model onto (default: "cpu").

        Returns:
            Optional[Union[Spectrogram2Waveform, ParallelWaveGANPretrainedVocoder]]:
            The constructed vocoder or None if the vocoder could not be built.

        Raises:
            ValueError: If the vocoder_file format is not supported.

        Examples:
            # Build vocoder using a configuration file and model file
            vocoder = TTSTask.build_vocoder_from_file(
                vocoder_config_file="path/to/vocoder_config.yaml",
                vocoder_file="path/to/vocoder_model.pkl",
                model=my_tts_model,
                device="cuda"
            )

            # Build vocoder using only a configuration file, defaults to Griffin-Lim
            vocoder = TTSTask.build_vocoder_from_file(
                vocoder_config_file="path/to/vocoder_config.yaml",
                model=my_tts_model
            )
        """
        # Build vocoder
        if vocoder_file is None:
            # If vocoder file is not provided, use griffin-lim as a vocoder
            vocoder_conf = {}
            if vocoder_config_file is not None:
                vocoder_config_file = Path(vocoder_config_file)
                with vocoder_config_file.open("r", encoding="utf-8") as f:
                    vocoder_conf = yaml.safe_load(f)
            if model.feats_extract is not None:
                vocoder_conf.update(model.feats_extract.get_parameters())
            if (
                "n_fft" in vocoder_conf
                and "n_shift" in vocoder_conf
                and "fs" in vocoder_conf
            ):
                return Spectrogram2Waveform(**vocoder_conf)
            else:
                logging.warning("Vocoder is not available. Skipped its building.")
                return None

        elif str(vocoder_file).endswith(".pkl"):
            # If the extension is ".pkl", the model is trained with parallel_wavegan
            vocoder = ParallelWaveGANPretrainedVocoder(
                vocoder_file, vocoder_config_file
            )
            return vocoder.to(device)

        else:
            raise ValueError(f"{vocoder_file} is not supported format.")
