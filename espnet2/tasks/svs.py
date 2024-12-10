"""Singing-voice-synthesis task."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import typechecked

from espnet2.gan_svs.joint import JointScore2Wav
from espnet2.gan_svs.vits import VITS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.svs.abs_svs import AbsSVS
from espnet2.svs.espnet_model import ESPnetSVSModel
from espnet2.svs.feats_extract.score_feats_extract import (
    FrameScoreFeats,
    SyllableScoreFeats,
)
from espnet2.svs.naive_rnn.naive_rnn import NaiveRNN
from espnet2.svs.naive_rnn.naive_rnn_dp import NaiveRNNDP

# TODO(Yuning): Models to be added
from espnet2.svs.singing_tacotron.singing_tacotron import singing_tacotron
from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing

# from espnet2.svs.encoder_decoder.transformer.transformer import Transformer
# from espnet2.svs.mlp_singer.mlp_singer import MLPSinger
# from espnet2.svs.glu_transformer.glu_transformer import GLU_Transformer
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import SVSPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.tts.feats_extract.ying import Ying

# from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing_noDP
# from espnet2.svs.bytesing.bytesing import ByteSing
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

# TODO(Yuning): Add singing augmentation

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

score_feats_extractor_choices = ClassChoices(
    "score_feats_extract",
    classes=dict(
        frame_score_feats=FrameScoreFeats, syllable_score_feats=SyllableScoreFeats
    ),
    type_check=AbsFeatsExtract,
    default="frame_score_feats",
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
ying_extractor_choices = ClassChoices(
    "ying_extract",
    classes=dict(ying=Ying),
    type_check=AbsFeatsExtract,
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
svs_choices = ClassChoices(
    "svs",
    classes=dict(
        # transformer=Transformer,
        # glu_transformer=GLU_Transformer,
        # bytesing=ByteSing,
        naive_rnn=NaiveRNN,
        naive_rnn_dp=NaiveRNNDP,
        xiaoice=XiaoiceSing,
        # xiaoice_noDP=XiaoiceSing_noDP,
        vits=VITS,
        joint_score2wav=JointScore2Wav,
        # mlp=MLPSinger,
        singing_tacotron=singing_tacotron,
    ),
    type_check=AbsSVS,
    default="naive_rnn",
)


class SVSTask(AbsTask):
    """
        Singing Voice Synthesis (SVS) task class for managing the training and
    evaluation of singing voice synthesis models.

    This class extends the abstract task class `AbsTask` and provides
    methods for adding task-specific arguments, building models,
    collating data, and preprocessing inputs.

    Attributes:
        num_optimizers (int): Number of optimizers to be used in training.
        class_choices_list (list): List of class choices for various
            components in the SVS task.
        trainer (Trainer): Trainer class used for training the models.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser):
            Adds task-specific arguments to the provided argument parser.

        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Builds a collate function for batching data during training or
            evaluation.

        build_preprocess_fn(args: argparse.Namespace, train: bool) ->
            Optional[Callable]:
            Builds a preprocessing function for input data based on the
            provided arguments.

        required_data_names(train: bool = True, inference: bool = False)
            -> Tuple[str, ...]:
            Returns the names of the required data for training or inference.

        optional_data_names(train: bool = True, inference: bool = False)
            -> Tuple[str, ...]:
            Returns the names of the optional data for training or inference.

        build_model(args: argparse.Namespace) -> ESPnetSVSModel:
            Constructs the SVS model based on the provided arguments.

        build_vocoder_from_file(vocoder_config_file: Union[Path, str] = None,
            vocoder_file: Union[Path, str] = None,
            model: Optional[ESPnetSVSModel] = None,
            device: str = "cpu"):
            Builds a vocoder from the provided configuration and model.

    Examples:
        # Example usage of adding task arguments
        parser = argparse.ArgumentParser()
        SVSTask.add_task_arguments(parser)

        # Example of building a model
        args = parser.parse_args()
        model = SVSTask.build_model(args)

    Note:
        Ensure that the necessary dependencies for SVS are installed and
        properly configured.

    Todo:
        - Add singing augmentation features.
        - Include more models in the SVS choices.
    """

    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --score_extractor and --score_extractor_conf
        score_feats_extractor_choices,
        # --feats_extractor and --feats_extractor_conf
        feats_extractor_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --svs and --svs_conf
        svs_choices,
        # --pitch_extract and --pitch_extract_conf
        pitch_extractor_choices,
        # --pitch_normalize and --pitch_normalize_conf
        pitch_normalize_choices,
        # --ying_extract and --ying_extract_conf
        ying_extractor_choices,
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
        Add task-specific arguments to the argument parser.

        This method adds various command-line arguments related to the
        singing voice synthesis (SVS) task to the provided argument
        parser. The arguments include configurations for tokenization,
        feature extraction, normalization, and the model.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which
                the task-specific arguments will be added.

        Note:
            The method uses an underscore (_) instead of a hyphen (-) to
            avoid confusion in argument naming.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> SVSTask.add_task_arguments(parser)
            >>> args = parser.parse_args()

        Raises:
            ValueError: If the argument configurations are invalid.
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
            default=get_default_kwargs(ESPnetSVSModel),
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
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=[
                None,
                "g2p_en",
                "g2p_en_no_space",
                "pyopenjtalk",
                "pyopenjtalk_kana",
                "pyopenjtalk_accent",
                "pyopenjtalk_accent_with_pause",
                "pypinyin_g2p",
                "pypinyin_g2p_phone",
                "pypinyin_g2p_phone_without_prosody",
                "espeak_ng_arabic",
            ],
            default=None,
            help="Specify g2p method if --token_type=phn",
        )

        parser.add_argument(
            "--fs",
            type=int,
            default=24000,  # BUG: another fs in feats_extract_conf
            help="sample rate",
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
            Builds a collate function for the SVSTask.

        This method constructs a callable that collates a batch of data during
        training or evaluation. It pads sequences appropriately and prepares
        tensors for input into the model.

        Args:
            args (argparse.Namespace): The parsed command line arguments.
            train (bool): A flag indicating whether the function is being used
                          for training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]],
                      Tuple[List[str], Dict[str, torch.Tensor]]]]:
            A collate function that processes a batch of data.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace()
            >>> collate_fn = SVSTask.build_collate_fn(args, train=True)
            >>> batch = [("file1", {"feature": np.array([1, 2, 3])}),
                         ("file2", {"feature": np.array([4, 5])})]
            >>> collated_data = collate_fn(batch)
            >>> print(collated_data)
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
        Builds a preprocessing function for the singing voice synthesis task.

        This method creates a preprocessing function based on the specified
        arguments. If preprocessing is enabled, it initializes an instance of
        `SVSPreprocessor` with the given configuration parameters. If
        preprocessing is not enabled, it returns `None`.

        Args:
            cls: The class reference.
            args (argparse.Namespace): The arguments namespace containing
                configuration parameters.
            train (bool): Indicates whether the function is being built for
                training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]]:
                A preprocessing function if `args.use_preprocessor` is `True`,
                otherwise `None`.

        Examples:
            >>> args = argparse.Namespace(use_preprocessor=True, token_type='phn',
            ...                            token_list='path/to/token_list.txt',
            ...                            bpemodel=None, non_linguistic_symbols=None,
            ...                            cleaner=None, g2p=None, fs=24000,
            ...                            feats_extract_conf={"hop_length": 256})
            >>> preprocess_fn = SVSTask.build_preprocess_fn(args, train=True)
            >>> print(preprocess_fn)
            <function SVSPreprocessor at 0x...>
        """
        if args.use_preprocessor:
            retval = SVSPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                fs=args.fs,
                hop_length=args.feats_extract_conf["hop_length"],
            )
        else:
            retval = None

        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Get the required data names for the singing voice synthesis task.

        This method returns a tuple of required data names based on whether the
        task is in training or inference mode. The data names vary depending on
        the mode.

        Args:
            train (bool): Indicates if the task is in training mode. Default is
                True.
            inference (bool): Indicates if the task is in inference mode. Default
                is False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of the required data.

        Examples:
            >>> SVSTask.required_data_names(train=True, inference=False)
            ('text', 'singing', 'score', 'label')

            >>> SVSTask.required_data_names(train=False, inference=True)
            ('text', 'score', 'label')

        Note:
            The required data names differ when the task is in inference mode,
            where 'singing' is not required.
        """
        if not inference:
            retval = ("text", "singing", "score", "label")
        else:
            # Inference mode
            retval = ("text", "score", "label")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
                Optional data names for the singing voice synthesis task.

        This method provides a list of optional data names that can be used during
        training or inference for the singing voice synthesis task. The returned
        names depend on the mode (training or inference) and can include various
        features and attributes related to the synthesis process.

        Args:
            train (bool): A flag indicating whether the task is in training mode.
                Defaults to True.
            inference (bool): A flag indicating whether the task is in inference
                mode. Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of optional data.

        Examples:
            # In training mode
            optional_data = SVSTask.optional_data_names(train=True)
            print(optional_data)
            # Output: ('spembs', 'durations', 'pitch', 'energy', 'sids', 'lids', 'feats', 'ying')

            # In inference mode
            optional_data = SVSTask.optional_data_names(inference=True)
            print(optional_data)
            # Output: ('spembs', 'singing', 'pitch', 'durations', 'sids', 'lids')
        """
        if not inference:
            retval = (
                "spembs",
                "durations",
                "pitch",
                "energy",
                "sids",
                "lids",
                "feats",
                "ying",
            )
        else:
            # Inference mode
            retval = ("spembs", "singing", "pitch", "durations", "sids", "lids")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSVSModel:
        """
            Build and return an instance of the ESPnetSVSModel based on the provided
        arguments.

        This method constructs a singing voice synthesis model by setting up
        various components such as feature extraction, normalization, and the
        model architecture itself. The model is built using configurations
        specified in the `args` parameter.

        Args:
            args (argparse.Namespace): The command line arguments containing model
            configuration settings, including token list, output dimension,
            feature extractors, normalization methods, and SVS architecture
            parameters.

        Returns:
            ESPnetSVSModel: An instance of the ESPnet singing voice synthesis
            model configured according to the specified arguments.

        Raises:
            RuntimeError: If `token_list` is neither a string nor a list/tuple.

        Examples:
            # Example of building a model with specified arguments
            args = argparse.Namespace(
                token_list='path/to/token_list.txt',
                odim=80,
                feats_extract='fbank',
                feats_extract_conf={'hop_length': 256},
                normalize='global_mvn',
                svs='naive_rnn',
                model_conf={}
            )
            model = SVSTask.build_model(args)
            print(model)

        Note:
            The `token_list` can be provided as a path to a file or directly as
            a list of tokens. The method handles both cases and will read the
            token list from the file if a string path is provided.

        Todo:
            - Add support for additional SVS models and configurations in future
            updates.
        """
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

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

        # 3. SVS
        svs_class = svs_choices.get_class(args.svs)
        svs = svs_class(idim=vocab_size, odim=odim, **args.svs_conf)

        # 4. Extra components
        score_feats_extract = None
        pitch_extract = None
        ying_extract = None
        energy_extract = None
        pitch_normalize = None
        energy_normalize = None
        logging.info(f"args:{args}")
        if getattr(args, "score_feats_extract", None) is not None:
            score_feats_extract_class = score_feats_extractor_choices.get_class(
                args.score_feats_extract
            )
            score_feats_extract = score_feats_extract_class(
                **args.score_feats_extract_conf
            )
        if getattr(args, "pitch_extract", None) is not None:
            pitch_extract_class = pitch_extractor_choices.get_class(args.pitch_extract)
            if args.pitch_extract_conf.get("reduction_factor", None) is not None:
                assert args.pitch_extract_conf.get(
                    "reduction_factor", None
                ) == args.svs_conf.get("reduction_factor", 1)
            else:
                args.pitch_extract_conf["reduction_factor"] = args.svs_conf.get(
                    "reduction_factor", 1
                )
            pitch_extract = pitch_extract_class(**args.pitch_extract_conf)
        if getattr(args, "ying_extract", None) is not None:
            ying_extract_class = ying_extractor_choices.get_class(
                args.ying_extract,
            )

            ying_extract = ying_extract_class(
                **args.ying_extract_conf,
            )
        if getattr(args, "energy_extract", None) is not None:
            if args.energy_extract_conf.get("reduction_factor", None) is not None:
                assert args.energy_extract_conf.get(
                    "reduction_factor", None
                ) == args.svs_conf.get("reduction_factor", 1)
            else:
                args.energy_extract_conf["reduction_factor"] = args.svs_conf.get(
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
        model = ESPnetSVSModel(
            text_extract=score_feats_extract,
            feats_extract=feats_extract,
            score_feats_extract=score_feats_extract,
            label_extract=score_feats_extract,
            pitch_extract=pitch_extract,
            ying_extract=ying_extract,
            duration_extract=score_feats_extract,
            energy_extract=energy_extract,
            normalize=normalize,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            svs=svs,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_vocoder_from_file(
        cls,
        vocoder_config_file: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        model: Optional[ESPnetSVSModel] = None,
        device: str = "cpu",
    ):
        """
                Builds a vocoder from the specified configuration and model files.

        This method allows for the construction of a vocoder based on a provided
        configuration file and an optional vocoder file. If the vocoder file is not
        provided, it defaults to using the Griffin-Lim algorithm for vocoding.

        Args:
            vocoder_config_file (Union[Path, str], optional): Path to the vocoder
                configuration file in YAML format. If provided, it will be used to
                initialize the vocoder parameters.
            vocoder_file (Union[Path, str], optional): Path to the vocoder model
                file. If this is a `.pkl` file, it is expected to be a trained model
                using Parallel WaveGAN.
            model (Optional[ESPnetSVSModel], optional): An instance of the SVS model
                from which to extract features if `vocoder_file` is not specified.
            device (str, optional): The device on which to load the vocoder model.
                Defaults to "cpu".

        Returns:
            Union[None, Spectrogram2Waveform, ParallelWaveGANPretrainedVocoder]:
                Returns an instance of the vocoder (either `Spectrogram2Waveform`
                or `ParallelWaveGANPretrainedVocoder`) if successfully built;
                otherwise returns None if the vocoder could not be constructed.

        Raises:
            ValueError: If the provided `vocoder_file` format is not supported.

        Examples:
            To build a vocoder using a configuration file:

            ```python
            vocoder = SVSTask.build_vocoder_from_file(
                vocoder_config_file='path/to/vocoder_config.yaml',
                model=my_svs_model
            )
            ```

            To build a vocoder using a pre-trained model file:

            ```python
            vocoder = SVSTask.build_vocoder_from_file(
                vocoder_file='path/to/vocoder_model.pkl',
                vocoder_config_file='path/to/vocoder_config.yaml'
            )
            ```

        Note:
            If no vocoder file is provided, Griffin-Lim will be used as a fallback
            vocoder.
        """
        logging.info(f"vocoder_file: {vocoder_file}")

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
