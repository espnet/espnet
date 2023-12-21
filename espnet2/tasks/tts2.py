"""Text-to-speech task."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import check_argument_types, check_return_type

# from espnet2.gan_tts.jets import JETS
# from espnet2.gan_tts.joint import JointText2Wav
# from espnet2.gan_tts.vits import VITS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.tts2.abs_tts2 import AbsTTS2
from espnet2.tts2.espnet_model import ESPnetTTS2Model

# from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts2.fastspeech2 import FastSpeech2Discrete
from espnet2.tts2.feats_extract.abs_feats_extract import AbsFeatsExtractDiscrete
from espnet2.tts2.feats_extract.identity import IdentityFeatureExtract

# TTS continuous feature extraction operators
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram

# from espnet2.tts.prodiff import ProDiff
# from espnet2.tts.tacotron2 import Tacotron2
# from espnet2.tts.transformer import Transformer
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
        fastspeech2=FastSpeech2Discrete,
    ),
    type_check=AbsTTS2,
    default="fastspeech2",
)


class TTS2Task(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --discrete_feats_extractor and --discrete_feats_extractor_conf
        discrete_feats_extractor_choices,
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
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
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
            "--odim",
            type=int_or_none,
            default=None,
            help="The number of dimension of output feature",
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
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=0,
            not_sequence=["spembs", "sids", "lids"],
        )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
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
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
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
    def build_model(cls, args: argparse.Namespace) -> ESPnetTTS2Model:
        assert check_argument_types()
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

        # 0. discrete feature extraction
        discrete_feats_extract_class = discrete_feats_extractor_choices.get_class(
            args.discrete_feats_extract
        )
        discrete_feats_extract = discrete_feats_extract_class(
            **args.discrete_feats_extract_conf
        )

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
            feats_extract=feats_extract,
            pitch_extract=pitch_extract,
            energy_extract=energy_extract,
            normalize=normalize,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            tts=tts,
            **args.model_conf,
        )
        assert check_return_type(model)
        return model

    @classmethod
    def build_vocoder_from_file(
        cls,
        vocoder_config_file: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        model: Optional[ESPnetTTS2Model] = None,
        device: str = "cpu",
    ):
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
