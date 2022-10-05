"""Singing-voice-synthesis task."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import check_argument_types, check_return_type

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

# TODO(Yuning): Models to be added
# from espnet2.svs.encoder_decoder.transformer.transformer import Transformer
# from espnet2.svs.mlp_singer.mlp_singer import MLPSinger
# from espnet2.svs.glu_transformer.glu_transformer import GLU_Transformer
# from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing
# from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing_noDP
# from espnet2.svs.bytesing.bytesing import ByteSing
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
        # mlp=MLPSinger,
        # xiaoice=XiaoiceSing,
        # xiaoice_noDP=XiaoiceSing_noDP,
        vits=VITS,
    ),
    type_check=AbsSVS,
    default="transformer",
)


class SVSTask(AbsTask):
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
    ) -> Optional[Callable[[str, Dict[str, np.array], float], Dict[str, np.ndarray]]]:
        assert check_argument_types()
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
            )
        else:
            retval = None
        # FIXME (jiatong): sometimes checking is not working here
        # assert check_return_type(retval)
        return retval

    # TODO(Yuning): check new names
    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("text", "singing", "midi", "label")
        else:
            # Inference mode
            retval = ("text", "midi", "label")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("spembs", "durations", "pitch", "energy", "sids", "lids")
        else:
            # Inference mode
            retval = ("spembs", "singing", "durations", "sids", "lids")
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetSVSModel:
        assert check_argument_types()
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
            tempo_extract=score_feats_extract,
            beat_extract=score_feats_extract,
            energy_extract=energy_extract,
            normalize=normalize,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            svs=svs,
            **args.model_conf,
        )
        assert check_return_type(model)
        return model

    @classmethod
    def build_vocoder_from_file(
        cls,
        vocoder_config_file: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        model: Optional[ESPnetSVSModel] = None,
        device: str = "cpu",
    ):

        logging.info(f"vocoder_config_file: {vocoder_config_file}")
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
