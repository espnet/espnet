"""Voice conversion task."""

import argparse
import logging
import yaml

from pathlib import Path
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.vc.espnet_model import ESPnetVCModel
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.vc.abs_vc import AbsVC
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.vc.transformer import Transformer
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

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
input_feats_extractor_choices = ClassChoices(
    "input_feats_extract",
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
input_normalize_choices = ClassChoices(
    "input_normalize",
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
vc_choices = ClassChoices(
    "vc",
    classes=dict(
        transformer=Transformer,
    ),
    type_check=AbsVC,
    default="transformer",
)


class VCTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --feats_extractor and --feats_extract_conf
        feats_extractor_choices,
        # --input_feats_extractor and --input_feats_extract_conf
        input_feats_extractor_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --input_normalize and --input_normalize_conf
        input_normalize_choices,
        # --vc and --vc_conf
        vc_choices,
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

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--idim",
            type=int_or_none,
            default=None,
            help="The number of dimensions in each input feature",
        )
        group.add_argument(
            "--odim",
            type=int_or_none,
            default=None,
            help="The number of dimensions in each output feature",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetVCModel),
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
                token_type=args.token_type,
                token_list=args.token_list,
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
        if not inference:
            retval = ("in_speech", "out_speech")
        else:
            # Inference mode
            retval = ("in_speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("text", "spembs", "durations", "pitch", "energy", "sids", "lids")
        else:
            # Inference mode
            retval = ("text", "spembs", "out_speech", "durations", "sids", "lids")
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetVCModel:
        assert check_argument_types()
        if args.token_list is not None:
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
        if args.idim is None:
            # Extract features in the model
            input_feats_extract_class = input_feats_extractor_choices.get_class(
                args.input_feats_extract
            )
            input_feats_extract = input_feats_extract_class(
                **args.input_feats_extract_conf
            )
            idim = input_feats_extract.output_size()
        else:
            # Give features from data-loader
            args.input_feats_extract = None
            args.input_feats_extract_conf = None
            input_feats_extract = None
            idim = args.idim

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None
        if args.input_normalize is not None:
            input_normalize_class = input_normalize_choices.get_class(
                args.input_normalize
            )
            input_normalize = input_normalize_class(**args.input_normalize_conf)
        else:
            input_normalize = None

        # 3. VC
        vc_class = vc_choices.get_class(args.vc)
        vc = vc_class(idim=idim, odim=odim, **args.vc_conf)

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
                ) == args.vc_conf.get("reduction_factor", 1)
            else:
                args.pitch_extract_conf["reduction_factor"] = args.vc_conf.get(
                    "reduction_factor", 1
                )
            pitch_extract = pitch_extract_class(**args.pitch_extract_conf)
        if getattr(args, "energy_extract", None) is not None:
            if args.energy_extract_conf.get("reduction_factor", None) is not None:
                assert args.energy_extract_conf.get(
                    "reduction_factor", None
                ) == args.vc_conf.get("reduction_factor", 1)
            else:
                args.energy_extract_conf["reduction_factor"] = args.vc_conf.get(
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
        model = ESPnetVCModel(
            feats_extract=feats_extract,
            input_feats_extract=input_feats_extract,
            pitch_extract=pitch_extract,
            energy_extract=energy_extract,
            normalize=normalize,
            input_normalize=input_normalize,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            vc=vc,
            **args.model_conf,
        )
        assert check_return_type(model)
        return model

    @classmethod
    def build_vocoder_from_file(
        cls,
        vocoder_config_file: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        model: Optional[ESPnetVCModel] = None,
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
