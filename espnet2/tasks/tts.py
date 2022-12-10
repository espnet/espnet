"""Text-to-speech task."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import check_argument_types, check_return_type

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
from espnet2.tts.espnet_model_md import ESPnetTTSMDModel
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
)
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder

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
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
asr_decoder_choices = ClassChoices(
    "asr_decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)
asr_encoder_choices = ClassChoices(
    "asr_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        branchformer=BranchformerEncoder,
    ),
    type_check=AbsEncoder,
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
        # NOTE(kan-bayashi): available only for inference
        vits=VITS,
        joint_text2wav=JointText2Wav,
        jets=JETS,
    ),
    type_check=AbsTTS,
    default="tacotron2",
)


class TTSTask(AbsTask):
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
        # --asr_decoder and --asr_decoder_conf
        asr_decoder_choices,
        # --encoder_mt and --encoder_mt_conf
        asr_encoder_choices,
        # --frontend and --frontend_conf
        frontend_choices,
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
            "--use_multidecoder",
            type=str2bool,
            default=False,
            help="Use multidecoder model",
        )
        group.add_argument(
            "--speech_attn",
            type=str2bool,
            default=False,
            help="Use speech attention model",
        )
        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
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
        # import pdb;pdb.set_trace()
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
    def build_model(cls, args: argparse.Namespace) -> Union[ESPnetTTSModel,ESPnetTTSMDModel]:
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
        # if args.input_size is None:
            # Extract features in the model
        frontend_class = frontend_choices.get_class(args.frontend)
        frontend = frontend_class(**args.frontend_conf)
        input_size = frontend.output_size()
        # else:
        #     # Give features from data-loader
        #     args.frontend = None
        #     args.frontend_conf = {}
        #     frontend = None
        #     input_size = args.input_size
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
        use_md_model = getattr(args, "use_multidecoder", False)
        speech_attn = getattr(args, "speech_attn", False)
        # import pdb;pdb.set_trace()
        if use_md_model:
            # 7. ASR decoder
            asr_encoder_class = asr_encoder_choices.get_class(
                args.asr_encoder
            )
            asr_encoder = asr_encoder_class(
                input_size=input_size, 
                **args.asr_encoder_conf,
            )
            encoder_output_size = asr_encoder.output_size()
            asr_decoder_class = asr_decoder_choices.get_class(
                args.asr_decoder
            )
            asr_decoder = asr_decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **args.asr_decoder_conf,
            )
            
            if token_list is not None:
                ctc = CTC(
                    odim=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.ctc_conf,
                )
            else:
                ctc = None
            # asr_decoder_output_size_bf_softmax = asr_decoder.output_size_bf_softmax()

            # # 8. Encoder MT
            # if getattr(args, "encoder_mt", None) is not None:
            #     encoder_mt_class = encoder_mt_choices.get_class(args.encoder_mt)
            #     encoder_mt = encoder_mt_class(input_size=asr_decoder_output_size_bf_softmax, **args.encoder_mt_conf)
            # else:
            #     encoder_mt= None

            model = ESPnetTTSMDModel(
                    vocab_size=vocab_size,
                    frontend=frontend,
                    asr_encoder=asr_encoder,
                    asr_decoder=asr_decoder,
                    ctc=ctc,
                    token_list=token_list,
                    speech_attn = speech_attn,
                    feats_extract=feats_extract,
                    pitch_extract=pitch_extract,
                    energy_extract=energy_extract,
                    normalize=normalize,
                    pitch_normalize=pitch_normalize,
                    energy_normalize=energy_normalize,
                    tts=tts,
                    **args.model_conf,
                )
        else:
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
        assert check_return_type(model)
        return model

    @classmethod
    def build_vocoder_from_file(
        cls,
        vocoder_config_file: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        model: Optional[ESPnetTTSModel] = None,
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
