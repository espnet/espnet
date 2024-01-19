import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import check_argument_types, check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
)
from espnet2.asr.encoder.linear_encoder import LinearEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.s2st.aux_attention.abs_aux_attention import AbsS2STAuxAttention
from espnet2.s2st.aux_attention.multihead import MultiHeadAttention
from espnet2.s2st.espnet_model import ESPnetS2STModel
from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet2.s2st.losses.attention_loss import S2STAttentionLoss
from espnet2.s2st.losses.ctc_loss import S2STCTCLoss
from espnet2.s2st.losses.guided_attention_loss import S2STGuidedAttentionLoss
from espnet2.s2st.losses.tacotron_loss import S2STTacotron2Loss
from espnet2.s2st.synthesizer.abs_synthesizer import AbsSynthesizer
from espnet2.s2st.synthesizer.discrete_synthesizer import TransformerDiscreteSynthesizer
from espnet2.s2st.synthesizer.translatotron import Translatotron
from espnet2.s2st.tgt_feats_extract.abs_tgt_feats_extract import AbsTgtFeatsExtract
from espnet2.s2st.tgt_feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.s2st.tgt_feats_extract.log_mel_fbank import LogMelFbank
from espnet2.s2st.tgt_feats_extract.log_spectrogram import LogSpectrogram
from espnet2.tasks.st import STTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import MutliTokenizerCommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default=None,
)
tgt_feats_extract_choices = ClassChoices(
    name="tgt_feats_extract",
    classes=dict(
        fbank=LogMelFbank,
        spectrogram=LogSpectrogram,
        linear_spectrogram=LinearSpectrogram,
    ),
    type_check=AbsTgtFeatsExtract,
    default=None,
    optional=True,
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
src_normalize_choices = ClassChoices(
    "src_normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
tgt_normalize_choices = ClassChoices(
    "tgt_normalize",
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
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        linear=LinearEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
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
    default=None,
)
st_decoder_choices = ClassChoices(
    "st_decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
    ),
    type_check=AbsDecoder,
    default=None,
)
aux_attention_choices = ClassChoices(
    "aux_attention",
    classes=dict(
        multihead=MultiHeadAttention,
    ),
    type_check=AbsS2STAuxAttention,
)
unit_encoder_choices = ClassChoices(
    "unit_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        linear=LinearEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
)
synthesizer_choices = ClassChoices(
    "synthesizer",
    classes=dict(
        translatotron=Translatotron,
        discrete_unit=TransformerDiscreteSynthesizer,
    ),
    type_check=AbsSynthesizer,
    default="discrete_unit",
)
loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        tacotron=S2STTacotron2Loss,
        guided_attention=S2STGuidedAttentionLoss,
        attention=S2STAttentionLoss,
        ctc=S2STCTCLoss,
    ),
    type_check=AbsS2STLoss,
    default="tacotron",
)


class S2STTask(STTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --tgt_feats_extract and --tgt_feats_extract_conf
        tgt_feats_extract_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --{src, tgt}_normalize and --{src, tgt}_normalize_conf
        src_normalize_choices,
        tgt_normalize_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --asr_decoder and --asr_decoder_conf
        asr_decoder_choices,
        # --st_decoder and --st_decoder_conf
        st_decoder_choices,
        # --aux_attention and --aux_attention_conf
        aux_attention_choices,
        # --unit_encoder and --unit_encoder_conf
        unit_encoder_choices,
        # --synthesizer and --synthesizer_conf
        synthesizer_choices,
        loss_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")

        group.add_argument(
            "--s2st_type",
            type=str,
            default="discrete_unit",
            help="Types of S2ST",
            choices=[
                "translatotron",
                "translatotron2",
                "discrete_unit",
                "unity",
            ],
        )
        group.add_argument(
            "--tgt_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for target language)",
        )
        group.add_argument(
            "--src_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for source language)",
        )
        group.add_argument(
            "--unit_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for discrete_unit)",
        )
        group.add_argument(
            "--odim",
            type=int_or_none,
            default=None,
            help="The number of dimension of output feature",
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
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        group.add_argument(
            "--output_size",
            type=int_or_none,
            default=None,
            help="The number of output dimension of the feature",
        )
        group.add_argument(
            "--asr_ctc",
            type=str2bool,
            default=False,
            help="whether to conduct CTC on ASR objectives",
        )
        group.add_argument(
            "--st_ctc",
            type=str2bool,
            default=False,
            help="whether to conduct CTC on ST objectives",
        )
        group.add_argument(
            "--asr_ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for ASR CTC class.",
        )
        group.add_argument(
            "--st_ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for ST CTC class.",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetS2STModel),
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
            "--tgt_token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The target text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--src_token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn", "none"],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--tgt_bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for target language)",
        )
        group.add_argument(
            "--src_bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for source language)",
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
            "--tgt_g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--src_g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--losses",
            action=NestedDictAction,
            default=[
                {
                    "name": "synthesis",
                    "conf": {},
                    "type": "attention",
                },
            ],
            help="The criterions binded with the loss wrappers.",
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
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
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
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.src_token_type == "none":
            args.src_token_type = None
        if args.unit_token_list is None:
            unit_token_type = None
        else:
            unit_token_type = "word"

        # NOTE(jiatong): if discrete unit is used, the tokenizer will parse discrete \
        #                unit sequence, otherwise the tgt_speech will left unchanged \
        #                for spectrogram.

        if args.use_preprocessor:
            retval = MutliTokenizerCommonPreprocessor(
                train=train,
                token_type=[args.tgt_token_type, args.src_token_type, unit_token_type],
                token_list=[
                    args.tgt_token_list,
                    args.src_token_list,
                    args.unit_token_list,
                ],
                bpemodel=[args.tgt_bpemodel, args.src_bpemodel, None],
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=[args.tgt_g2p, args.src_g2p, None],
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                short_noise_thres=args.short_noise_thres
                if hasattr(args, "short_noise_thres")
                else 0.5,
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "speech_volume_normalize")
                else None,
                speech_name="src_speech",
                text_name=["tgt_text", "src_text", "tgt_speech"],
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
            retval = ("src_speech", "tgt_speech")
        else:
            # Recognition mode
            retval = ("src_speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("src_text", "tgt_text")
        else:
            retval = ("tgt_speech",)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetS2STModel:
        assert check_argument_types()
        if args.tgt_token_list is not None:
            if isinstance(args.tgt_token_list, str):
                with open(args.tgt_token_list, encoding="utf-8") as f:
                    tgt_token_list = [line.rstrip() for line in f]

                # Overwriting token_list to keep it as "portable".
                args.tgt_token_list = list(tgt_token_list)
            elif isinstance(args.tgt_token_list, (tuple, list)):
                tgt_token_list = list(args.tgt_token_list)
            else:
                raise RuntimeError("token_list must be str or list")
            tgt_vocab_size = len(tgt_token_list)
            logging.info(f"Target Vocabulary size: {tgt_vocab_size }")
        else:
            tgt_token_list, tgt_vocab_size = None, None

        if args.src_token_list is not None:
            if isinstance(args.src_token_list, str):
                with open(args.src_token_list, encoding="utf-8") as f:
                    src_token_list = [line.rstrip() for line in f]

                # Overwriting src_token_list to keep it as "portable".
                args.src_token_list = list(src_token_list)
            elif isinstance(args.src_token_list, (tuple, list)):
                src_token_list = list(args.src_token_list)
            else:
                raise RuntimeError("token_list must be str or list")
            src_vocab_size = len(src_token_list)
            logging.info(f"Source vocabulary size: {src_vocab_size }")
        else:
            src_token_list, src_vocab_size = None, None

        if args.unit_token_list is not None:
            if isinstance(args.unit_token_list, str):
                with open(args.unit_token_list, encoding="utf-8") as f:
                    unit_token_list = [line.rstrip() for line in f]

                # Overwriting unit_token_list to keep it as "portable".
                args.unit_token_list = list(unit_token_list)
            elif isinstance(args.unit_token_list, (tuple, list)):
                unit_token_list = list(args.unit_token_list)
            else:
                raise RuntimeError("token_list must be str or list")
            unit_vocab_size = len(unit_token_list)
            logging.info(f"Discrete unit vocabulary size: {unit_vocab_size }")
        else:
            unit_token_list, unit_vocab_size = None, None

        # 1. frontend and tgt_feats_extract
        if args.input_size is None:
            # Extract source features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            raw_input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            raw_input_size = args.input_size

        if args.tgt_feats_extract is not None:
            # Extract target features in the model
            tgt_feats_extract_class = tgt_feats_extract_choices.get_class(
                args.tgt_feats_extract
            )
            tgt_feats_extract = tgt_feats_extract_class(**args.tgt_feats_extract_conf)
            output_size = tgt_feats_extract.output_size()
        else:
            # NOTE(jiatong): discrete unit cases
            assert (
                unit_vocab_size is not None
            ), "need a discrete unit token list for non-spectrograms target speech"
            output_size = unit_vocab_size
            tgt_feats_extract = None

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.src_normalize is not None:
            src_normalize_class = src_normalize_choices.get_class(args.src_normalize)
            src_normalize = src_normalize_class(**args.src_normalize_conf)
        else:
            src_normalize = None
        if args.tgt_normalize is not None:
            tgt_normalize_class = tgt_normalize_choices.get_class(args.tgt_normalize)
            tgt_normalize = tgt_normalize_class(**args.tgt_normalize_conf)
        else:
            tgt_normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            input_size = raw_input_size
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 5. Decoders
        if args.asr_decoder is not None:
            asr_decoder_class = asr_decoder_choices.get_class(args.asr_decoder)

            asr_decoder = asr_decoder_class(
                vocab_size=src_vocab_size,
                encoder_output_size=encoder_output_size,
                **args.asr_decoder_conf,
            )
        else:
            asr_decoder = None

        if args.st_decoder is not None:
            st_decoder_class = st_decoder_choices.get_class(args.st_decoder)

            st_decoder = st_decoder_class(
                vocab_size=tgt_vocab_size,
                encoder_output_size=encoder_output_size,
                **args.st_decoder_conf,
            )
        else:
            st_decoder = None

        # 6. CTCs
        if src_token_list is not None and args.asr_ctc:
            asr_ctc = CTC(
                odim=src_vocab_size,
                encoder_output_size=encoder_output_size,
                **args.asr_ctc_conf,
            )
        else:
            asr_ctc = None

        if tgt_token_list is not None and args.st_ctc:
            st_ctc = CTC(
                odim=tgt_vocab_size,
                encoder_output_size=encoder_output_size,
                **args.st_ctc_conf,
            )
        else:
            st_ctc = None

        # 7. Auxiliary Attention
        if args.aux_attention:
            aux_attention_class = aux_attention_choices.get_class(args.aux_attention)
            aux_attention = aux_attention_class(
                n_feat=encoder_output_size, **args.aux_attention_conf
            )
        else:
            aux_attention = None

        # 8. Unit encoder
        if args.unit_encoder:
            unit_encoder_class = unit_encoder_choices.get_class(args.unit_encoder)
            unit_encoder = unit_encoder_class(
                input_size=encoder_output_size,
                **args.unit_encoder_conf,
            )
        else:
            unit_encoder = None

        # 9. Synthesizer
        synthesizer_class = synthesizer_choices.get_class(args.synthesizer)
        synthesizer_idim = (
            encoder_output_size
            if args.aux_attention is None
            else 2 * encoder_output_size
        )
        logging.info(
            "synthesizer_idim: {}, encoder_output_size: {}, aux_attention {}".format(
                synthesizer_idim, encoder_output_size, args.aux_attention
            )
        )
        synthesizer = synthesizer_class(
            idim=synthesizer_idim, odim=output_size, **args.synthesizer_conf
        )

        # 10. Loss definition
        losses = {}
        if getattr(args, "losses", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.losses:
                logging.info("initialize loss: {}".format(ctr["name"]))
                if ctr["name"] == "src_attn":
                    loss = loss_choices.get_class(ctr["type"])(
                        vocab_size=src_vocab_size, **ctr["conf"]
                    )
                elif ctr["name"] == "tgt_attn":
                    loss = loss_choices.get_class(ctr["type"])(
                        vocab_size=tgt_vocab_size, **ctr["conf"]
                    )
                elif unit_vocab_size is not None and ctr["name"] == "synthesis":
                    loss = loss_choices.get_class(ctr["type"])(
                        vocab_size=unit_vocab_size, **ctr["conf"]
                    )
                else:
                    loss = loss_choices.get_class(ctr["type"])(**ctr["conf"])
                losses[ctr["name"]] = loss

        # 11. Build model
        model = ESPnetS2STModel(
            s2st_type=args.s2st_type,
            frontend=frontend,
            tgt_feats_extract=tgt_feats_extract,
            specaug=specaug,
            src_normalize=src_normalize,
            tgt_normalize=tgt_normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            asr_decoder=asr_decoder,
            st_decoder=st_decoder,
            aux_attention=aux_attention,
            unit_encoder=unit_encoder,
            synthesizer=synthesizer,
            asr_ctc=asr_ctc,
            st_ctc=st_ctc,
            losses=losses,
            tgt_vocab_size=tgt_vocab_size,
            tgt_token_list=tgt_token_list,
            src_vocab_size=src_vocab_size,
            src_token_list=src_token_list,
            unit_vocab_size=unit_vocab_size,
            unit_token_list=unit_token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 10. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

    @classmethod
    def build_vocoder_from_file(
        cls,
        vocoder_config_file: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        model: Optional[ESPnetS2STModel] = None,
        device: str = "cpu",
    ):
        # NOTE(jiatong): this is essentially the same as TTSTask
        # Build vocoder
        if vocoder_file is None:
            # If vocoder file is not provided, use griffin-lim as a vocoder
            vocoder_conf = {}
            if vocoder_config_file is not None:
                vocoder_config_file = Path(vocoder_config_file)
                with vocoder_config_file.open("r", encoding="utf-8") as f:
                    vocoder_conf = yaml.safe_load(f)
            if model.frontend is not None:
                vocoder_conf.update(model.tgt_feats_extract.get_parameters())
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
