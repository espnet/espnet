import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.hugging_face_transformers_decoder import (  # noqa: H301
    HuggingFaceTransformersDecoder,
)
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.avhubert_encoder import FairseqAVHubertEncoder
from espnet2.asr.encoder.beats_encoder import BeatsEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
    TorchAudioHuBERTPretrainEncoder,
)
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder
from espnet2.asr.encoder.multiconvformer_encoder import MultiConvConformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.transformer_encoder_multispkr import (
    TransformerEncoder as TransformerEncoderMultiSpkr,
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.whisper import WhisperFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.maskctc_model import MaskCTCModel
from espnet2.asr.pit_espnet_model import ESPnetASRModel as PITESPnetModel
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.postencoder.length_adaptor_postencoder import LengthAdaptorPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    CommonPreprocessor_multi,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        whisper=WhisperFrontend,
    ),  # If setting this to none, please make sure to provide input_size in the config.
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
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
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetASRModel,
        maskctc=MaskCTCModel,
        pit_espnet=PITESPnetModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
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
        transformer_multispkr=TransformerEncoderMultiSpkr,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        torchaudiohubert=TorchAudioHuBERTPretrainEncoder,
        longformer=LongformerEncoder,
        branchformer=BranchformerEncoder,
        whisper=OpenAIWhisperEncoder,
        e_branchformer=EBranchformerEncoder,
        avhubert=FairseqAVHubertEncoder,
        multiconv_conformer=MultiConvConformerEncoder,
        beats=BeatsEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
        length_adaptor=LengthAdaptorPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        transducer=TransducerDecoder,
        mlm=MLMDecoder,
        whisper=OpenAIWhisperDecoder,
        hugging_face_transformers=HuggingFaceTransformersDecoder,
        s4=S4Decoder,
    ),
    type_check=AbsDecoder,
    default=None,
    optional=True,
)
preprocessor_choices = ClassChoices(
    "preprocessor",
    classes=dict(
        default=CommonPreprocessor,
        multi=CommonPreprocessor_multi,
    ),
    type_check=AbsPreprocessor,
    default="default",
)


class ASRTask(AbsTask):
    """
    Automatic Speech Recognition (ASR) Task.

    This class defines the ASR task and contains methods for handling the task
    configuration, model building, data preprocessing, and argument parsing.

    Attributes:
        num_optimizers (int): Number of optimizers to use for training.
        class_choices_list (list): List of class choices for various components
            of the ASR system, including frontend, encoder, decoder, etc.
        trainer (Trainer): Trainer class for managing the training process.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser): Adds arguments
            specific to the ASR task to the provided argument parser.
        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Builds a collate function for batching data during training or
            evaluation.
        build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
            Builds a preprocessing function based on the specified arguments.
        required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of required data for the ASR task.
        optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of optional data for the ASR task.
        build_model(args: argparse.Namespace) -> ESPnetASRModel:
            Constructs and returns an instance of the ASR model based on the
            provided configuration arguments.

    Examples:
        To create an ASR task and add arguments:

        ```python
        parser = argparse.ArgumentParser()
        ASRTask.add_task_arguments(parser)
        args = parser.parse_args()
        ```

        To build the model:

        ```python
        model = ASRTask.build_model(args)
        ```

    Note:
        This class is intended to be used as part of the ESPnet framework for
        speech processing tasks. Ensure that the necessary components are
        installed and configured before using this class.
    """

    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --model and --model_conf
        model_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
            Adds task-related command-line arguments to the provided argument parser.

        This method defines various arguments required for configuring the ASR task.
        It adds options for token mapping, initialization methods, input size, CTC
        configuration, joint network configuration, preprocessing options, and more.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the
                task-related arguments will be added.

        Examples:
            To add task arguments to a parser, use the following code:

            ```python
            import argparse
            from your_module import ASRTask

            parser = argparse.ArgumentParser()
            ASRTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            This method modifies the parser in place and is intended to be called
            during the argument parsing setup of the ASR task.

        Raises:
            Any exceptions raised by the underlying methods for argument addition.

        Todo:
            - Consider adding more options for future task configurations.
        """
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
                "normal",
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
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--joint_net_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--use_lang_prompt",
            type=str2bool,
            default=False,
            help="Use language id as prompt",
        )
        group.add_argument(
            "--use_nlp_prompt",
            type=str2bool,
            default=False,
            help="Use natural language phrases as prompt",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=[
                "bpe",
                "char",
                "word",
                "phn",
                "hugging_face",
                "whisper_en",
                "whisper_multilingual",
            ],
            help="The text will be tokenized " "in the specified level token",
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
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[
                None,
                "tacotron",
                "jaconv",
                "vietnamese",
                "whisper_en",
                "whisper_basic",
            ],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
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
        group.add_argument(
            "--aux_ctc_tasks",
            type=str,
            nargs="+",
            default=[],
            help="Auxillary tasks to train on using CTC loss. ",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        """
        Build a collate function for the ASR task.

    This method constructs a collate function that can be used to process a
    batch of data during training or evaluation. The collate function handles
    padding of sequences to ensure uniform input sizes for the model.

    Args:
        args (argparse.Namespace): The command line arguments parsed from the
            configuration.
        train (bool): A flag indicating whether the collate function is being
            built for training or evaluation.

    Returns:
        Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                 Tuple[List[str], Dict[str, torch.Tensor]]]:
            A callable that takes a collection of tuples (where each tuple
            contains a string identifier and a dictionary of numpy arrays)
            and returns a tuple containing a list of string identifiers and
            a dictionary of tensors.

    Examples:
        >>> collate_fn = ASRTask.build_collate_fn(args, train=True)
        >>> batch_data = [
        ...     ("audio1", {"feature": np.array([1.0, 2.0])}),
        ...     ("audio2", {"feature": np.array([3.0, 4.0])}),
        ... ]
        >>> identifiers, tensor_dict = collate_fn(batch_data)

    Note:
        The int value 0 is reserved by the CTC-blank symbol.
        """[
            Collection[Tuple[str, Dict[str, np.ndarray]]]
        ],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
        Builds a preprocessing function based on the provided arguments.

        This method constructs a preprocessing function that applies
        the specified preprocessor to the input data. The preprocessor
        can be customized through the command-line arguments, allowing
        for different tokenization and data cleaning strategies.

        Args:
            args (argparse.Namespace): The command-line arguments containing
                the configuration for the preprocessor and other settings.
            train (bool): A flag indicating whether the function is being
                built for training or inference.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A callable function that takes a string and a dictionary
                as input and returns a dictionary of preprocessed data.
                Returns None if preprocessing is not enabled.

        Raises:
            Exception: Raises an exception if there is an error when
                retrieving or instantiating the preprocessor class.

        Examples:
            To use this function, you would typically call it in the context
            of an ASR task setup:

            ```python
            preprocess_fn = ASRTask.build_preprocess_fn(args, train=True)
            processed_data = preprocess_fn("sample.wav", {"text": "hello"})
            ```

            If preprocessing is not needed, it will return None:

            ```python
            preprocess_fn = ASRTask.build_preprocess_fn(args, train=False)
            assert preprocess_fn is None
            ```

        Note:
            Ensure that the `--use_preprocessor` argument is set to True
            to enable preprocessing.
        """
        if args.use_preprocessor:
            try:
                _ = getattr(args, "preprocessor")
            except AttributeError:
                setattr(args, "preprocessor", "default")
                setattr(args, "preprocessor_conf", dict())
            except Exception as e:
                raise e

            preprocessor_class = preprocessor_choices.get_class(args.preprocessor)
            retval = preprocessor_class(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
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
                short_noise_thres=(
                    args.short_noise_thres
                    if hasattr(args, "short_noise_thres")
                    else 0.5
                ),
                speech_volume_normalize=(
                    args.speech_volume_normalize if hasattr(args, "rir_scp") else None
                ),
                aux_task_names=(
                    args.aux_ctc_tasks if hasattr(args, "aux_ctc_tasks") else None
                ),
                use_lang_prompt=(
                    args.use_lang_prompt if hasattr(args, "use_lang_prompt") else None
                ),
                **args.preprocessor_conf,
                use_nlp_prompt=(
                    args.use_nlp_prompt if hasattr(args, "use_nlp_prompt") else None
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
            Retrieves the required data names for the ASR task based on the mode of
        operation (training or inference).

        This method returns a tuple containing the names of the required data. If
        the task is in inference mode, it returns only the 'speech' data name.
        Otherwise, it returns both 'speech' and 'text'.

        Args:
            train (bool): Indicates whether the task is in training mode.
                          Default is True.
            inference (bool): Indicates whether the task is in inference mode.
                              Default is False.

        Returns:
            Tuple[str, ...]: A tuple of required data names. If not in inference,
                             returns ("speech", "text"). If in inference, returns
                             ("speech",).

        Examples:
            >>> ASRTask.required_data_names(train=True, inference=False)
            ('speech', 'text')

            >>> ASRTask.required_data_names(train=True, inference=True)
            ('speech',)
        """
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns a tuple of optional data names used in the ASR task.

        This method generates a list of optional data names based on whether the
        task is for training or inference. The optional data names can include
        speaker-specific text references and prompts.

        Attributes:
            MAX_REFERENCE_NUM (int): The maximum number of reference speakers.

        Args:
            train (bool): A flag indicating whether the task is for training.
                Default is True.
            inference (bool): A flag indicating whether the task is for inference.
                Default is False.

        Returns:
            Tuple[str, ...]: A tuple containing the optional data names.

        Examples:
            >>> ASRTask.optional_data_names(train=True, inference=False)
            ('text_spk2', 'text_spk3', 'text_spk4', 'prompt')

            >>> ASRTask.optional_data_names(train=False, inference=True)
            ()
        """
        MAX_REFERENCE_NUM = 4

        retval = ["text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval = retval + ["prompt"]
        retval = tuple(retval)

        logging.info(f"Optional Data Names: {retval }")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
        """
        Build and initialize an ASR model based on the provided arguments.

        This method constructs the ASR model by sequentially creating
        components such as frontend, encoder, postencoder, and decoder
        according to the specified configurations in the `args` parameter.
        The method also handles token list initialization and the
        optional integration of a joint network for transducer models.

        Args:
            args (argparse.Namespace): The command line arguments parsed
                into a namespace, which include configurations for the
                model components.

        Returns:
            ESPnetASRModel: An instance of the ASR model configured as
                per the provided arguments.

        Raises:
            RuntimeError: If the token_list is neither a string nor a list.

        Examples:
            >>> args = argparse.Namespace()
            >>> args.token_list = "path/to/token_list.txt"
            >>> args.input_size = None
            >>> model = ASRTask.build_model(args)
            >>> print(model)

        Note:
            The method assumes that the necessary classes for each
            component (e.g., frontend, encoder, decoder) have been
            defined and registered in the respective class choices.
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

        # If use multi-blank transducer criterion,
        # big blank symbols are added just before the standard blank
        if args.model_conf.get("transducer_multi_blank_durations", None) is not None:
            sym_blank = args.model_conf.get("sym_blank", "<blank>")
            blank_idx = token_list.index(sym_blank)
            for dur in args.model_conf.get("transducer_multi_blank_durations"):
                if f"<blank{dur}>" not in token_list:  # avoid this during inference
                    token_list.insert(blank_idx, f"<blank{dur}>")
            args.token_list = token_list

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
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

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
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

        # 5. Decoder
        if getattr(args, "decoder", None) is not None:
            decoder_class = decoder_choices.get_class(args.decoder)

            if args.decoder == "transducer":
                decoder = decoder_class(
                    vocab_size,
                    embed_pad=0,
                    **args.decoder_conf,
                )

                joint_network = JointNetwork(
                    vocab_size,
                    encoder.output_size(),
                    decoder.dunits,
                    **args.joint_net_conf,
                )
            else:
                decoder = decoder_class(
                    vocab_size=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                joint_network = None
        else:
            decoder = None
            joint_network = None

        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 7. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            token_list=token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
