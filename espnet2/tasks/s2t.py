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
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
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
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.transformer_encoder_multispkr import (
    TransformerEncoder as TransformerEncoderMultiSpkr,
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.whisper import WhisperFrontend
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
from espnet2.s2t.espnet_model import ESPnetS2TModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import AbsPreprocessor, S2TPreprocessor
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
    ),
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
    name="normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
model_choices = ClassChoices(
    name="model",
    classes=dict(
        espnet=ESPnetS2TModel,
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
    ),
    type_check=AbsEncoder,
    default="rnn",
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
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
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
        s2t=S2TPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="s2t",
)


class S2TTask(AbsTask):
    """
        S2TTask is a class that defines the sequence-to-text (S2T) task for training
    and evaluating models in the ESPnet framework. It inherits from the AbsTask
    class and provides methods to manage task-specific configurations, data
    processing, and model building.

    Attributes:
        num_optimizers (int): The number of optimizers used in training.
        class_choices_list (list): A list of class choices for different
            components such as frontend, encoder, decoder, etc.
        trainer (Trainer): The trainer class used for training and evaluation.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser):
            Adds task-related arguments to the provided argument parser.

        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Builds a collate function for batching data.

        build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
            Builds a preprocessing function for input data.

        required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of the required data for the task.

        optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of the optional data for the task.

        build_model(args: argparse.Namespace) -> ESPnetS2TModel:
            Constructs and returns the ESPnet S2T model based on the given arguments.

    Examples:
        # Example of adding task arguments
        import argparse
        parser = argparse.ArgumentParser()
        S2TTask.add_task_arguments(parser)

        # Example of building a model
        args = parser.parse_args()
        model = S2TTask.build_model(args)

    Note:
        This class is intended to be used as part of the ESPnet framework for
        sequence-to-text tasks.

    Todo:
        - Add support for more complex task configurations.
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
                Adds task-related arguments to the provided argument parser.

        This method is responsible for defining and adding various command-line
        arguments that are specific to the S2T task. These arguments can include
        options for preprocessing, model configuration, and other task-specific
        parameters.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the
                task-related arguments will be added.

        Examples:
            To add task arguments to a parser, you can use:

            ```python
            import argparse
            from s2t_task import S2TTask

            parser = argparse.ArgumentParser()
            S2TTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            This method modifies the parser to include arguments necessary for
            configuring the S2T task, such as `--token_list`, `--input_size`,
            and others. The default values for some arguments are defined
            based on the class-level attributes.

        Todo:
            - Consider expanding the options for the `--token_type` argument to
              include more tokenization methods in the future.
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
        Build a collate function for batching input data.

        This function creates a collate function that will be used to
        prepare batches of input data for training or evaluation. It
        ensures that the input sequences are padded correctly to
        maintain consistent input shapes.

        Args:
            args (argparse.Namespace): The parsed command-line arguments.
            train (bool): A flag indicating whether the function is
                being used for training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]:
                A collate function that processes batches of input data.

        Note:
            The integer value `0` is reserved by the CTC blank symbol.

        Examples:
            >>> collate_fn = S2TTask.build_collate_fn(args, train=True)
            >>> batch = collate_fn(data)
            >>> print(batch)  # Output will depend on the input data structure
        """
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
                Builds a preprocessing function based on the provided arguments.

        This method checks if preprocessing should be applied based on the `use_preprocessor`
        argument. If true, it initializes a preprocessor class using the specified configuration
        and returns a callable function that processes the input data.

        Args:
            cls: The class that this method belongs to.
            args (argparse.Namespace): The arguments parsed from the command line.
            train (bool): A flag indicating whether the function is being built for training
                or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]: A function
            that takes an input string and a dictionary of features, returning a processed
            dictionary of features, or None if preprocessing is not enabled.

        Raises:
            AttributeError: If the specified preprocessor class is not found.
            Exception: If any other exception occurs during initialization of the preprocessor.

        Examples:
            To use this function in a training scenario, you might do the following:

            ```python
            preprocess_fn = S2TTask.build_preprocess_fn(args, train=True)
            processed_data = preprocess_fn(input_string, features_dict)
            ```

        Note:
            The method expects that the `args` object has attributes corresponding to the
            preprocessing options, including `token_type`, `token_list`, and various
            noise and RIR parameters.

        Todo:
            - Ensure backward compatibility for preprocessor attributes in future updates.
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
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                non_linguistic_symbols=args.non_linguistic_symbols,
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
                **args.preprocessor_conf,
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns the required data names for training or inference.

        This method determines the data names needed for the task based on whether
        it is in training or inference mode. In training mode, both 'speech' and
        'text' data are required. In inference mode, only 'speech' data is required.

        Args:
            train (bool): A flag indicating whether the method is called during
                training. Defaults to True.
            inference (bool): A flag indicating whether the method is called
                during inference. Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of the required data.
                For training, returns ('speech', 'text'). For inference, returns
                ('speech',).

        Examples:
            >>> S2TTask.required_data_names(train=True, inference=False)
            ('speech', 'text')

            >>> S2TTask.required_data_names(train=False, inference=True)
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
            Class representing the Speech-to-Text (S2T) task.

        This class provides methods to configure the S2T task, including the
        addition of task-specific arguments, building models, and handling
        data processing.

        Attributes:
            num_optimizers (int): The number of optimizers to be used. Default is 1.
            class_choices_list (List[ClassChoices]): List of class choices for
                various components such as frontend, encoder, decoder, etc.
            trainer (Trainer): The trainer class to be used for training.

        Args:
            parser (argparse.ArgumentParser): The argument parser to add task-related
                arguments.

        Returns:
            None

        Yields:
            None

        Raises:
            RuntimeError: If the token list is not a string or a list.

        Examples:
            To add task arguments:

            ```python
            parser = argparse.ArgumentParser()
            S2TTask.add_task_arguments(parser)
            ```

            To build a model:

            ```python
            args = parser.parse_args()
            model = S2TTask.build_model(args)
            ```

            To retrieve required data names:

            ```python
            data_names = S2TTask.required_data_names(train=True)
            ```

            To retrieve optional data names:

            ```python
            optional_data_names = S2TTask.optional_data_names(train=True)
            ```

        Note:
            The optional data names are logged for reference.

        Todo:
            Implement more flexible model building strategies.
        """
        MAX_REFERENCE_NUM = 4

        retval = ["text_prev", "text_ctc"] + [
            "text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)
        ]
        retval = tuple(retval)

        logging.info(f"Optional Data Names: {retval}")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetS2TModel:
        """
                Builds the S2T model based on the provided arguments.

        This method constructs an instance of the ESPnetS2TModel by assembling
        various components such as frontend, encoder, decoder, and CTC based
        on the configuration specified in the `args` argument. It also handles
        token list loading and initialization of model parameters.

        Args:
            args (argparse.Namespace): The arguments containing configuration
                parameters for model construction. This includes options for
                the frontend, encoder, decoder, and other components.

        Returns:
            ESPnetS2TModel: An instance of the constructed S2T model.

        Raises:
            RuntimeError: If `token_list` is neither a string nor a list.

        Examples:
            # Example of building a model with specific arguments
            args = argparse.Namespace(
                token_list='path/to/token_list.txt',
                input_size=None,
                frontend='default',
                encoder='transformer',
                decoder='lightweight_conv',
                ctc_conf={'some_param': 'value'},
                model_conf={}
            )
            model = S2TTask.build_model(args)

        Note:
            The function expects the `token_list` argument to be either a
            path to a text file or a list of tokens. The method will raise
            a RuntimeError if the `token_list` is not in the expected format.
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
            decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **args.decoder_conf,
            )
        else:
            decoder = None

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
            token_list=token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
