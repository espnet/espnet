import argparse
import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
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
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
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
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.slu.espnet_model import ESPnetSLUModel
from espnet2.slu.postdecoder.abs_postdecoder import AbsPostDecoder
from espnet2.slu.postdecoder.hugging_face_transformers_postdecoder import (
    HuggingFaceTransformersPostDecoder,
)
from espnet2.slu.postencoder.conformer_postencoder import ConformerPostEncoder
from espnet2.slu.postencoder.transformer_postencoder import TransformerPostEncoder
from espnet2.tasks.asr import ASRTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.preprocessor import SLUPreprocessor
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
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
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
        espnet=ESPnetSLUModel,
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
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        longformer=LongformerEncoder,
        branchformer=BranchformerEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
        conformer=ConformerPostEncoder,
        transformer=TransformerPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
deliberationencoder_choices = ClassChoices(
    name="deliberationencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
        conformer=ConformerPostEncoder,
        transformer=TransformerPostEncoder,
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
    ),
    type_check=AbsDecoder,
    default="rnn",
)
postdecoder_choices = ClassChoices(
    name="postdecoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostDecoder,
    ),
    type_check=AbsPostDecoder,
    default=None,
    optional=True,
)


class SLUTask(ASRTask):
    """
        SLUTask is a task class for Speech Language Understanding (SLU) that extends
    the ASRTask class. It handles the configuration and construction of SLU
    models, preprocesses the input data, and manages the training and evaluation
    process.

    Attributes:
        num_optimizers (int): Number of optimizers used in training.
        class_choices_list (list): List of class choices for various components
            such as frontend, encoder, decoder, etc.
        trainer (Trainer): Trainer class for managing the training process.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser):
            Adds task-related arguments to the argument parser.

        build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
            Constructs a preprocessing function based on the provided arguments.

        required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of the required data for the task.

        optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns the names of the optional data for the task.

        build_model(args: argparse.Namespace) -> ESPnetSLUModel:
            Builds and initializes the SLU model based on the given arguments.

    Examples:
        To add task arguments, you can do the following:
        ```python
        parser = argparse.ArgumentParser()
        SLUTask.add_task_arguments(parser)
        ```

        To build a preprocessing function:
        ```python
        args = parser.parse_args()
        preprocess_fn = SLUTask.build_preprocess_fn(args, train=True)
        ```

        To create an SLU model:
        ```python
        model = SLUTask.build_model(args)
        ```

    Note:
        The SLUTask class can be customized by modifying the class_choices_list
        or overriding methods such as `build_model` or `add_task_arguments`.

    Todo:
        - Implement more advanced error handling for argument parsing and model
          building.
        - Add support for additional frontend or backend components as needed.
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
        # --deliberationencoder and --deliberationencoder_conf
        deliberationencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --postdecoder and --postdecoder_conf
        postdecoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
                Add task-specific arguments to the given argument parser.

        This method is responsible for adding command-line arguments related to the
        task configuration, such as token lists, model initialization methods, and
        various preprocessor options. The arguments added will allow the user to
        customize the behavior of the SLU task during execution.

        Args:
            parser (argparse.ArgumentParser): The argument parser instance to which
                the task-related arguments will be added.

        Examples:
            To add task arguments for an SLU task, you can use the following code:

            ```python
            import argparse
            from your_package import SLUTask

            parser = argparse.ArgumentParser()
            SLUTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            This method uses the `NestedDictAction` to allow for nested configuration
            dictionaries for certain arguments.

        Todo:
            - Consider refactoring to make argument addition more modular or
              configurable.
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
            "--transcript_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token for transcripts",
        )
        group.add_argument(
            "--two_pass",
            type=str2bool,
            default=False,
            help="Run 2-pass SLU",
        )
        group.add_argument(
            "--pre_postencoder_norm",
            type=str2bool,
            default=False,
            help="pre_postencoder_norm",
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
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
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
            choices=[None, "tacotron", "jaconv", "vietnamese"],
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
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
        Build a preprocessing function based on the provided arguments.

        This method constructs a preprocessing function that is used to
        prepare the input data for training or inference. If preprocessing
        is enabled via the `args.use_preprocessor` flag, it initializes an
        instance of `SLUPreprocessor` with the appropriate configurations.

        Args:
            args (argparse.Namespace): The command-line arguments parsed by
                argparse, containing configurations for preprocessing.
            train (bool): A flag indicating whether the function is being
                built for training (`True`) or inference (`False`).

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A callable function that takes a string and a dictionary
                of numpy arrays as input, returning a dictionary of
                processed numpy arrays. If preprocessing is not enabled,
                returns `None`.

        Examples:
            To build a preprocessing function for training:

            ```python
            args = parser.parse_args()
            preprocess_fn = SLUTask.build_preprocess_fn(args, train=True)
            ```

            To build a preprocessing function for inference:

            ```python
            args = parser.parse_args()
            preprocess_fn = SLUTask.build_preprocess_fn(args, train=False)
            ```

        Note:
            The preprocessing function is utilized to handle tasks such as
            tokenization, noise reduction, and other data preparation steps
            as specified by the command-line arguments.
        """
        if args.use_preprocessor:
            retval = SLUPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                transcript_token_list=(
                    None
                    if "transcript_token_list" not in args
                    else args.transcript_token_list
                ),
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=getattr(args, "rir_scp", None),
                rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
                noise_scp=getattr(args, "noise_scp", None),
                noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
                noise_db_range=getattr(args, "noise_db_range", "13_15"),
                short_noise_thres=getattr(args, "short_noise_thres", 0.5),
                speech_volume_normalize=getattr(args, "rir_scp", None),
                **getattr(args, "preprocessor_conf", {}),
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Defines the SLUTask class for building and training a Spoken Language
        Understanding (SLU) model.

        This class is a subclass of ASRTask and manages various components
        involved in SLU, including data preprocessing, model building, and
        argument parsing.

        Attributes:
            num_optimizers (int): Number of optimizers to use. Default is 1.
            class_choices_list (list): List of class choices for various
                components (frontend, encoder, decoder, etc.).
            trainer (Trainer): The Trainer class used for training the model.

        Args:
            train (bool): Indicates if the data is for training. Default is True.
            inference (bool): Indicates if the data is for inference. Default is
                False.

        Returns:
            Tuple[str, ...]: A tuple of required data names based on the
            training or inference mode.

        Examples:
            To get required data names for training:

            >>> SLUTask.required_data_names(train=True)
            ('speech', 'text')

            To get required data names for inference:

            >>> SLUTask.required_data_names(inference=True)
            ('speech',)

        Note:
            The method distinguishes between training and inference modes to
            specify the required data. During training, both 'speech' and
            'text' data are needed, whereas during inference, only 'speech'
            data is required.

        Raises:
            ValueError: If an invalid argument is passed to any method.

        Todo:
            - Implement additional features for handling different model
            architectures.
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
            SLUTask is a class that encapsulates the task-related configurations
        and methods for Speech Language Understanding (SLU) using an ASR model.

        Attributes:
            num_optimizers (int): The number of optimizers to be used during training.
            class_choices_list (list): A list of class choices for various components
                like frontend, encoder, decoder, etc.
            trainer (Trainer): The class to be used for training procedures.

        Methods:
            add_task_arguments(parser: argparse.ArgumentParser): Adds task-related
                arguments to the provided argument parser.
            build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
                Builds a preprocessing function based on the provided arguments.
            required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
                Returns the required data names based on the training and inference modes.
            optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
                Returns the optional data names.
            build_model(args: argparse.Namespace) -> ESPnetSLUModel:
                Builds and returns an instance of the ESPnetSLUModel based on the
                provided arguments.

        Examples:
            To add task arguments:

            >>> parser = argparse.ArgumentParser()
            >>> SLUTask.add_task_arguments(parser)

            To build a model:

            >>> args = parser.parse_args()
            >>> model = SLUTask.build_model(args)

        Note:
            The class supports a variety of configurations for frontends,
            encoders, decoders, and other components necessary for SLU tasks.
        """
        retval = ("transcript",)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSLUModel:
        """
                Builds the SLU model based on the provided arguments.

        This method constructs the SLU model by configuring various components
        such as the frontend, encoder, decoder, and additional layers based on
        the parameters defined in the `args` namespace. It reads token lists
        from files if provided and initializes the model accordingly.

        Attributes:
            args (argparse.Namespace): Command line arguments containing model
            configurations and options.

        Args:
            args (argparse.Namespace): The command line arguments containing
            configuration options for building the model.

        Returns:
            ESPnetSLUModel: An instance of the SLU model constructed with the
            specified configurations.

        Raises:
            RuntimeError: If `token_list` or `transcript_token_list` is not
            a string or a list.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     token_list="path/to/token_list.txt",
            ...     transcript_token_list="path/to/transcript_token_list.txt",
            ...     input_size=None,
            ...     frontend="default",
            ...     encoder="conformer",
            ...     decoder="transformer",
            ...     model_conf={},
            ... )
            >>> model = SLUTask.build_model(args)
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
        if "transcript_token_list" in args:
            if args.transcript_token_list is not None:
                if isinstance(args.transcript_token_list, str):
                    with open(args.transcript_token_list, encoding="utf-8") as f:
                        transcript_token_list = [line.rstrip() for line in f]

                    # Overwriting token_list to keep it as "portable".
                    args.transcript_token_list = list(transcript_token_list)
                elif isinstance(args.token_list, (tuple, list)):
                    transcript_token_list = list(args.transcript_token_list)
                else:
                    raise RuntimeError(" Transcript token_list must be str or list")
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

        if getattr(args, "deliberationencoder", None) is not None:
            deliberationencoder_class = deliberationencoder_choices.get_class(
                args.deliberationencoder
            )
            deliberationencoder = deliberationencoder_class(
                input_size=encoder_output_size, **args.deliberationencoder_conf
            )
            encoder_output_size = deliberationencoder.output_size()
        else:
            deliberationencoder = None

        if getattr(args, "postdecoder", None) is not None:
            postdecoder_class = postdecoder_choices.get_class(args.postdecoder)
            postdecoder = postdecoder_class(**args.postdecoder_conf)
            encoder_output_size = encoder_output_size
        else:
            postdecoder = None

        # 5. Decoder
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

        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 7. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        if "transcript_token_list" in args:
            if args.transcript_token_list is not None:
                args.model_conf["transcript_token_list"] = transcript_token_list
                args.model_conf["two_pass"] = args.two_pass
                args.model_conf["pre_postencoder_norm"] = args.pre_postencoder_norm
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            deliberationencoder=deliberationencoder,
            decoder=decoder,
            postdecoder=postdecoder,
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
