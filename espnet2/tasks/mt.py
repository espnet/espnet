import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

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
from espnet2.asr.discrete_asr_espnet_model import ESPnetDiscreteASRModel
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.mt.espnet_model import ESPnetMTModel
from espnet2.mt.frontend.embedding import Embedding, PatchEmbedding
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import MutliTokenizerCommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        embed=Embedding,
        patch=PatchEmbedding,
    ),
    type_check=AbsFrontend,
    default="embed",
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
        branchformer=BranchformerEncoder,
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
    ),
    type_check=AbsDecoder,
    default="rnn",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        mt=ESPnetMTModel,
        discrete_asr=ESPnetDiscreteASRModel,
    ),
    type_check=AbsESPnetModel,
    default="mt",
)


class MTTask(AbsTask):
    """
        MTTask is a class that handles the configuration and execution of machine
    translation tasks. It inherits from the AbsTask class and provides methods
    for setting up task-specific arguments, building models, and processing data
    for training and inference.

    Attributes:
        num_optimizers (int): Number of optimizers to be used in the task. Default is 1.
        class_choices_list (list): A list of class choices for various components
            (frontend, specaug, preencoder, encoder, postencoder, decoder, model).
        trainer (Trainer): The Trainer class used for modifying train or eval
            procedures.

    Args:
        parser (argparse.ArgumentParser): Argument parser to add task-related
            arguments.

    Returns:
        Callable: A function to collate input data for training.

    Yields:
        Optional[Callable]: A function to preprocess input data based on the
            provided arguments.

    Raises:
        RuntimeError: If token_list or src_token_list is not a string or list.

    Examples:
        To add task arguments:
            parser = argparse.ArgumentParser()
            MTTask.add_task_arguments(parser)

        To build a model:
            args = parser.parse_args()
            model = MTTask.build_model(args)

        To build a collate function:
            collate_fn = MTTask.build_collate_fn(args, train=True)

    Note:
        Ensure to provide the correct token lists and model configurations for
        successful model building.

    Todo:
        - Add more detailed error handling for specific scenarios.
    """

    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
                Adds task-specific arguments to the provided argument parser.

        This method defines command-line arguments that are related to the task. It
        creates groups for task-related and preprocessing-related arguments, and it
        also incorporates choices for various components of the model architecture,
        such as frontend, encoder, decoder, and more.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which task
                arguments will be added.

        Examples:
            parser = argparse.ArgumentParser()
            MTTask.add_task_arguments(parser)

        Note:
            The method is designed to allow flexible addition of command-line
            arguments while ensuring that the arguments are organized and
            documented properly. Some arguments are required, and their presence
            can be checked during runtime.

        Todo:
            - Consider adding more validation checks for argument values.
            - Expand the documentation to include more detailed usage examples.
        """
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["src_token_list", "token_list"]

        group.add_argument(
            "--token_list",
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
            choices=["bpe", "char", "word", "phn", None],
            help="The target text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--src_token_type",
            type=str_or_none,
            default="bpe",
            choices=["bpe", "char", "word", "phn", None],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
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
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        parser.add_argument(
            "--tokenizer_encode_conf",
            type=dict,
            default=None,
            help="Tokenization encoder conf, "
            "e.g. BPE dropout: enable_sampling=True, alpha=0.1, nbest_size=-1",
        )
        parser.add_argument(
            "--src_tokenizer_encode_conf",
            type=dict,
            default=None,
            help="Src tokenization encoder conf, "
            "e.g. BPE dropout: enable_sampling=True, alpha=0.1, nbest_size=-1",
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
            Build a collate function for the task.

        This method constructs a callable function that can be used to collate
        batches of data. It is specifically designed for use during training and
        evaluation of the MTTask model.

        Args:
            args (argparse.Namespace): The command-line arguments containing the
                configuration for the task.
            train (bool): A flag indicating whether the function is being built for
                training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]:
                A callable that takes a collection of tuples containing string
                identifiers and dictionaries of numpy arrays, and returns a tuple
                containing a list of strings and a dictionary of PyTorch tensors.

        Examples:
            >>> from espnet2.tasks.mt import MTTask
            >>> args = argparse.Namespace(...)
            >>> collate_fn = MTTask.build_collate_fn(args, train=True)
            >>> batch = [
            ...     ("example_1", {"input": np.array([1, 2, 3])}),
            ...     ("example_2", {"input": np.array([4, 5, 6])}),
            ... ]
            >>> collated_data = collate_fn(batch)
            >>> print(collated_data)

        Note:
            The int value = 0 is reserved by the CTC-blank symbol.
        """
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
        Builds a preprocessing function for the task.

        This function constructs a callable that can be used to preprocess
        input data based on the specified arguments. If preprocessing is
        enabled, it returns a `MutliTokenizerCommonPreprocessor` instance
        configured with the provided arguments. Otherwise, it returns `None`.

        Args:
            args (argparse.Namespace): The command-line arguments containing
                configuration options for preprocessing.
            train (bool): A flag indicating whether the function is being
                built for training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.ndarray]], Dict[str,
            np.ndarray]]]: A preprocessing function if `args.use_preprocessor`
            is True, otherwise None.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(use_preprocessor=True, token_type='bpe',
            ...                  src_token_type='bpe', token_list='path/to/token_list.txt',
            ...                  src_token_list='path/to/src_token_list.txt',
            ...                  bpemodel='path/to/bpemodel',
            ...                  src_bpemodel='path/to/src_bpemodel',
            ...                  non_linguistic_symbols='path/to/symbols',
            ...                  cleaner='tacotron', g2p='g2p_method',
            ...                  tokenizer_encode_conf={},
            ...                  src_tokenizer_encode_conf={})
            >>> preprocess_fn = MTTask.build_preprocess_fn(args, train=True)
            >>> preprocessed_data = preprocess_fn("some text", {"key": np.array([1, 2, 3])})
        """
        if args.use_preprocessor:
            retval = MutliTokenizerCommonPreprocessor(
                train=train,
                token_type=[args.token_type, args.src_token_type],
                token_list=[args.token_list, args.src_token_list],
                bpemodel=[args.bpemodel, args.src_bpemodel],
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                text_name=["text", "src_text"],
                tokenizer_encode_conf=(
                    [
                        args.tokenizer_encode_conf,
                        args.src_tokenizer_encode_conf,
                    ]
                    if train
                    else [dict(), dict()]
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
            Returns the required data names for training or inference.

        This method determines the necessary data inputs for the task based on
        whether the operation is in training or inference mode. In training mode,
        both source and target texts are required, while in inference mode, only
        the source text is needed.

        Args:
            train (bool): A flag indicating if the task is in training mode.
                          Defaults to True.
            inference (bool): A flag indicating if the task is in inference mode.
                              Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of the required data.
                             Returns ("src_text", "text") in training mode and
                             ("src_text",) in inference mode.

        Examples:
            >>> MTTask.required_data_names(train=True, inference=False)
            ('src_text', 'text')

            >>> MTTask.required_data_names(train=False, inference=True)
            ('src_text',)
        """
        if not inference:
            retval = ("src_text", "text")
        else:
            # Recognition mode
            retval = ("src_text",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
                MTTask is a class that defines the task for Machine Translation (MT) within the
        ESPnet framework. It manages the configuration and setup of various components
        needed for the MT task, including the frontend, encoder, decoder, and model.

        Attributes:
            num_optimizers (int): The number of optimizers used for training. Default is 1.
            class_choices_list (List[ClassChoices]): A list of class choices for various
                components (frontend, specaug, preencoder, encoder, postencoder,
                decoder, model).
            trainer (Trainer): The class responsible for training procedures.

        Args:
            parser (argparse.ArgumentParser): The argument parser for command-line
                arguments.

        Returns:
            Callable: A callable function for collating training data.

        Yields:
            None

        Raises:
            RuntimeError: If `token_list` or `src_token_list` is not a string or list.

        Examples:
            # Adding task arguments to the parser
            MTTask.add_task_arguments(parser)

            # Building the collate function for data processing
            collate_fn = MTTask.build_collate_fn(args, train=True)

            # Building the preprocessing function
            preprocess_fn = MTTask.build_preprocess_fn(args, train=True)

            # Getting required data names for training
            required_names = MTTask.required_data_names(train=True)

            # Getting optional data names for inference
            optional_names = MTTask.optional_data_names(inference=True)

            # Building the model based on the configuration
            model = MTTask.build_model(args)

        Note:
            This class is a subclass of AbsTask and provides additional functionality
            specific to machine translation tasks.
        """
        if not inference:
            retval = ()
        else:
            retval = ()
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetMTModel:
        """
                Builds and initializes the model for the MTTask.

        This method constructs the model based on the provided arguments,
        configures various components such as the frontend, encoder,
        decoder, and CTC, and initializes the model parameters.

        Args:
            args (argparse.Namespace): The namespace containing all the arguments
                required to build the model. This includes paths to token lists,
                model configuration, and initialization settings.

        Returns:
            ESPnetMTModel: An instance of the ESPnetMTModel configured with the
                specified components.

        Raises:
            RuntimeError: If `token_list` or `src_token_list` is not a string or list.

        Examples:
            # Example of creating a model with specified arguments
            parser = argparse.ArgumentParser()
            MTTask.add_task_arguments(parser)
            args = parser.parse_args()
            model = MTTask.build_model(args)

        Note:
            Ensure that the token lists are correctly specified and accessible
            before invoking this method.

        Todo:
            - Consider implementing a more robust error handling for file I/O
              operations related to token lists.
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

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(input_size=src_vocab_size, **args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if getattr(args, "specaug", None) is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Pre-encoder input block
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
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            **args.decoder_conf,
        )

        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 8. Build model
        try:
            model_class = model_choices.get_class(args.model)
            if args.model == "discrete_asr":
                extra_model_conf = dict(ctc=ctc, specaug=specaug)
            else:
                extra_model_conf = dict()
        except AttributeError:
            model_class = model_choices.get_class("mt")
            extra_model_conf = dict()
        model = model_class(
            vocab_size=vocab_size,
            src_vocab_size=src_vocab_size,
            frontend=frontend,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            token_list=token_list,
            src_token_list=src_token_list,
            **args.model_conf,
            **extra_model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
