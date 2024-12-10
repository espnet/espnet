import argparse
import json
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

# CoreLM
from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.core_lm.ar_multiscale import MultiScaleLM
from espnet2.speechlm.core_lm.valle import ValleLM
from espnet2.speechlm.espnet_model import ESPnetSpeechLMModel
from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize

# Top-level model
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import SpeechLMPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

corelm_choices = ClassChoices(
    "corelm",
    classes=dict(
        multiscale=MultiScaleLM,
        valle=ValleLM,
    ),
    type_check=AbsCoreLM,
    default="valle",
)

tokenizer_choices = ClassChoices(
    "tokenizer",
    classes=dict(
        codec=CodecTokenizer,
    ),
    type_check=AbsTokenizer,
    default=None,
)

model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetSpeechLMModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)


class SpeechLMTask(AbsTask):
    """
        SpeechLMTask is a class for handling tasks related to Speech Language Modeling
    within the ESPnet framework. It provides methods to define, preprocess, and
    collate data for training and evaluation of speech language models.

    Attributes:
        num_optimizers (int): The number of optimizers used in the task.
        class_choices_list (List[ClassChoices]): A list of class choices for core
            language models, tokenizers, and models.
        trainer (Trainer): The trainer class used for training the models.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which task-related
            arguments will be added.

    Returns:
        argparse.ArgumentParser: The updated argument parser with task-related
            arguments.

    Yields:
        None

    Raises:
        RuntimeError: If the `token_list` or `token_bias` arguments are not
            provided in the expected format.

    Examples:
        To add task arguments to an argument parser:

            parser = argparse.ArgumentParser()
            SpeechLMTask.add_task_arguments(parser)

        To build a collate function for batching:

            collate_fn = SpeechLMTask.build_collate_fn(args, train=True)

        To build a preprocessing function:

            preprocess_fn = SpeechLMTask.build_preprocess_fn(args, train=True)

        To build the model:

            model = SpeechLMTask.build_model(args)

    Note:
        This class is a subclass of AbsTask and assumes that the necessary
        configurations and classes for the core language model, tokenizer, and
        model have been defined.

    Todo:
        - Implement additional methods for more complex training and evaluation
            procedures if necessary.
    """

    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --corelm and --corelm_conf
        corelm_choices,
        # --tokenizer and --tokenizer_conf
        tokenizer_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
        Adds task-specific arguments to the argument parser.

        This method is responsible for adding all necessary command line
        arguments related to the task configuration, including token lists,
        initialization methods, and preprocessor options. It groups the
        arguments logically for better organization and user experience.

        Args:
            cls: The class type of the current task.
            parser (argparse.ArgumentParser): The argument parser instance
                to which the arguments will be added.

        Returns:
            argparse.ArgumentParser: The updated argument parser instance.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> SpeechLMTask.add_task_arguments(parser)
            >>> args = parser.parse_args(["--token_list", "tokens.txt"])
            >>> print(args.token_list)
            tokens.txt

        Note:
            The arguments `--token_list` and `--token_bias` are required
            for task execution. The method ensures that the correct
            configurations are passed in for the SpeechLM task.

        Todo:
            Extend this method to include additional arguments for future
            task features if necessary.
        """
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list", "token_bias"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--token_bias",
            type=str_or_none,
            default=None,
            help="A json file to specify the start index of each modality",
        )
        group.add_argument(
            "--encoder_decoder_format",
            type=str2bool,
            default=False,
            help="If true, work with encoder-decoder; otherwise decoder-only",
        )
        group.add_argument(
            "--speaker_prompt_length",
            type=int,
            default=150,
            help="the length of speaker prompt, in #frame",
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

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file fo sentencepiece",
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
        group.add_argument(
            "--codec_token_per_frame",
            type=int,
            default=1,
            help="Number of original codec codes for each frame",
        )
        group.add_argument(
            "--codec_token_in_use",
            type=int_or_none,
            default=None,
            help="Number of codec codes in exact use",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

        return parser

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        """
        Build a collate function for batching input data.

        This method constructs a collate function that takes a collection of
        input data tuples and combines them into a batch. The collate function
        will pad sequences and convert them into tensors for processing by
        the model.

        Args:
            args (argparse.Namespace): The arguments parsed from command line,
                which includes configurations such as token_list.
            train (bool): A flag indicating whether the function is being
                called for training or not.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
            Tuple[List[str], Dict[str, torch.Tensor]]]: A collate function
            that can be used to batch data during training or evaluation.

        Examples:
            >>> from espnet2.tasks.speechlm import SpeechLMTask
            >>> parser = argparse.ArgumentParser()
            >>> SpeechLMTask.add_task_arguments(parser)
            >>> args = parser.parse_args(["--token_list", "token_list.txt"])
            >>> collate_fn = SpeechLMTask.build_collate_fn(args, train=True)
            >>> batch = collate_fn([("id1", {"data": np.array([1, 2, 3])}),
            ...                      ("id2", {"data": np.array([4, 5])})])
            >>> print(batch)

        Note:
            The token list should include a "<pad>" token for padding
            sequences.
        """
        int_pad = args.token_list.index("<pad>")
        return CommonCollateFn(int_pad_value=int_pad)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
        Builds the preprocessing function for the SpeechLM task.

        This function constructs a preprocessing function that is used to prepare
        input data for the SpeechLM model. It takes task-specific arguments to
        configure the preprocessing behavior, such as tokenization and cleaning.

        Args:
            cls: The class reference for the SpeechLMTask.
            args (argparse.Namespace): Command line arguments containing the
                configuration for the task.
            train (bool): A flag indicating whether the function is being built
                for training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A preprocessing function that takes a string and a dictionary
                of numpy arrays and returns a dictionary of numpy arrays.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(
            ...     token_list='path/to/token_list.txt',
            ...     token_bias='path/to/token_bias.json',
            ...     encoder_decoder_format=False,
            ...     bpemodel='path/to/bpemodel',
            ...     non_linguistic_symbols='path/to/non_linguistic_symbols.txt',
            ...     cleaner='tacotron',
            ...     g2p='method',
            ...     codec_token_per_frame=1,
            ...     codec_token_in_use=None,
            ...     speaker_prompt_length=150
            ... )
            >>> preprocess_fn = SpeechLMTask.build_preprocess_fn(args, train=True)
            >>> result = preprocess_fn("input_text", {"key": np.array([1, 2, 3])})

        Note:
            This function will always use the SpeechLMPreprocessor for its
            preprocessing needs, ensuring consistency across the task.

        Todo:
            - Add more preprocessing options in the future if necessary.
        """

        # (Jinchuan) SpeechLM task will always use the preprocess_fn
        retval = SpeechLMPreprocessor(
            token_list=args.token_list,
            token_bias=args.token_bias,
            encoder_decoder_format=args.encoder_decoder_format,
            bpemodel=args.bpemodel,
            non_linguistic_symbols=args.non_linguistic_symbols,
            text_cleaner=args.cleaner,
            g2p_type=args.g2p,
            codec_token_per_frame=args.codec_token_per_frame,
            codec_token_in_use=args.codec_token_in_use,
            speaker_prompt_length=args.speaker_prompt_length,
        )

        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns the required data names for the SpeechLMTask.

        This method provides a tuple of required data names that are necessary
        for the SpeechLMTask. The names returned by this method are typically
        used during the training and inference phases to identify the data
        that the task expects.

        Args:
            train (bool): A flag indicating if the data is for training.
                          Defaults to True.
            inference (bool): A flag indicating if the data is for inference.
                              Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the required data names.

        Examples:
            >>> required_names = SpeechLMTask.required_data_names()
            >>> print(required_names)
            ('dec_seq',)

        Note:
            The returned names can be used in data loading and processing
            functions to ensure that the correct data is being utilized
            for the task.
        """
        retval = ("dec_seq",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns the optional data names required for the SpeechLM task.

        This method provides a tuple of optional data names that can be utilized
        during the training and inference processes of the SpeechLM task. These
        data names can be used for various purposes such as managing input data
        and facilitating model training or evaluation.

        Args:
            train (bool): A flag indicating whether the function is called during
                training. Default is True.
            inference (bool): A flag indicating whether the function is called
                during inference. Default is False.

        Returns:
            Tuple[str, ...]: A tuple containing the optional data names.
                Typically includes 'enc_seq' and 'prefix_len'.

        Examples:
            >>> optional_data = SpeechLMTask.optional_data_names()
            >>> print(optional_data)
            ('enc_seq', 'prefix_len')
        """
        retval = ("enc_seq", "prefix_len")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> Union[AbsESPnetModel]:
        """
        Builds and initializes the speech language model based on the provided
        arguments.

        This method is responsible for constructing the core language model
        and the overall model using the specified configurations. It also
        handles the initialization of model parameters based on the selected
        initialization method.

        Args:
            args (argparse.Namespace): The parsed command-line arguments
                containing model configurations, token list, token bias,
                and other parameters necessary for building the model.

        Returns:
            Union[AbsESPnetModel]: An instance of the speech language model
                that has been built and initialized.

        Raises:
            RuntimeError: If `token_list` is not of type `str` or `list`,
                or if `token_bias` is not of type `str` or `dict`.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(
            ...     token_list="path/to/token_list.txt",
            ...     token_bias="path/to/token_bias.json",
            ...     corelm="valle",
            ...     model="espnet",
            ...     init="xavier_uniform",
            ...     corelm_conf={},
            ...     model_conf={}
            ... )
            >>> model = SpeechLMTask.build_model(args)
            >>> print(type(model))
            <class 'espnet2.train.abs_espnet_model.AbsESPnetModel'>

        Note:
            Ensure that the token list and token bias files are correctly
            formatted and accessible before calling this method.
        """

        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip("\n") for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or list")

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        if isinstance(args.token_bias, str):
            token_bias = json.load(open(args.token_bias))
            args.token_bias = token_bias
        elif isinstance(args.token_bias, Dict):
            token_bias = args.token_bias
        else:
            raise RuntimeError("token_list must be str or dict")
        logging.info(f"Token Bias: {token_bias}")

        # 1. Build CoreLM module
        corelm_class = corelm_choices.get_class(args.corelm)
        corelm = corelm_class(
            vocab_size=len(token_list), nq=args.codec_token_in_use, **args.corelm_conf
        )

        # 3. Build model
        model_class = model_choices.get_class(args.model)
        model = model_class(corelm=corelm, **args.model_conf)

        # 4. Initialize
        if args.init is not None:
            initialize(model, args.init)
        else:
            for m in model.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.Embedding):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        return model
