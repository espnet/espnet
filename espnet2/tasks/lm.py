import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM
from espnet2.lm.espnet_model import ESPnetLanguageModel
from espnet2.lm.espnet_model_multitask import ESPnetMultitaskLanguageModel
from espnet2.lm.huggingface_pretrained_opt_lm import HuggingfaceOPTModel
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.types import str2bool, str_or_none

lm_choices = ClassChoices(
    "lm",
    classes=dict(
        seq_rnn=SequentialRNNLM,
        transformer=TransformerLM,
        transformer_opt=HuggingfaceOPTModel,
    ),
    type_check=AbsLM,
    default="seq_rnn",
)

model_choices = ClassChoices(
    "model",
    classes=dict(
        lm=ESPnetLanguageModel,
        lm_multitask=ESPnetMultitaskLanguageModel,
    ),
    type_check=AbsESPnetModel,
    default="lm",
)


class LMTask(AbsTask):
    """
        LMTask is a class that handles the language modeling tasks in the ESPnet2
    framework. It extends the AbsTask class and provides methods for argument
    parsing, model building, data preprocessing, and collation of data for
    training and evaluation.

    Attributes:
        num_optimizers (int): The number of optimizers to be used (default is 1).
        class_choices_list (list): A list of ClassChoices instances for models
            and language models.
        trainer (Trainer): The Trainer class to be used for training and
            evaluation.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which task
            related arguments will be added.

    Returns:
        argparse.ArgumentParser: The updated argument parser with task related
            arguments.

    Yields:
        Callable: A function that collates data for training or evaluation.

    Raises:
        RuntimeError: If the token_list is not of type str or dict.

    Examples:
        # To add task arguments to a parser
        parser = argparse.ArgumentParser()
        LMTask.add_task_arguments(parser)

        # To build a model
        args = parser.parse_args()
        model = LMTask.build_model(args)

    Note:
        If you need to modify the training or evaluation procedures, you can
        change the Trainer class assigned to the 'trainer' attribute.

    Todo:
        Implement additional features for improved model training and
        evaluation.
    """

    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        lm_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
            Adds command-line arguments specific to the language modeling task.

        This method configures the argument parser with options relevant to the
        language modeling task, including token lists, initialization methods,
        and preprocessing options. It allows users to specify various parameters
        that control the behavior of the task during execution.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the
                task-related arguments will be added.

        Returns:
            argparse.ArgumentParser: The updated argument parser with task
                arguments included.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> LMTask.add_task_arguments(parser)
            >>> args = parser.parse_args(["--token_list", "path/to/token_list.txt"])
            >>> print(args.token_list)
            path/to/token_list.txt

        Note:
            The `--token_list` argument is crucial for mapping integer IDs to
            tokens. The initialization method can be chosen from a predefined
            set of options. Additionally, preprocessing options can be enabled
            or disabled.

        Todo:
            - Consider adding more preprocessing options in the future.
        """
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
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
            choices=["bpe", "char", "word"],
            help="",
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
        Builds a collate function for batching data.

        This method constructs a callable that collates a batch of data
        samples into a single tensor, ensuring that the sequences are
        properly padded to the same length. The resulting batch will
        include both the tokenized text and corresponding feature tensors.

        Args:
            args (argparse.Namespace): The command-line arguments containing
                configuration options for the task.
            train (bool): A flag indicating whether the collate function
                is being built for training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]:
                A function that takes a collection of samples and returns
                a tuple containing a list of texts and a dictionary of
                tensors.

        Examples:
            >>> collate_fn = LMTask.build_collate_fn(args, train=True)
            >>> batch_samples = [
            ...     ("sample1", {"feature": np.array([1, 2, 3])}),
            ...     ("sample2", {"feature": np.array([4, 5])}),
            ... ]
            >>> texts, features = collate_fn(batch_samples)
            >>> print(texts)
            ['sample1', 'sample2']
            >>> print(features)
            {'feature': tensor([[1, 2, 3],
                                 [4, 5, 0]])}  # Example padded tensor

        Note:
            This method uses the CommonCollateFn for the actual
            implementation of the collate functionality.
        """
        return CommonCollateFn(int_pad_value=0)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
            Builds a preprocessing function based on the provided arguments.

        This method checks if preprocessing is enabled and returns a
        CommonPreprocessor instance configured with the specified parameters.
        If preprocessing is not enabled, it returns None.

        Args:
            cls: The class reference (usually the class itself).
            args (argparse.Namespace): The namespace containing command-line
                arguments.
            train (bool): A flag indicating whether the preprocessing is for
                training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A callable preprocessing function if preprocessing is enabled,
                otherwise None.

        Examples:
            To create a preprocessing function for training:

            >>> args = argparse.Namespace(use_preprocessor=True, token_type='bpe',
            ... token_list='tokens.txt', bpemodel='model.bpe',
            ... cleaner='tacotron', g2p=None, non_linguistic_symbols=None)
            >>> preprocess_fn = LMTask.build_preprocess_fn(args, train=True)
            >>> output = preprocess_fn("sample text", {"key": np.array([1, 2, 3])})

            If preprocessing is disabled:

            >>> args.use_preprocessor = False
            >>> preprocess_fn = LMTask.build_preprocess_fn(args, train=True)
            >>> assert preprocess_fn is None

        Note:
            This method relies on the CommonPreprocessor class for
            preprocessing tasks, which should be defined elsewhere in the
            codebase.

        Todo:
            Consider adding more preprocessing options in the future.
        """
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                non_linguistic_symbols=args.non_linguistic_symbols,
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns the required data names for the task.

        This method specifies the data names that are essential for the
        training or inference of the language model task. By default, it
        returns a tuple containing the string "text", which indicates
        that text data is required.

        Args:
            train (bool, optional): A flag indicating whether the method is
                called for training. Defaults to True.
            inference (bool, optional): A flag indicating whether the method
                is called for inference. Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple of required data names. In this case,
            it will return ("text",).

        Examples:
            >>> LMTask.required_data_names(train=True)
            ('text',)

            >>> LMTask.required_data_names(inference=True)
            ('text',)
        """
        retval = ("text",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns the optional data names required by the task.

        This method returns a tuple of optional data names that the task may use
        during training or inference. By default, it returns an empty tuple,
        indicating that there are no optional data names.

        Args:
            train (bool): Indicates whether the data is for training. Default is
                True.
            inference (bool): Indicates whether the data is for inference.
                Default is False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of optional data.

        Examples:
            >>> optional_names = LMTask.optional_data_names()
            >>> print(optional_names)
            ()  # Returns an empty tuple by default

        Note:
            This method can be overridden in subclasses to specify additional
            optional data names.
        """
        retval = ()
        return retval

    @classmethod
    @typechecked
    def build_model(
        cls, args: argparse.Namespace
    ) -> Union[ESPnetLanguageModel, ESPnetMultitaskLanguageModel]:
        """
                Builds and initializes the language model based on the provided arguments.

        This method constructs a language model using the specified model type and
        configuration parameters. It handles the creation of both the language model
        and the ESPnet model, initializing them with the appropriate parameters.

        Args:
            args (argparse.Namespace): The parsed command line arguments containing
                model configuration options, including the token list and model type.

        Returns:
            Union[ESPnetLanguageModel, ESPnetMultitaskLanguageModel]: An instance of
            the selected language model class.

        Raises:
            RuntimeError: If the token_list is not provided as a string or list.

        Examples:
            To build a model with a token list specified:

                import argparse

                parser = argparse.ArgumentParser()
                parser.add_argument('--token_list', type=str, required=True)
                parser.add_argument('--lm', type=str, default='seq_rnn')
                parser.add_argument('--model', type=str, default='lm')
                args = parser.parse_args()

                model = LMTask.build_model(args)

        Note:
            This method assumes that the token list is either a path to a file
            containing the tokens or a list of tokens. The vocabulary size is
            derived from the token list, which is essential for initializing
            the language model correctly.
        """
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

        # 1. Build LM model
        lm_class = lm_choices.get_class(args.lm)
        lm = lm_class(vocab_size=vocab_size, **args.lm_conf)

        # 2. Build ESPnetModel
        # Assume the last-id is sos_and_eos
        try:
            model_class = model_choices.get_class(args.model)
            if args.model == "lm_multitask":
                extra_model_conf = dict(token_list=token_list)
            else:
                extra_model_conf = dict()
        except AttributeError:
            model_class = model_choices.get_class("lm")
            extra_model_conf = dict()

        model = model_class(
            lm=lm, vocab_size=vocab_size, **args.model_conf, **extra_model_conf
        )

        # FIXME(kamo): Should be done in model?
        # 3. Initialize
        if args.init is not None:
            initialize(model, args.init)

        if args.lm == "transformer_opt":
            # loading opt parameters
            model.lm.reload_pretrained_parameters()

        return model
