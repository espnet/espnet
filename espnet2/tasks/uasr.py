import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.uasr_trainer import UASRTrainer
from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.uasr.discriminator.conv_discriminator import ConvDiscriminator
from espnet2.uasr.espnet_model import ESPnetUASRModel
from espnet2.uasr.generator.abs_generator import AbsGenerator
from espnet2.uasr.generator.conv_generator import ConvGenerator
from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.uasr.loss.discriminator_loss import UASRDiscriminatorLoss
from espnet2.uasr.loss.gradient_penalty import UASRGradientPenalty
from espnet2.uasr.loss.phoneme_diversity_loss import UASRPhonemeDiversityLoss
from espnet2.uasr.loss.pseudo_label_loss import UASRPseudoLabelLoss
from espnet2.uasr.loss.smoothness_penalty import UASRSmoothnessPenalty
from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter
from espnet2.uasr.segmenter.join_segmenter import JoinSegmenter
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

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
segmenter_choices = ClassChoices(
    name="segmenter",
    classes=dict(
        join=JoinSegmenter,
    ),
    type_check=AbsSegmenter,
    default=None,
    optional=True,
)
discriminator_choices = ClassChoices(
    name="discriminator",
    classes=dict(
        conv=ConvDiscriminator,
    ),
    type_check=AbsDiscriminator,
    default="conv",
)
generator_choices = ClassChoices(
    name="generator",
    classes=dict(
        conv=ConvGenerator,
    ),
    type_check=AbsGenerator,
    default="conv",
)
loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        discriminator_loss=UASRDiscriminatorLoss,
        gradient_penalty=UASRGradientPenalty,
        smoothness_penalty=UASRSmoothnessPenalty,
        phoneme_diversity_loss=UASRPhonemeDiversityLoss,
        pseudo_label_loss=UASRPseudoLabelLoss,
    ),
    type_check=AbsUASRLoss,
    default="discriminator_loss",
)


class UASRTask(AbsTask):
    """
        UASRTask is a class for Unsupervised ASR (Automatic Speech Recognition) tasks.

    This class inherits from AbsTask and provides functionalities to manage
    the configurations, training, and evaluation of UASR models. It includes
    methods to build collate functions, preprocess functions, and the UASR
    model itself. The class also defines various choices for frontends,
    segmenters, discriminators, generators, and loss functions.

    Attributes:
        num_optimizers (int): Number of optimizers used in training.
        class_choices_list (List[ClassChoices]): List of class choices
            for frontend, segmenter, discriminator, generator, and loss.
        trainer (Type[UASRTrainer]): Trainer class used for training
            procedures.

    Args:
        parser (argparse.ArgumentParser): Argument parser to add task-related
            arguments.

    Returns:
        Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]]]: A function that
        collates the input data for training or evaluation.

    Yields:
        Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        A function that preprocesses the input data.

    Raises:
        RuntimeError: If the token list is not a string or a list.

    Examples:
        # To add task arguments to the parser
        UASRTask.add_task_arguments(parser)

        # To build the collate function
        collate_fn = UASRTask.build_collate_fn(args, train=True)

        # To build the preprocess function
        preprocess_fn = UASRTask.build_preprocess_fn(args, train=True)

        # To build the model
        model = UASRTask.build_model(args)

        # To build optimizers for the model
        optimizers = UASRTask.build_optimizers(args, model)

    Note:
        If you need to modify the training or evaluation procedures,
        you can change the trainer class here.

    Todo:
        - Consider adding more frontend or loss options in the future.
    """

    # If you need more than one optimizers, change this value
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --segmenter and --segmenter_conf
        segmenter_choices,
        # --discriminator and --discriminator_conf
        discriminator_choices,
        # --generator and --generator_conf
        generator_choices,
        loss_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = UASRTrainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
                Adds task-specific arguments to the argument parser.

        This method is responsible for adding various command-line arguments that are
        specific to the UASR task. These arguments can include configurations for the
        frontend, segmenter, discriminator, generator, and loss functions, as well as
        other task-related parameters.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which task
                arguments will be added.

        Examples:
            To use the `add_task_arguments` method, create an argument parser and
            call the method as follows:

            ```python
            import argparse
            from uasr_task import UASRTask

            parser = argparse.ArgumentParser()
            UASRTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            This method cannot use `add_arguments(..., required=True)` for the `--print_config`
            mode, hence the workaround with required tokens.
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
            choices=["phn"],
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
            "--losses",
            action=NestedDictAction,
            default=[
                {
                    "name": "discriminator_loss",
                    "conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
            # Loss format would be like:
            # losses:
            #   - name: loss1
            #     conf:
            #       weight: 1.0
            #       smoothed: false
            #   - name: loss2
            #     conf:
            #       weight: 0.1
            #       smoothed: false
        )
        group = parser.add_argument_group(description="Task related")
        group.add_argument(
            "--kenlm_path",
            type=str,
            help="path of n-gram kenlm for validation",
        )

        parser.add_argument(
            "--int_pad_value",
            type=int,
            default=0,
            help="Integer padding value for real token sequence",
        )

        parser.add_argument(
            "--fairseq_checkpoint",
            type=str,
            help="Fairseq checkpoint to initialize model",
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
            Build a collate function for the UASRTask.

        This method constructs a collate function that is used to process a batch
        of data during training or evaluation. The collate function handles padding
        of input sequences and ensures that the data is in the correct format for
        further processing.

        Args:
            args (argparse.Namespace): Command line arguments containing configuration
                settings such as padding values.
            train (bool): A flag indicating whether the collate function is being
                built for training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]: A callable collate
                     function that takes a collection of tuples and returns a tuple
                     containing a list of keys and a dictionary of padded tensors.

        Examples:
            >>> collate_fn = UASRTask.build_collate_fn(args, train=True)
            >>> batch = [("id1", {"feature": np.array([1.0, 2.0])}),
            ...          ("id2", {"feature": np.array([3.0, 4.0, 5.0])})]
            >>> keys, padded_data = collate_fn(batch)

        Note:
            The padding values are determined by the `int_pad_value` attribute
            from the provided arguments. It is important to ensure that the
            data passed to this function is in the expected format to avoid
            errors during processing.
        """
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=args.int_pad_value)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
            Builds a preprocessing function based on the provided arguments.

        This method returns a callable that can preprocess the input data
        depending on the configuration specified in the arguments. If
        preprocessing is not enabled, it returns None.

        Args:
            cls: The class type of the UASRTask.
            args (argparse.Namespace): Command-line arguments containing
                configurations for preprocessing.
            train (bool): A flag indicating whether the function is being
                built for training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A preprocessing function that takes a string input and a
                dictionary of feature data, returning a processed dictionary
                of features, or None if preprocessing is disabled.

        Examples:
            To build a preprocessing function for training:

            ```python
            preprocess_fn = UASRTask.build_preprocess_fn(args, train=True)
            ```

            To use the preprocessing function on some input data:

            ```python
            processed_data = preprocess_fn(input_data, feature_dict)
            ```

        Note:
            Ensure that `args.use_preprocessor` is set to True in order
            for the preprocessing function to be created.
        """
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
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
            Returns the names of the required data for the UASR task.

        The method determines which data is necessary based on whether the
        task is in training or inference mode. In training mode, both
        "speech" and "text" data are required. In inference mode, only
        "speech" data is required.

        Args:
            train (bool): A flag indicating whether the task is in training
                mode. Defaults to True.
            inference (bool): A flag indicating whether the task is in
                inference mode. Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of the required
            data.

        Examples:
            >>> UASRTask.required_data_names(train=True, inference=False)
            ('speech', 'text')

            >>> UASRTask.required_data_names(train=False, inference=True)
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
            Returns the optional data names used in the UASR task.

        This method provides a tuple of strings representing the names of optional
        data that can be utilized during training or inference in the UASR task.
        By default, it includes "pseudo_labels" and "input_cluster_id". This
        allows for greater flexibility in managing the data that the task can
        operate on.

        Args:
            train (bool): A flag indicating whether the data is for training.
                          Defaults to True.
            inference (bool): A flag indicating whether the data is for inference.
                              Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the optional data names.

        Examples:
            >>> optional_data = UASRTask.optional_data_names()
            >>> print(optional_data)
            ('pseudo_labels', 'input_cluster_id')

        Note:
            This method is part of the UASRTask class which is designed for
            Unsupervised Automatic Speech Recognition tasks.
        """
        retval = ("pseudo_labels", "input_cluster_id")
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetUASRModel:
        """
                Builds and returns an instance of the ESPnetUASRModel.

        This method constructs the model based on the provided arguments. It
        initializes various components of the model, such as the frontend,
        segmenter, discriminator, generator, and loss functions. The function
        also handles loading pre-trained weights from a Fairseq checkpoint if
        specified.

        Args:
            args (argparse.Namespace): Command line arguments containing
                configuration options for the model components.

        Returns:
            ESPnetUASRModel: An instance of the ESPnetUASRModel class.

        Raises:
            RuntimeError: If `token_list` is neither a string nor a list.

        Examples:
            # Example usage of build_model
            parser = argparse.ArgumentParser()
            UASRTask.add_task_arguments(parser)
            args = parser.parse_args()
            model = UASRTask.build_model(args)

        Note:
            The method assumes that `args.token_list` is provided and that it
            contains the necessary information for building the model.
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
        logging.info(f"Vocabulary size: {vocab_size}")

        # load from fairseq checkpoint
        load_fairseq_model = False
        cfg = None
        if args.fairseq_checkpoint is not None:
            load_fairseq_model = True
            ckpt = args.fairseq_checkpoint
            logging.info(f"Loading parameters from fairseq: {ckpt}")

            state_dict = torch.load(ckpt)
            if "cfg" in state_dict and state_dict["cfg"] is not None:
                model_cfg = state_dict["cfg"]["model"]
                logging.info(f"Building model from {model_cfg}")
            else:
                raise RuntimeError(f"Bad 'cfg' in state_dict of {ckpt}")

        # 1. frontend
        if args.write_collected_feats:
            # Extract features in the model
            # Note(jiatong): if we use write_collected_feats=True (we use
            #                pre-extracted feature for training): we still initial
            #                frontend to allow inference with raw speech signal
            #                but the frontend is not used in training
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            if args.input_size is None:
                input_size = frontend.output_size()
            else:
                input_size = args.input_size
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Segmenter
        if args.segmenter is not None:
            segmenter_class = segmenter_choices.get_class(args.segmenter)
            segmenter = segmenter_class(cfg=cfg, **args.segmenter_conf)
        else:
            segmenter = None

        # 3. Discriminator
        discriminator_class = discriminator_choices.get_class(args.discriminator)
        discriminator = discriminator_class(
            cfg=cfg, input_dim=vocab_size, **args.discriminator_conf
        )

        # 4. Generator
        generator_class = generator_choices.get_class(args.generator)
        generator = generator_class(
            cfg=cfg, input_dim=input_size, output_dim=vocab_size, **args.generator_conf
        )

        # 5. Loss definition
        losses = {}
        if getattr(args, "losses", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.losses:
                logging.info("initialize loss: {}".format(ctr["name"]))
                if ctr["name"] == "gradient_penalty":
                    loss = loss_choices.get_class(ctr["name"])(
                        discriminator=discriminator, **ctr["conf"]
                    )
                else:
                    loss = loss_choices.get_class(ctr["name"])(**ctr["conf"])
                losses[ctr["name"]] = loss

        # 6. Build model
        logging.info(f"kenlm_path is: {args.kenlm_path}")
        model = ESPnetUASRModel(
            cfg=cfg,
            frontend=frontend,
            segmenter=segmenter,
            discriminator=discriminator,
            generator=generator,
            losses=losses,
            kenlm_path=args.kenlm_path,
            token_list=args.token_list,
            max_epoch=args.max_epoch,
            vocab_size=vocab_size,
            use_collected_training_feats=args.write_collected_feats,
        )

        # FIXME(kamo): Should be done in model?
        # 7. Initialize
        if load_fairseq_model:
            logging.info(f"Initializing model from {ckpt}")
            model.load_state_dict(state_dict["model"], strict=False)
        else:
            if args.init is not None:
                initialize(model, args.init)

        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetUASRModel,
    ) -> List[torch.optim.Optimizer]:
        """
            Build optimizers for the UASR model.

        This method constructs and returns a list of optimizers for the model's
        generator and discriminator. The method checks for the presence of the
        necessary parameters in the model and creates the optimizers based on
        the provided configuration.

        Args:
            args (argparse.Namespace): The argument namespace containing
                configuration options for the optimizers.
            model (ESPnetUASRModel): The UASR model for which optimizers are
                being built.

        Returns:
            List[torch.optim.Optimizer]: A list of optimizers for the model's
            generator and discriminator.

        Raises:
            ValueError: If the specified optimizer classes are not found in the
                available optimizer classes.

        Examples:
            >>> from espnet2.uasr.espnet_model import ESPnetUASRModel
            >>> args = argparse.Namespace(optim='adam', optim_conf={'lr': 0.001},
            ...                            optim2='sgd', optim2_conf={'lr': 0.01})
            >>> model = ESPnetUASRModel(...)
            >>> optimizers = UASRTask.build_optimizers(args, model)
            >>> len(optimizers)
            2

        Note:
            Ensure that the model has the required attributes (`generator` and
            `discriminator`) before calling this method.
        """
        # check
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")

        generator_param_list = list(model.generator.parameters())
        discriminator_param_list = list(model.discriminator.parameters())

        # Add optional sets of model parameters
        if model.use_segmenter is not None:
            generator_param_list += list(model.segmenter.parameters())
        if (
            "pseudo_label_loss" in model.losses.keys()
            and model.losses["pseudo_label_loss"].weight > 0
        ):
            generator_param_list += list(
                model.losses["pseudo_label_loss"].decoder.parameters()
            )

        # define generator optimizer
        optim_generator_class = optim_classes.get(args.optim)
        if optim_generator_class is None:
            raise ValueError(
                f"must be one of {list(optim_classes)}: {args.optim_generator}"
            )
        optim_generator = optim_generator_class(
            generator_param_list,
            **args.optim_conf,
        )
        optimizers = [optim_generator]

        # define discriminator optimizer
        optim_discriminator_class = optim_classes.get(args.optim2)
        if optim_discriminator_class is None:
            raise ValueError(
                f"must be one of {list(optim_classes)}: {args.optim_discriminator}"
            )
        optim_discriminator = optim_discriminator_class(
            discriminator_param_list,
            **args.optim2_conf,
        )
        optimizers += [optim_discriminator]

        return optimizers
