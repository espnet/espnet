import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder

# TODO(checkpoint1): import conformer class class
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asvspoof.decoder.abs_decoder import AbsDecoder
from espnet2.asvspoof.decoder.linear_decoder import LinearDecoder
from espnet2.asvspoof.espnet_model import ESPnetASVSpoofModel
from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss
from espnet2.asvspoof.loss.am_softmax_loss import ASVSpoofAMSoftmaxLoss
from espnet2.asvspoof.loss.binary_loss import ASVSpoofBinaryLoss
from espnet2.asvspoof.loss.oc_softmax_loss import ASVSpoofOCSoftmaxLoss
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
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
        # TODO(checkpoint2): add conformer option in encoder
        transformer=TransformerEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        linear=LinearDecoder,
    ),
    type_check=AbsDecoder,
    default="linear",
)
losses_choices = ClassChoices(
    name="losses",
    classes=dict(
        binary_loss=ASVSpoofBinaryLoss,
        am_softmax_loss=ASVSpoofAMSoftmaxLoss,
        oc_softmax_loss=ASVSpoofOCSoftmaxLoss,
    ),
    type_check=AbsASVSpoofLoss,
    default=None,
)


class ASVSpoofTask(AbsTask):
    """
        ASVSpoofTask is a task class for the ASVspoof challenge, inheriting from
    AbsTask. It manages the configuration of various components such as
    frontend, encoder, decoder, and loss functions, and provides methods to
    add task-specific arguments and build the model.

    Attributes:
        num_optimizers (int): Number of optimizers to use.
        class_choices_list (List[ClassChoices]): List of available class
            choices for various components.
        trainer (Trainer): Trainer class used for training.

    Args:
        parser (argparse.ArgumentParser): Argument parser to add task-related
            arguments.

    Returns:
        Callable: A function that collates data into batches.

    Yields:
        None

    Raises:
        None

    Examples:
        To add task-specific arguments to an argument parser:
            parser = argparse.ArgumentParser()
            ASVSpoofTask.add_task_arguments(parser)

        To build a model based on the provided arguments:
            args = parser.parse_args()
            model = ASVSpoofTask.build_model(args)

    Note:
        - The `add_task_arguments` method adds arguments related to the task
          to the given parser.
        - The `build_model` method constructs the ASVSpoof model using the
          specified configurations for the frontend, encoder, decoder,
          and losses.

    Todo:
        - Implement conformer class for encoder.
        - Add conformer option in encoder choices.
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
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
            Adds task-related arguments to the given argument parser.

        This method defines various command-line arguments that are specific to the
        ASVSpoof task. These arguments include configurations for initialization,
        input size, preprocessing, and loss functions, as well as options for
        frontend, specaug, normalization, preencoder, encoder, and decoder.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the task
                arguments will be added.

        Examples:
            To use this method, you can set up an argument parser and add the task
            arguments like so:

            ```python
            import argparse
            from espnet2.asvspoof.task import ASVSpoofTask

            parser = argparse.ArgumentParser()
            ASVSpoofTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            The method cannot use `required=True` for arguments to allow the use of
            `--print_config` mode.
        """
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
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
            "--losses",
            action=NestedDictAction,
            default=[
                {
                    "name": "sigmoid_loss",
                    "conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        """
        Builds a collate function for batching data during training or evaluation.

        This method returns a callable that can be used to collate a list of
        samples into a batch. The collate function is designed to handle the
        specific input format required for the ASVSpoofTask.

        Args:
            args (argparse.Namespace): The parsed arguments containing task
                configurations.
            train (bool): A flag indicating whether the collate function is
                being built for training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]]:
                A collate function that takes a collection of tuples, each
                containing a sample identifier and a dictionary of feature
                arrays, and returns a tuple of a list of identifiers and a
                dictionary of tensors.

        Note:
            The function uses `CommonCollateFn` to handle padding and batching
            of the input data. The integer value `-1` is reserved for CTC-blank
            symbols, and `0.0` is used for float padding.

        Examples:
            >>> collate_fn = ASVSpoofTask.build_collate_fn(args, train=True)
            >>> batch = collate_fn([("sample1", {"feature": np.array([1, 2])}),
            ...                      ("sample2", {"feature": np.array([3, 4, 5])})])
            >>> print(batch)
            (['sample1', 'sample2'], {'feature': tensor([[1, 2], [3, 4, 5]])})
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
        Build a preprocessing function based on the provided arguments.

        This method creates a preprocessing function that can be used to apply
        transformations to the input data before it is fed into the model. If
        preprocessing is enabled through the `use_preprocessor` argument, it
        initializes a `CommonPreprocessor` with the appropriate settings.

        Args:
            args (argparse.Namespace): The command-line arguments containing
                settings for the preprocessing.
            train (bool): A flag indicating whether the function is being
                built for training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
                A callable preprocessing function that takes a file path and
                a dictionary of input features and returns a dictionary of
                processed features. Returns None if preprocessing is disabled.

        Examples:
            >>> args = argparse.Namespace(use_preprocessor=True)
            >>> preprocess_fn = ASVSpoofTask.build_preprocess_fn(args, train=True)
            >>> processed_data = preprocess_fn("path/to/audio.wav", {"feature": data})

        Note:
            The `CommonPreprocessor` will be initialized only if
            `args.use_preprocessor` is set to True. Otherwise, the function
            will return None, indicating no preprocessing is to be applied.

        Todo:
            - Extend preprocessing options based on future requirements.
        """
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
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

        This method provides a tuple of required data names based on the
        training and inference modes. If inference mode is not active, both
        "speech" and "label" data names are required. In inference mode, only
        the "speech" data name is needed.

        Args:
            train (bool): A flag indicating if the task is in training mode.
                          Defaults to True.
            inference (bool): A flag indicating if the task is in inference mode.
                              Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the required data names.

        Examples:
            >>> ASVSpoofTask.required_data_names(train=True, inference=False)
            ('speech', 'label')

            >>> ASVSpoofTask.required_data_names(train=False, inference=True)
            ('speech',)
        """
        if not inference:
            retval = ("speech", "label")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Class representing the ASVSpoofTask for automatic speaker verification
        spoofing detection.

        This task defines various components including frontend, encoder,
        decoder, and loss functions for building a model that can detect
        spoofing in audio data. It provides methods for adding task-specific
        arguments, building collate and preprocess functions, and creating
        the model itself.

        Attributes:
            num_optimizers (int): The number of optimizers to use.
            class_choices_list (list): A list of class choices for task
                components including frontend, specaug, normalization,
                preencoder, encoder, and decoder.
            trainer (Trainer): The class used for training the model.

        Args:
            parser (argparse.ArgumentParser): The argument parser to add
                task-related arguments.

        Returns:
            None

        Examples:
            To add task arguments to an argument parser:

            ```python
            import argparse
            from espnet2.asvspoof import ASVSpoofTask

            parser = argparse.ArgumentParser()
            ASVSpoofTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

            To build a model with the specified arguments:

            ```python
            model = ASVSpoofTask.build_model(args)
            ```

        Note:
            This task may require specific configurations for each of the
            components involved, and users should refer to the documentation
            for each component for detailed setup instructions.

        Todo:
            - Add conformer class options for the encoder.
            - Implement additional features as needed.
        """
        retval = ()
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetASVSpoofModel:
        """
                Builds and returns an ESPnetASVSpoofModel based on the provided configuration.

        This method constructs the model architecture for the ASVSpoof task by
        assembling various components such as frontend, spec augmentation,
        normalization, pre-encoder, encoder, decoder, and loss functions based
        on the arguments passed.

        Attributes:
            num_optimizers (int): The number of optimizers used in training. Default is 1.
            class_choices_list (list): A list of class choices for frontend,
                specaug, normalize, preencoder, encoder, and decoder.

        Args:
            args (argparse.Namespace): A namespace object containing the model
                configuration parameters.

        Returns:
            ESPnetASVSpoofModel: An instance of the ESPnetASVSpoofModel built
            according to the specified configuration.

        Raises:
            ValueError: If the configuration is invalid or if required arguments
            are missing.

        Examples:
            >>> from espnet2.asvspoof.tasks import ASVSpoofTask
            >>> args = parser.parse_args(["--frontend", "default", "--input_size", "40"])
            >>> model = ASVSpoofTask.build_model(args)

        Note:
            The method will automatically select components based on the arguments
            provided. If `args.input_size` is None, the input size will be derived
            from the frontend. The method initializes the model with the specified
            initialization method if provided.
        """

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
        encoder_output_size = encoder.output_size()

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            encoder_output_size=encoder_output_size,
            **args.decoder_conf,
        )

        # 6. Loss definition
        losses = {}
        if getattr(args, "losses", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.losses:
                if "softmax" in ctr["name"]:
                    loss = losses_choices.get_class(ctr["name"])(
                        enc_dim=encoder_output_size, **ctr["conf"]
                    )
                else:
                    loss = losses_choices.get_class(ctr["name"])(**ctr["conf"])
                losses[ctr["name"]] = loss

        # 7. Build model
        model = ESPnetASVSpoofModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            decoder=decoder,
            losses=losses,
        )

        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
