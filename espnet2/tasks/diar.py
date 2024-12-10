import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.diar.attractor.abs_attractor import AbsAttractor
from espnet2.diar.attractor.rnn_attractor import RnnAttractor
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.diar.decoder.linear_decoder import LinearDecoder
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.label_aggregation import LabelAggregate
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
    optional=True,
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
label_aggregator_choices = ClassChoices(
    "label_aggregator",
    classes=dict(label_aggregator=LabelAggregate),
    default="label_aggregator",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(linear=LinearDecoder),
    type_check=AbsDecoder,
    default="linear",
)
attractor_choices = ClassChoices(
    "attractor",
    classes=dict(
        rnn=RnnAttractor,
    ),
    type_check=AbsAttractor,
    default=None,
    optional=True,
)


class DiarizationTask(AbsTask):
    """
        DiarizationTask is a task class for speaker diarization that extends the
    abstract task class (AbsTask). It encapsulates the necessary configurations,
    model building, and data handling required for training and evaluating a
    diarization model.

    Attributes:
        num_optimizers (int): The number of optimizers to use.
        class_choices_list (list): List of available class choices for task
            configurations, including frontend, specaug, normalization, encoder,
            decoder, label aggregator, and attractor.
        trainer (Trainer): The trainer class to use for training and evaluation.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the task
            related arguments will be added.

    Returns:
        None

    Yields:
        None

    Raises:
        None

    Examples:
        To add task-specific arguments to a parser, use:

            parser = argparse.ArgumentParser()
            DiarizationTask.add_task_arguments(parser)

        To build a model, first parse the arguments and then call:

            args = parser.parse_args()
            model = DiarizationTask.build_model(args)

    Note:
        This class requires the definition of various components including
        frontends, encoders, decoders, and others as specified in the
        class_choices_list.

    Todo:
        - Extend support for more data processing options.
        - Improve error handling and input validation.
    """

    # If you need more than one optimizer, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --label_aggregator and --label_aggregator_conf
        label_aggregator_choices,
        # --attractor and --attractor_conf
        attractor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
            Adds task-related arguments to the argument parser for the DiarizationTask.

        This method extends the provided argument parser by adding various
        command-line arguments related to the diarization task. These arguments
        include configurations for the number of speakers, initialization methods,
        input size, and model configurations, as well as preprocessing options.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the
                task-related arguments will be added.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> DiarizationTask.add_task_arguments(parser)
            >>> args = parser.parse_args(["--num_spk", "2", "--init", "xavier_uniform"])

        Note:
            Ensure that the argument parser is properly initialized before calling
            this method.

        Raises:
            ValueError: If an invalid argument is provided or if there are
                conflicting options.

        Todo:
            - Add more specific argument descriptions and validation checks
              as needed in future iterations.
        """
        group = parser.add_argument_group(description="Task related")

        group.add_argument(
            "--num_spk",
            type=int_or_none,
            default=None,
            help="The number fo speakers (for each recording) used in system training",
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
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetDiarizationModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
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

    This method constructs a callable that takes a collection of tuples, each 
    containing a string and a dictionary of NumPy arrays, and returns a tuple 
    of a list of strings and a dictionary of PyTorch tensors. The collate 
    function is designed to handle variable-length input sequences by padding 
    them to a uniform length.

    Args:
        args (argparse.Namespace): The command-line arguments containing the 
            configuration for the task.
        train (bool): A flag indicating whether the function is being called 
            during training or evaluation.

    Returns:
        Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]], 
        Tuple[List[str], Dict[str, torch.Tensor]]]: A collate function that 
        processes the input data into a batched format.

    Note:
        The integer value `0` is reserved for the CTC-blank symbol.

    Examples:
        >>> collate_fn = DiarizationTask.build_collate_fn(args, train=True)
        >>> batch = collate_fn([("utt1", {"feature": np.array([1, 2])}),
        ...                     ("utt2", {"feature": np.array([3])})])
        >>> print(batch)
        (['utt1', 'utt2'], {'feature': tensor([[1, 2],
                                               [3, 0]])})
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
        Build a preprocessing function based on the task arguments.

        This method constructs a preprocessing function that applies
        transformations to the input data if the `--use_preprocessor`
        argument is set to True. If not, it returns None.

        Args:
            args (argparse.Namespace): The parsed command line arguments.
            train (bool): A flag indicating whether the function is being
                built for training or evaluation.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]],
            Dict[str, np.ndarray]]]: A preprocessing function that takes
            the input data and returns the processed output, or None
            if preprocessing is not to be applied.

        Examples:
            >>> args = argparse.Namespace(use_preprocessor=True)
            >>> preprocess_fn = DiarizationTask.build_preprocess_fn(args, train=True)
            >>> output = preprocess_fn("input.wav", {"feature": np.array([1, 2, 3])})

        Note:
            The preprocessing function can be modified to include more
            arguments in the future.
        """
        if args.use_preprocessor:
            # FIXME (jiatong): add more argument here
            retval = CommonPreprocessor(train=train)
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Get the required data names for the DiarizationTask.

        This method returns the names of the data that are required for training
        or inference in the diarization task. The output varies based on whether
        the task is in inference mode or not.

        Args:
            train (bool, optional): A flag indicating whether the task is in
                training mode. Defaults to True.
            inference (bool, optional): A flag indicating whether the task is
                in inference mode. Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the names of the required data.
                If not in inference mode, it returns ("speech", "spk_labels").
                If in inference mode, it returns ("speech",).

        Examples:
            >>> DiarizationTask.required_data_names(train=True, inference=False)
            ('speech', 'spk_labels')

            >>> DiarizationTask.required_data_names(train=False, inference=True)
            ('speech',)

        Note:
            The method is designed to return the appropriate data names based on
            the mode of operation (training or inference).
        """
        if not inference:
            retval = ("speech", "spk_labels")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Class for handling the diarization task within the ESPnet framework.

        This class extends the abstract task class `AbsTask` and provides
        methods for managing various components required for diarization
        including the frontend, spec augmentation, normalization, encoder,
        decoder, and attractor. It also handles the configuration of these
        components through command-line arguments and builds the model based
        on the provided configuration.

        Attributes:
            num_optimizers (int): Number of optimizers used in training.
            class_choices_list (List[ClassChoices]): List of class choices for
                various components of the diarization task.
            trainer (Trainer): Trainer class used for training and evaluation.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which
                task-related arguments will be added.

        Returns:
            Tuple[str, ...]: Required data names based on the mode of operation.

        Yields:
            Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
            A function that preprocesses the input data.

        Raises:
            ValueError: If any configuration parameter is invalid.

        Examples:
            # To add task-related arguments to the parser
            DiarizationTask.add_task_arguments(parser)

            # To build the model from command line arguments
            model = DiarizationTask.build_model(args)

            # To get required data names
            data_names = DiarizationTask.required_data_names(train=True)

            # To get optional data names
            optional_data_names = DiarizationTask.optional_data_names(train=True)
        """
        # (Note: jiatong): no optional data names for now
        retval = ()
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetDiarizationModel:
        """
                Builds the ESPnetDiarizationModel based on the provided configuration.

        This method constructs a diarization model by initializing its components
        such as frontend, data augmentation, normalization, label aggregator,
        encoder, decoder, and attractor based on the arguments provided.

        Args:
            args (argparse.Namespace): The arguments namespace containing
                configuration options for model building.

        Returns:
            ESPnetDiarizationModel: An instance of the ESPnetDiarizationModel
                constructed with the specified configurations.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     input_size=128,
            ...     frontend='default',
            ...     frontend_conf={},
            ...     specaug=None,
            ...     normalize='utterance_mvn',
            ...     label_aggregator='label_aggregator',
            ...     encoder='transformer',
            ...     encoder_conf={},
            ...     decoder='linear',
            ...     decoder_conf={},
            ...     attractor=None,
            ...     model_conf={}
            ... )
            >>> model = DiarizationTask.build_model(args)
            >>> print(type(model))
            <class 'espnet2.diar.espnet_model.ESPnetDiarizationModel'>

        Note:
            This method assumes that the necessary classes for frontend,
            specaug, normalize, label aggregator, encoder, decoder, and
            attractor are properly registered in their respective class choices.

        Todo:
            - Improve error handling for unsupported configurations.
        """

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        elif args.input_size is not None and args.frontend is not None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = args.input_size + frontend.output_size()
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

        # 4. Label Aggregator layer
        label_aggregator_class = label_aggregator_choices.get_class(
            args.label_aggregator
        )
        label_aggregator = label_aggregator_class(**args.label_aggregator_conf)

        # 5. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        # Note(jiatong): Diarization may not use subsampling when processing
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 6a. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(
            num_spk=args.num_spk,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # 6b. Attractor
        if getattr(args, "attractor", None) is not None:
            attractor_class = attractor_choices.get_class(args.attractor)
            attractor = attractor_class(
                encoder_output_size=encoder.output_size(),
                **args.attractor_conf,
            )
        else:
            attractor = None

        # 7. Build model
        model = ESPnetDiarizationModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            label_aggregator=label_aggregator,
            encoder=encoder,
            decoder=decoder,
            attractor=attractor,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
