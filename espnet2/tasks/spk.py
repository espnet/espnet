import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.asteroid_frontend import AsteroidFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.melspec_torch import MelSpectrogramTorch
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.spk.encoder.conformer_encoder import MfaConformerEncoder
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.spk.encoder.identity_encoder import IdentityEncoder
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.spk.encoder.ska_tdnn_encoder import SkaTdnnEncoder
from espnet2.spk.encoder.xvector_encoder import XvectorEncoder
from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.spk.loss.aamsoftmax import AAMSoftmax
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.pooling.mean_pooling import MeanPooling
from espnet2.spk.pooling.stat_pooling import StatsPooling
from espnet2.spk.projector.abs_projector import AbsProjector
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector
from espnet2.spk.projector.ska_tdnn_projector import SkaTdnnProjector
from espnet2.spk.projector.xvector_projector import XvectorProjector
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    SpkPreprocessor,
)
from espnet2.train.spk_trainer import SpkTrainer as Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

# Check and understand
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        asteroid_frontend=AsteroidFrontend,
        default=DefaultFrontend,
        fused=FusedFrontends,
        melspec_torch=MelSpectrogramTorch,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
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

normalize_choices = ClassChoices(
    name="normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(
        ecapa_tdnn=EcapaTdnnEncoder,
        identity=IdentityEncoder,
        mfaconformer=MfaConformerEncoder,
        rawnet3=RawNet3Encoder,
        ska_tdnn=SkaTdnnEncoder,
        xvector=XvectorEncoder,
    ),
    type_check=AbsEncoder,
    default="rawnet3",
)

pooling_choices = ClassChoices(
    name="pooling",
    classes=dict(
        chn_attn_stat=ChnAttnStatPooling,
        mean=MeanPooling,
        stats=StatsPooling,
    ),
    type_check=AbsPooling,
    default="chn_attn_stat",
)

projector_choices = ClassChoices(
    name="projector",
    classes=dict(
        rawnet3=RawNet3Projector,
        ska_tdnn=SkaTdnnProjector,
        xvector=XvectorProjector,
    ),
    type_check=AbsProjector,
    default="rawnet3",
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        common=CommonPreprocessor,
        spk=SpkPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="spk",
)

loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        aamsoftmax=AAMSoftmax,
        aamsoftmax_sc_topk=ArcMarginProduct_intertopk_subcenter,
    ),
    default="aamsoftmax",
)


class SpeakerTask(AbsTask):
    """
        SpeakerTask is a class that defines a speaker recognition task in the ESPnet2
    framework. It inherits from AbsTask and encapsulates methods for managing
    training and inference, handling data processing, and building the model
    architecture.

    Attributes:
        num_optimizers (int): The number of optimizers used in the task.
        class_choices_list (list): A list of ClassChoices instances for various
            components such as frontend, specaug, normalize, encoder, pooling,
            projector, preprocessor, and loss.
        trainer (Trainer): The trainer class associated with this task.

    Methods:
        add_task_arguments(parser: argparse.ArgumentParser):
            Adds task-specific arguments to the provided argument parser.

        build_collate_fn(args: argparse.Namespace, train: bool) -> Callable:
            Constructs a collate function based on the provided arguments.

        build_preprocess_fn(args: argparse.Namespace, train: bool) -> Optional[Callable]:
            Constructs a preprocessing function based on the provided arguments.

        required_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns a tuple of required data names based on the training or inference mode.

        optional_data_names(train: bool = True, inference: bool = False) -> Tuple[str, ...]:
            Returns a tuple of optional data names based on the training or inference mode.

        build_model(args: argparse.Namespace) -> ESPnetSpeakerModel:
            Constructs and returns an ESPnetSpeakerModel based on the provided arguments.

    Examples:
        To add task arguments to a parser:
            import argparse
            parser = argparse.ArgumentParser()
            SpeakerTask.add_task_arguments(parser)

        To build a model:
            args = parser.parse_args()
            model = SpeakerTask.build_model(args)

    Note:
        This class requires the ESPnet2 library and is designed for speaker
        recognition tasks.

    Todo:
        - Consider adding support for additional loss functions and preprocessing
          methods in the future.
    """

    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        encoder_choices,
        pooling_choices,
        projector_choices,
        preprocessor_choices,
        loss_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
                Adds task-related arguments to the provided argument parser.

        This method is used to define various command-line arguments that are
        specific to the speaker task. The arguments include options for
        initialization methods, preprocessing settings, input size, target
        duration, and speaker-related configurations.

        Args:
            parser (argparse.ArgumentParser): The argument parser instance to
                which the task arguments will be added.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> SpeakerTask.add_task_arguments(parser)
            >>> args = parser.parse_args()

        Note:
            The method dynamically adds arguments based on the choices defined
            in the class-level `class_choices_list`.
        """
        group = parser.add_argument_group(description="Task related")

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
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--target_duration",
            type=float,
            default=3.0,
            help="Duration (in seconds) of samples in a minibatch",
        )

        group.add_argument(
            "--spk2utt",
            type=str,
            default="",
            help="Directory of spk2utt file to be used in label mapping",
        )

        group.add_argument(
            "--spk_num",
            type=int,
            default=None,
            help="specify the number of speakers during training",
        )

        group.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            help="Sampling rate",
        )

        group.add_argument(
            "--num_eval",
            type=int,
            default=10,
            help="Number of segments to make from one utterance in the "
            "inference phase",
        )

        group.add_argument(
            "--rir_scp",
            type=str,
            default="",
            help="Directory of the rir data to be augmented",
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetSpeakerModel),
            help="The keyword arguments for model class.",
        )

        for class_choices in cls.class_choices_list:
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        """
            Builds a collate function for processing batches of data during training or
        inference.

        This method constructs a collate function that will be used by a data loader
        to collate data samples into a batch. The collate function is responsible for
        ensuring that all samples in the batch are appropriately padded and organized
        for model input.

        Args:
            args (argparse.Namespace): The command-line arguments containing various
                configurations for the task.
            train (bool): A flag indicating whether the function is being called in
                training mode. This can affect the way data is collated.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                     Tuple[List[str], Dict[str, torch.Tensor]]]:
                A callable that takes a collection of data samples and returns a
                tuple containing a list of keys and a dictionary of batched tensors.

        Examples:
            >>> from espnet2.tasks.speaker_task import SpeakerTask
            >>> args = argparse.Namespace(use_preprocessor=True, ...)
            >>> collate_fn = SpeakerTask.build_collate_fn(args, train=True)
            >>> batch = collate_fn(data_samples)

        Note:
            The function utilizes CommonCollateFn to handle the actual collating
            logic, which can be customized based on specific requirements.
        """
        return CommonCollateFn()

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
        Builds a preprocessing function based on the provided arguments.

        This function constructs a callable that processes input data
        according to the specified preprocessing class, which can be
        configured for training or inference.

        Args:
            args (argparse.Namespace): Command-line arguments containing
                preprocessing configuration.
            train (bool): A flag indicating whether the function is
                being built for training or inference.

        Returns:
            Optional[Callable[[str, Dict[str, np.array]],
            Dict[str, np.ndarray]]]: A preprocessing function that takes
            a file path and a dictionary of features, returning a
            processed dictionary of features. If preprocessing is not
            enabled, returns None.

        Examples:
            >>> args = argparse.Namespace(use_preprocessor=True,
            ...                            preprocessor='spk',
            ...                            spk2utt='path/to/spk2utt',
            ...                            preprocessor_conf={})
            >>> preprocess_fn = SpeakerTask.build_preprocess_fn(args, train=True)
            >>> processed_data = preprocess_fn('path/to/audio.wav',
            ...                                  {'feature': np.array([...])})

        Note:
            The actual preprocessing class is selected based on the
            `args.preprocessor` attribute. Make sure the specified
            preprocessor is available in the choices.

        Todo:
            Implement additional preprocessing options as required.
        """
        if args.use_preprocessor:
            if train:
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    spk2utt=args.spk2utt,
                    train=train,
                    **args.preprocessor_conf,
                )
            else:
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    train=train,
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

        This method provides the necessary data names based on the task mode.
        For training, it typically includes the speech data and speaker labels,
        while for inference, it usually only includes the speech data.

        Args:
            train (bool, optional): A flag indicating whether the mode is training.
                Defaults to True.
            inference (bool, optional): A flag indicating whether the mode is
                inference. Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the required data names.

        Examples:
            >>> SpeakerTask.required_data_names(train=True)
            ('speech', 'spk_labels')

            >>> SpeakerTask.required_data_names(train=False)
            ('speech',)
        """
        if train:
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
            A task for speaker recognition and related operations.

        This class extends the `AbsTask` class and provides methods to handle
        various aspects of speaker recognition tasks, including model building,
        preprocessing, and argument parsing.

        Attributes:
            num_optimizers (int): The number of optimizers to use.
            class_choices_list (List[ClassChoices]): List of available class choices
                for various components such as frontend, encoder, and loss.
            trainer (Type[Trainer]): The trainer class for the task.

        Args:
            parser (argparse.ArgumentParser): The argument parser instance to add
                task-related arguments.

        Returns:
            None

        Yields:
            None

        Raises:
            None

        Examples:
            To add task-related arguments to a parser:

            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> SpeakerTask.add_task_arguments(parser)

            To build a model based on provided arguments:

            >>> args = parser.parse_args()
            >>> model = SpeakerTask.build_model(args)

        Note:
            This task supports optional components for frontend, normalization,
            encoding, pooling, projection, and loss functions.

        Todo:
            Implement model configuration loading from `model_conf` when it exists.
        """
        # When calculating EER, we need trials where each trial has two
        # utterances. speech2 corresponds to the second utterance of each
        # trial pair in the validation/inference phase.
        retval = ("speech2", "trial", "spk_labels", "task_tokens")

        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSpeakerModel:
        """
                Builds the speaker model based on the provided arguments.

        This method constructs an `ESPnetSpeakerModel` by utilizing various components
        such as frontend, spec augmentation, normalization, encoder, pooling, projector,
        and loss functions. The components are instantiated based on the specified
        arguments, and their configurations.

        Args:
            args (argparse.Namespace): The arguments containing configurations for
                model components including frontend, specaug, normalize, encoder,
                pooling, projector, and loss.

        Returns:
            ESPnetSpeakerModel: An instance of the constructed speaker model.

        Examples:
            >>> args = argparse.Namespace()
            >>> args.frontend = "default"
            >>> args.specaug = "specaug"
            >>> args.normalize = "global_mvn"
            >>> args.encoder = "ecapa_tdnn"
            >>> args.pooling = "mean"
            >>> args.projector = "rawnet3"
            >>> args.loss = "aamsoftmax"
            >>> args.spk_num = 10
            >>> model = SpeakerTask.build_model(args)

        Note:
            This method assumes that all necessary configurations are provided in the
            `args` namespace. It will raise an error if required arguments are missing.

        Todo:
            - Implement handling for model_conf when available.
        """

        if args.frontend is not None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()

        else:
            # Give features from data-loader (e.g., precompute features).
            frontend = None
            input_size = args.input_size

        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size()

        pooling_class = pooling_choices.get_class(args.pooling)
        pooling = pooling_class(input_size=encoder_output_size, **args.pooling_conf)
        pooling_output_size = pooling.output_size()

        projector_class = projector_choices.get_class(args.projector)
        projector = projector_class(
            input_size=pooling_output_size, **args.projector_conf
        )
        projector_output_size = projector.output_size()

        loss_class = loss_choices.get_class(args.loss)
        loss = loss_class(
            nout=projector_output_size, nclasses=args.spk_num, **args.loss_conf
        )

        model = ESPnetSpeakerModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            pooling=pooling,
            projector=projector,
            loss=loss,
            # **args.model_conf, # uncomment when model_conf exists
        )

        if args.init is not None:
            initialize(model, args.init)

        return model
