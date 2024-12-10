# Copyright 2024 Jiatong Shi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based neural codec task."""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.dac.dac import DAC
from espnet2.gan_codec.encodec.encodec import Encodec
from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
from espnet2.gan_codec.funcodec.funcodec import FunCodec
from espnet2.gan_codec.soundstream.soundstream import SoundStream
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.gan_trainer import GANTrainer
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

codec_choices = ClassChoices(
    "codec",
    classes=dict(
        soundstream=SoundStream,
        encodec=Encodec,
        dac=DAC,
        funcodec=FunCodec,
    ),
    default="soundstream",
)


class GANCodecTask(AbsTask):
    """
        GANCodecTask is a class for implementing a GAN-based neural codec task.

    This class extends the AbsTask class and provides functionalities for
    training and building a GAN-based codec model. It supports configuration
    of various codec types and their respective parameters. The task utilizes
    the GANTrainer for optimization and training processes.

    Attributes:
        num_optimizers (int): The number of optimizers required for GAN training.
        class_choices_list (list): A list of codec class choices available for
            this task.
        trainer (type): The trainer class used for this task, which is GANTrainer.

    Args:
        parser (argparse.ArgumentParser): The argument parser instance used to
            define the command-line arguments for the task.

    Returns:
        Callable: A function that processes input data for training or inference.

    Yields:
        None

    Raises:
        ValueError: If the specified optimizer is not valid.

    Examples:
        To use this class, one might create an argument parser and add task
        arguments as follows:

        ```python
        import argparse
        from espnet2.tasks.gan_codec import GANCodecTask

        parser = argparse.ArgumentParser(description="GAN Codec Task")
        GANCodecTask.add_task_arguments(parser)
        args = parser.parse_args()
        ```

        After setting up the arguments, you can build a model:

        ```python
        model = GANCodecTask.build_model(args)
        ```

        To create optimizers for training:

        ```python
        optimizers = GANCodecTask.build_optimizers(args, model)
        ```

    Note:
        This class requires the presence of specific codec implementations
        such as SoundStream, Encodec, DAC, and FunCodec.

    Todo:
        - Add more error handling and logging for better debugging.
    """

    # GAN requires two optimizers
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --codec and --codec_conf
        codec_choices,
    ]

    # Use GANTrainer instead of Trainer
    trainer = GANTrainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """
            Adds command-line arguments specific to the GANCodecTask.

        This method defines the arguments related to the task and preprocessing
        settings. It also allows for adding codec-specific arguments through the
        class choices defined in `class_choices_list`.

        Args:
            parser (argparse.ArgumentParser): The argument parser instance to which
                the task-specific arguments will be added.

        Examples:
            To use this method, you can initialize an argument parser and call
            the `add_task_arguments` method:

            ```python
            import argparse
            from gan_codec_task import GANCodecTask

            parser = argparse.ArgumentParser()
            GANCodecTask.add_task_arguments(parser)
            args = parser.parse_args()
            ```

        Note:
            The `--print_config` mode cannot be used with `required=True` in the
            `add_arguments` method, so this has been handled appropriately.

        Raises:
            argparse.ArgumentError: If there is an issue with adding the arguments
                to the parser.
        """
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetGANCodecModel),
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
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        """
            Build a collate function for the GANCodecTask.

        This method creates a callable function that collates a batch of data
        samples. The collate function is used to process a collection of
        tuples, where each tuple consists of a string identifier and a
        dictionary of numpy arrays. The collate function returns a tuple
        containing a list of string identifiers and a dictionary of
        PyTorch tensors.

        Args:
            args (argparse.Namespace): Command line arguments containing
                configuration options for the task.
            train (bool): A flag indicating whether the function is being
                called during training or evaluation.

        Returns:
            Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]],
                      Tuple[List[str], Dict[str, torch.Tensor]]]]:
                A function that collates data into a format suitable for
                model input.

        Examples:
            >>> args = argparse.Namespace()
            >>> args.some_arg = 'value'
            >>> collate_fn = GANCodecTask.build_collate_fn(args, train=True)
            >>> batch = [('sample1', {'data': np.array([1, 2, 3])}),
            ...          ('sample2', {'data': np.array([4, 5, 6])})]
            >>> identifiers, tensors = collate_fn(batch)
            >>> print(identifiers)  # Output: ['sample1', 'sample2']
            >>> print(tensors)      # Output: {'data': tensor([[1, 2, 3], [4, 5, 6]])}

        Note:
            This method relies on the CommonCollateFn class for its
            implementation, which handles the specifics of padding and
            converting numpy arrays to PyTorch tensors.
        """
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=0,
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """
            Builds a preprocessing function based on the task arguments.

        This method checks if preprocessing is enabled in the arguments and
        constructs a `CommonPreprocessor` if it is. If not, it returns None.

        Args:
            cls: The class type of the calling object.
            args (argparse.Namespace): The arguments namespace containing task
                configuration options.
            train (bool): A flag indicating whether the function is being built
                for training or not.

        Returns:
            Optional[Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]]:
                A preprocessing function that takes a string and a dictionary
                of numpy arrays, and returns a dictionary of numpy arrays, or
                None if preprocessing is not enabled.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(use_preprocessor=True, iterator_type='chunk',
            ... chunk_length=16000)
            >>> preprocess_fn = GANCodecTask.build_preprocess_fn(args, train=True)
            >>> audio_data = {"audio": np.random.rand(16000)}
            >>> processed_data = preprocess_fn("audio", audio_data)

        Note:
            The preprocessing function is designed to work with single-channel
            audio data only.

        Todo:
            - Consider adding more preprocessing options based on future
              requirements.
        """
        if args.use_preprocessor:
            # additional check for chunk iterator, to use short utterance in training
            if args.iterator_type == "chunk":
                min_sample_size = args.chunk_length
            else:
                min_sample_size = -1

            retval = CommonPreprocessor(
                train=train,
                token_type=None,  # disable the text process
                speech_name="audio",
                min_sample_size=min_sample_size,
                audio_pad_value=0.0,
                force_single_channel=True,  # NOTE(jiatong): single channel only now
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
        Returns the required data names for the GAN codec task.

        This method provides the necessary data names based on the mode of
        operation, which can be either training or inference. In this
        implementation, the required data name is "audio" for both modes.

        Args:
            train (bool): Indicates if the data is for training. Defaults to
                True.
            inference (bool): Indicates if the data is for inference.
                Defaults to False.

        Returns:
            Tuple[str, ...]: A tuple containing the required data names.
            In this case, it returns ("audio",) for both training and
            inference modes.

        Examples:
            >>> GANCodecTask.required_data_names(train=True, inference=False)
            ('audio',)

            >>> GANCodecTask.required_data_names(train=False, inference=True)
            ('audio',)
        """
        if not inference:
            retval = ("audio",)
        else:
            # Inference mode
            retval = ("audio",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """
            Returns the optional data names used in the GAN codec task.

        This method can be overridden by subclasses to specify any optional data
        names that the task may utilize. By default, it returns an empty tuple,
        indicating that no optional data names are defined.

        Args:
            train (bool): Indicates whether the task is in training mode. Default is
                True.
            inference (bool): Indicates whether the task is in inference mode.
                Default is False.

        Returns:
            Tuple[str, ...]: A tuple of optional data names used in the task.

        Examples:
            >>> optional_data = GANCodecTask.optional_data_names()
            >>> print(optional_data)
            ()  # This will output an empty tuple by default.

        Note:
            This method is a class method and can be called directly on the
            class without creating an instance.
        """
        return ()

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetGANCodecModel:
        """
        Builds and returns an ESPnetGANCodecModel instance.

        This method constructs the model by first selecting the appropriate
        codec class based on the `codec` argument and then initializing
        an `ESPnetGANCodecModel` instance using the selected codec and
        additional configuration provided in `model_conf`.

        Args:
            args (argparse.Namespace): The parsed command line arguments,
                containing model configuration and codec information.

        Returns:
            ESPnetGANCodecModel: An instance of the ESPnetGANCodecModel
                initialized with the specified codec and model configurations.

        Raises:
            ValueError: If the codec class cannot be found based on the
                provided `args.codec`.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument("--codec", type=str, default="soundstream")
            >>> parser.add_argument("--model_conf", type=dict, default={})
            >>> args = parser.parse_args()
            >>> model = GANCodecTask.build_model(args)
            >>> print(type(model))
            <class 'espnet2.gan_codec.espnet_model.ESPnetGANCodecModel'>
        """

        # 1. Codec
        codec_class = codec_choices.get_class(args.codec)
        codec = codec_class(**args.codec_conf)

        # 2. Build model
        model = ESPnetGANCodecModel(
            codec=codec,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetGANCodecModel,
    ) -> List[torch.optim.Optimizer]:
        """
            Builds optimizers for the generator and discriminator of the GAN model.

        This method initializes two optimizers: one for the generator and one for
        the discriminator, based on the specified optimization algorithms. It checks
        if the model has the required components and raises appropriate errors if
        any of the optimizers are not found.

        Args:
            args (argparse.Namespace): The command-line arguments containing
                optimizer configurations and flags.
            model (ESPnetGANCodecModel): The GAN codec model which contains
                generator and discriminator components.

        Returns:
            List[torch.optim.Optimizer]: A list containing the optimizers for
            both the generator and discriminator.

        Raises:
            ValueError: If the specified optimizer class for the generator or
            discriminator is not valid.
            RuntimeError: If the fairscale library is required but not installed.

        Examples:
            >>> from argparse import Namespace
            >>> args = Namespace(optim='Adam', optim_conf={'lr': 0.001},
            ...                  optim2='SGD', optim2_conf={'lr': 0.01},
            ...                  sharded_ddp=False)
            >>> model = ESPnetGANCodecModel(...)  # Assuming model is created properly
            >>> optimizers = GANCodecTask.build_optimizers(args, model)
            >>> len(optimizers)  # Should return 2
            2
        """
        # check
        assert hasattr(model.codec, "generator")
        assert hasattr(model.codec, "discriminator")

        # define generator optimizer
        optim_g_class = optim_classes.get(args.optim)
        if optim_g_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_g = fairscale.optim.oss.OSS(
                params=model.codec.generator.parameters(),
                optim=optim_g_class,
                **args.optim_conf,
            )
        else:
            optim_g = optim_g_class(
                model.codec.generator.parameters(),
                **args.optim_conf,
            )
        optimizers = [optim_g]

        # define discriminator optimizer
        optim_d_class = optim_classes.get(args.optim2)
        if optim_d_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_d = fairscale.optim.oss.OSS(
                params=model.codec.discriminator.parameters(),
                optim=optim_d_class,
                **args.optim2_conf,
            )
        else:
            optim_d = optim_d_class(
                model.codec.discriminator.parameters(),
                **args.optim2_conf,
            )
        optimizers += [optim_d]

        return optimizers
