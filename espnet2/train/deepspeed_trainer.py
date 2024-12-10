""" DeepSpeed Trainer Module """

import argparse
import dataclasses
import json
import logging

import torch
import torch.distributed as dist

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except ImportError:
    logging.warning("deepspeed is not installed")
    deepspeed = None
    DeepSpeedEngine = None

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.train.trainer import Trainer
from espnet2.utils.build_dataclass import build_dataclass


@dataclasses.dataclass
class DeepSpeedTrainerOptions:
    """
        DeepSpeed Trainer Options for configuring training with DeepSpeed.

    This class holds the options required to set up the DeepSpeed trainer,
    including parameters for resuming training, setting seeds, data types,
    logging intervals, output directories, and maximum epochs.

    Attributes:
        resume (bool): Whether to resume training from a checkpoint.
        seed (int): Seed for random number generation.
        train_dtype (Union[str, torch.dtype]): Data type for training (e.g., 'fp16').
        log_interval (Optional[int]): Interval for logging training progress.
        output_dir (Union[Path, str]): Directory for saving outputs and checkpoints.
        max_epoch (int): Maximum number of epochs for training.
        deepspeed_config (Union[Path, str]): Path to the DeepSpeed configuration file.

    Examples:
        >>> options = DeepSpeedTrainerOptions(
        ...     resume=True,
        ...     seed=42,
        ...     train_dtype='fp16',
        ...     log_interval=100,
        ...     output_dir='./output',
        ...     max_epoch=10,
        ...     deepspeed_config='./deepspeed_config.json'
        ... )
        >>> print(options)
        DeepSpeedTrainerOptions(resume=True, seed=42, train_dtype='fp16',
        ... log_interval=100, output_dir=PosixPath('output'),
        ... max_epoch=10, deepspeed_config=PosixPath('deepspeed_config.json'))
    """

    resume: bool
    seed: int
    train_dtype: Union[str, torch.dtype]
    log_interval: Optional[int]
    output_dir: Union[Path, str]
    max_epoch: int
    deepspeed_config: Union[Path, str]


class DeepSpeedTrainer(Trainer):
    """
        DeepSpeed Trainer Module for training deep learning models using DeepSpeed.

    This class extends the Trainer class and integrates with the DeepSpeed library
    to facilitate efficient training of models. It manages the training loop,
    validation, checkpointing, and resuming from checkpoints.

    Attributes:
        None

    Args:
        model (Union[AbsESPnetModel, DeepSpeedEngine]): The model to be trained.
        train_iter_factory (AbsIterFactory): Factory to create training iterators.
        valid_iter_factory (AbsIterFactory): Factory to create validation iterators.
        trainer_options (DeepSpeedTrainerOptions): Options for the DeepSpeed trainer.
        **kwargs: Additional arguments.

    Returns:
        None

    Yields:
        None

    Raises:
        ImportError: If the DeepSpeed library is not installed.

    Examples:
        # Example usage of DeepSpeedTrainer
        trainer = DeepSpeedTrainer()
        trainer.run(model, train_iter_factory, valid_iter_factory, trainer_options)

    Note:
        Ensure that the DeepSpeed library is installed in your environment.

    Todo:
        - Add support for more training options.
    """

    @classmethod
    @typechecked
    def build_options(cls, args: argparse.Namespace) -> DeepSpeedTrainerOptions:
        """
                Build options for the DeepSpeedTrainer from command-line arguments.

        This method constructs a `DeepSpeedTrainerOptions` instance, which contains
        various configuration settings necessary for training using the DeepSpeed
        library. It utilizes the provided command-line arguments parsed into an
        `argparse.Namespace` object.

        Args:
            cls: The class that calls this method (typically the DeepSpeedTrainer class).
            args (argparse.Namespace): The command-line arguments parsed into a
                Namespace object.

        Returns:
            DeepSpeedTrainerOptions: An instance of DeepSpeedTrainerOptions
                containing the configuration options.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument('--resume', type=bool, default=False)
            >>> parser.add_argument('--seed', type=int, default=42)
            >>> parser.add_argument('--train_dtype', type=str, default='float32')
            >>> parser.add_argument('--log_interval', type=int, default=10)
            >>> parser.add_argument('--output_dir', type=str, default='./output')
            >>> parser.add_argument('--max_epoch', type=int, default=100)
            >>> parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')
            >>> args = parser.parse_args()
            >>> options = DeepSpeedTrainer.build_options(args)
            >>> print(options)
            DeepSpeedTrainerOptions(resume=False, seed=42, train_dtype='float32',
                                    log_interval=10, output_dir=Path('./output'),
                                    max_epoch=100, deepspeed_config=Path('ds_config.json'))
        """
        return build_dataclass(DeepSpeedTrainerOptions, args)

    @staticmethod
    @typechecked
    def resume(
        model: DeepSpeedEngine,
        reporter: Reporter,
        output_dir: Path,
    ):
        """
                DeepSpeed Trainer Module

        This module provides the DeepSpeedTrainer class, which facilitates training
        using the DeepSpeed library. It includes functionality for building options,
        resuming training from checkpoints, and running training and validation
        epochs.

        Attributes:
            resume (bool): Flag to indicate whether to resume training from a checkpoint.
            seed (int): Seed for random number generation.
            train_dtype (Union[str, torch.dtype]): Data type for training (e.g., float32).
            log_interval (Optional[int]): Interval for logging training metrics.
            output_dir (Union[Path, str]): Directory to save output and checkpoints.
            max_epoch (int): Maximum number of epochs for training.
            deepspeed_config (Union[Path, str]): Path to the DeepSpeed configuration file.

        Args:
            model (DeepSpeedEngine): The DeepSpeed model to be trained.
            reporter (Reporter): The reporter instance for logging metrics.
            output_dir (Path): Directory containing checkpoints to resume from.

        Returns:
            None: This method does not return any value.

        Raises:
            ImportError: If the DeepSpeed library is not installed.

        Examples:
            To resume training from the latest checkpoint:
                trainer.resume(model, reporter, output_dir)

            To build options from command-line arguments:
                options = DeepSpeedTrainer.build_options(args)
        """
        ckpts = [
            item
            for item in output_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint_")
        ]

        if len(ckpts) == 0:
            logging.info("Try to resume but find no checkpoint")
            return

        ckpt_num = max([int(item.name.split("_")[-1]) for item in ckpts])
        ckpt_path = output_dir / f"checkpoint_{ckpt_num}"
        logging.info(f"Resume training from {ckpt_path}")

        _, clinet_states = model.load_checkpoint(ckpt_path)

        reporter.load_state_dict(clinet_states["reporter"])

    @classmethod
    @typechecked
    def run(
        cls,
        model: Union[AbsESPnetModel, DeepSpeedEngine],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        trainer_options: DeepSpeedTrainerOptions,
        **kwargs,
    ) -> None:
        """
        Run the training and validation process for the DeepSpeedTrainer.

        This method initializes the DeepSpeed engine, sets up the reporter,
        and orchestrates the training and validation loops for the specified
        number of epochs. It also handles checkpointing and resuming training
        if required.

        Args:
            model (Union[AbsESPnetModel, DeepSpeedEngine]): The model to be trained.
            train_iter_factory (AbsIterFactory): Factory for creating training data
                iterators.
            valid_iter_factory (AbsIterFactory): Factory for creating validation
                data iterators.
            trainer_options (DeepSpeedTrainerOptions): Options containing training
                configurations such as max epochs, seed, etc.
            **kwargs: Additional keyword arguments (not used).

        Raises:
            ImportError: If the DeepSpeed package is not installed.

        Examples:
            >>> from espnet2.train.deepspeed_trainer import DeepSpeedTrainer
            >>> options = DeepSpeedTrainerOptions(
            ...     resume=False,
            ...     seed=42,
            ...     train_dtype='fp16',
            ...     log_interval=100,
            ...     output_dir='output',
            ...     max_epoch=10,
            ...     deepspeed_config='ds_config.json'
            ... )
            >>> trainer = DeepSpeedTrainer()
            >>> trainer.run(model, train_iter_factory, valid_iter_factory, options)
        """

        # (1) arguments needed in previous trainer but not this one. Delete them
        del kwargs

        # (2) initailize deepspeed
        if deepspeed is None:
            raise ImportError("Cannot proceed as deepspeed is not installed")
        deepspeed_config = json.load(open(trainer_options.deepspeed_config))
        trainer_options.train_dtype = cls.setup_data_dtype(deepspeed_config)
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config,
        )

        # (3) setup reporter, output_dir, dataloader etc.
        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()

        # (4) resume
        if trainer_options.resume:
            cls.resume(
                model=model,
                reporter=reporter,
                output_dir=output_dir,
            )

        # (5) loop on epochs
        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            set_all_random_seed(trainer_options.seed + iepoch)
            reporter.set_epoch(iepoch)

            # (5.1) train one epoch
            with reporter.observe("train") as sub_reporter:
                cls.train_one_epoch(
                    model=model,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                )

            # (5.2) valid one epoch
            with reporter.observe("valid") as sub_reporter:
                cls.valid_one_epoch(
                    model=model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                )

            # (5.3) save checkpoint
            checkpoint_path = output_dir / f"checkpoint_{iepoch}"
            model.save_checkpoint(
                checkpoint_path,
                tag=f"{iepoch}",
                client_state={"reporter": reporter.state_dict()},
            )

            # (5.4) reporter
            if dist.get_rank() == 0:
                logging.info(reporter.log_message())
                reporter.matplotlib_plot(output_dir / "images")

    @classmethod
    @typechecked
    def train_one_epoch(
        cls,
        model,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: DeepSpeedTrainerOptions,
    ) -> None:
        """
                Train the model for one epoch using the provided data iterator.

        This method handles the training loop for a single epoch, performing
        forward and backward passes through the model, logging statistics,
        and updating model parameters. It utilizes distributed training
        techniques to ensure synchronization across multiple devices.

        Attributes:
            model (DeepSpeedEngine): The model to be trained.
            iterator (Iterable[Tuple[List[str], Dict[str, torch.Tensor]]]):
                An iterable that provides batches of training data.
            reporter (SubReporter): An object for logging and reporting
                training metrics.
            options (DeepSpeedTrainerOptions): Options that configure the
                training process.

        Args:
            cls: The class reference.
            model: The model to train, expected to be a DeepSpeedEngine instance.
            iterator: An iterable that yields tuples containing utterance IDs
                and batches of data.
            reporter: An instance of SubReporter for logging purposes.
            options: A DeepSpeedTrainerOptions instance containing training
                configuration options.

        Returns:
            None: This method does not return any value.

        Raises:
            AssertionError: If the batch is not a dictionary.

        Examples:
            >>> trainer.train_one_epoch(model, data_iterator, reporter, options)

        Note:
            This method is designed to work in a distributed training setup
            where multiple processes may be running concurrently. It ensures
            that all processes synchronize at certain points to maintain
            consistency in training.

        Todo:
            - Implement additional logging options.
            - Optimize training loop for better performance.
        """
        model.train()
        iterator_stop = torch.tensor(0).cuda()

        log_interval = options.log_interval
        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            with reporter.measure_time("step_time"):
                # (0) ensure all ranks have not finished.
                dist.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

                # (1) forward
                batch["utt_id"] = utt_id
                batch = to_device(batch, "cuda", dtype=options.train_dtype)
                loss, stats, weight = model(**batch)

                # (2) all-reduce statistics and logging on model side
                stats = {k: v for k, v in stats.items() if v is not None}
                stats, weight = recursive_average(stats, weight, True)
                reporter.register(stats, weight)

                # (3) backward and logging on trainer side
                loss = loss / weight * dist.get_world_size()
                model.backward(loss)
                model.step()

                reporter.register(
                    dict(
                        grad_norm=model.get_global_grad_norm(),
                        loss_scale=model.loss_scale(),
                        learning_rate=model.get_lr()[0],
                    )
                )

                reporter.next()
                if iiter % log_interval == 0:
                    logging.info(reporter.log_message(-log_interval))

        else:
            iterator_stop.fill_(1)
            dist.all_reduce(iterator_stop, ReduceOp.SUM)

    @classmethod
    @typechecked
    @torch.no_grad()
    def valid_one_epoch(
        cls,
        model,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: DeepSpeedTrainerOptions,
    ) -> None:
        """
                Validates the model for one epoch.

        This method evaluates the model's performance on the validation dataset for one
        epoch. It computes the loss and statistics while ensuring that all distributed
        ranks are synchronized during the validation process.

        Args:
            model (Union[AbsESPnetModel, DeepSpeedEngine]): The model to be validated.
            iterator (Iterable[Tuple[List[str], Dict[str, torch.Tensor]]]): An iterator
                that provides batches of validation data, where each batch is a tuple
                containing utterance IDs and a dictionary of tensors.
            reporter (SubReporter): An object responsible for reporting metrics and
                statistics during the validation process.
            options (DeepSpeedTrainerOptions): Options for the DeepSpeed trainer,
                including data types and configuration settings.

        Yields:
            None

        Raises:
            None

        Examples:
            >>> from my_package import MyModel, MyDataLoader
            >>> model = MyModel()
            >>> valid_iterator = MyDataLoader()
            >>> reporter = SubReporter()
            >>> options = DeepSpeedTrainerOptions(...)
            >>> DeepSpeedTrainer.valid_one_epoch(model, valid_iterator, reporter, options)

        Note:
            This method is designed to work in a distributed environment where
            synchronization between ranks is necessary. It will stop processing if
            any rank has completed its validation.
        """
        model.eval()
        iterator_stop = torch.tensor(0).cuda()

        for iiter, (utt_id, batch) in enumerate(iterator):
            assert isinstance(batch, dict), type(batch)

            # (0) ensure all ranks have not finished.
            dist.all_reduce(iterator_stop, ReduceOp.SUM)
            if iterator_stop > 0:
                break

            # (1) forward
            batch["utt_id"] = utt_id
            batch = to_device(batch, "cuda", dtype=options.train_dtype)
            loss, stats, weight = model(**batch)

            # (2) all-reduce statistics and logging on model side
            stats = {k: v for k, v in stats.items() if v is not None}
            stats, weight = recursive_average(stats, weight, True)

            reporter.register(stats, weight)
            reporter.next()

        else:
            iterator_stop.fill_(1)
            dist.all_reduce(iterator_stop, ReduceOp.SUM)

    @classmethod
    @typechecked
    def setup_data_dtype(cls, deepspeed_config: Dict):
        """
                Sets up the data type for training based on the DeepSpeed configuration.

        This method determines the appropriate data type (dtype) for training based on
        the provided DeepSpeed configuration. It checks for specific keys in the
        configuration dictionary to decide between `bfloat16`, `float16`, or `float32`.

        Attributes:
            cls: The class method reference.

        Args:
            deepspeed_config (Dict): A dictionary containing DeepSpeed configuration
            options, which can include "bf16", "fp16", or "amp".

        Returns:
            torch.dtype: The data type to be used for training.

        Examples:
            >>> deepspeed_config = {"bf16": True}
            >>> dtype = DeepSpeedTrainer.setup_data_dtype(deepspeed_config)
            >>> print(dtype)
            torch.bfloat16

            >>> deepspeed_config = {"fp16": True}
            >>> dtype = DeepSpeedTrainer.setup_data_dtype(deepspeed_config)
            >>> print(dtype)
            torch.float16

            >>> deepspeed_config = {}
            >>> dtype = DeepSpeedTrainer.setup_data_dtype(deepspeed_config)
            >>> print(dtype)
            torch.float

        Note:
            The method checks for the presence of "bf16", "fp16", and "amp" keys in the
            configuration. The choice of dtype may depend on the capabilities of the
            underlying hardware.
        """
        if "bf16" in deepspeed_config:
            return torch.bfloat16

        elif "fp16" in deepspeed_config:
            return torch.float16

        elif "amp" in deepspeed_config:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16

        else:
            return torch.float
