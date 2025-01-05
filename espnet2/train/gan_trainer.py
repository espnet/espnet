# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Trainer module for GAN-based training."""

import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from packaging.version import parse as V
from torch.nn.parallel import DistributedDataParallel as DDP
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler, AbsScheduler
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.types import str2bool

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


@dataclasses.dataclass
class GANTrainerOptions(TrainerOptions):
    """
        Trainer option dataclass for GANTrainer.

    This class encapsulates the options that are specific to the GAN training
    process, inheriting from TrainerOptions. The options defined here include
    parameters that control the behavior of the generator and discriminator during
    training.

    Attributes:
        generator_first (bool): Indicates whether to update the generator first
            during training.
        skip_discriminator_prob (float): The probability of skipping the
            discriminator update step. If this value is greater than 0, the
            discriminator will be updated with this probability.

    Examples:
        To create an instance of GANTrainerOptions with specific settings:

        ```python
        options = GANTrainerOptions(generator_first=True, skip_discriminator_prob=0.1)
        ```

        This instance indicates that the generator should be updated first, and
        the discriminator update will be skipped with a probability of 0.1.
    """

    generator_first: bool
    skip_discriminator_prob: float


class GANTrainer(Trainer):
    """
        Trainer module for GAN-based training.

    This class implements a trainer specifically designed for Generative Adversarial
    Networks (GANs). It is intended to be used with models that inherit from
    espnet.train.abs_gan_espnet_model.AbsGANESPnetModel. The GANTrainer manages the
    training process, including forward and backward passes for both the generator
    and discriminator networks.

    Attributes:
        generator_first (bool): Indicates whether to update the generator first.
        skip_discriminator_prob (float): Probability of skipping the discriminator step.

    Args:
        args (argparse.Namespace): Command line arguments parsed by argparse.

    Returns:
        TrainerOptions: An instance of GANTrainerOptions containing the parsed options.

    Raises:
        NotImplementedError: If certain options (accum_grad > 1 or grad_noise) are used.

    Examples:
        To train a GAN using this trainer, you can create a model instance and call
        the training methods as follows:

        ```python
        trainer = GANTrainer()
        trainer.train_one_epoch(model, iterator, optimizers, schedulers, scaler,
                                reporter, summary_writer, options, distributed_option)
        ```

        You can also validate the model with:

        ```python
        trainer.validate_one_epoch(model, iterator, reporter, options,
                                    distributed_option)
        ```

    Note:
        The GANTrainer requires a model that implements specific interfaces to handle
        the GAN training process correctly.

    Todo:
        - Support for additional options such as accum_grad > 1 and grad_noise in
          GAN-based training.
    """

    @classmethod
    @typechecked
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """
        Build options consumed by train(), eval(), and plot_attention().

        This method constructs a set of options for the GANTrainer based on
        the provided command-line arguments. It creates an instance of
        GANTrainerOptions, which includes parameters specifically for GAN
        training.

        Args:
            args (argparse.Namespace): The command-line arguments parsed.

        Returns:
            TrainerOptions: An instance of GANTrainerOptions populated with
            the specified arguments.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument('--generator_first', type=str2bool, default=False)
            >>> parser.add_argument('--skip_discriminator_prob', type=float, default=0.0)
            >>> args = parser.parse_args()
            >>> options = GANTrainer.build_options(args)
            >>> print(options.generator_first)
            False
            >>> print(options.skip_discriminator_prob)
            0.0
        """
        return build_dataclass(GANTrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """
            Add additional arguments for GAN-trainer.

        This method extends the command-line argument parser with specific options
        for the GANTrainer. It allows the user to specify whether to update the
        generator first and the probability of skipping the discriminator step.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the
                arguments will be added.

        Examples:
            >>> import argparse
            >>> parser = argparse.ArgumentParser()
            >>> GANTrainer.add_arguments(parser)
            >>> args = parser.parse_args(["--generator_first", "True"])
            >>> print(args.generator_first)  # Output: True
            >>> print(args.skip_discriminator_prob)  # Output: 0.0 (default)
        """
        parser.add_argument(
            "--generator_first",
            type=str2bool,
            default=False,
            help="Whether to update generator first.",
        )
        parser.add_argument(
            "--skip_discriminator_prob",
            type=float,
            default=0.0,
            help="If > 0, skip the discriminator step with a probability",
        )

    @classmethod
    @typechecked
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: GANTrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        """
        Train one epoch.

        This method performs a single epoch of training for the GAN model.
        It handles the forward and backward passes for both the generator and
        discriminator, applying optimizations and logging the training
        statistics.

        Args:
            model (torch.nn.Module): The GAN model to be trained.
            iterator (Iterable[Tuple[List[str], Dict[str, torch.Tensor]]]):
                An iterable that provides batches of data for training.
            optimizers (Sequence[torch.optim.Optimizer]): A sequence of
                optimizers for the generator and discriminator.
            schedulers (Sequence[Optional[AbsScheduler]]): A sequence of
                schedulers for adjusting the learning rate.
            scaler (Optional[GradScaler]): A GradScaler for mixed precision
                training.
            reporter (SubReporter): An object for reporting training
                statistics.
            summary_writer: A writer for logging summaries (e.g., TensorBoard).
            options (GANTrainerOptions): The options for the GAN training
                process.
            distributed_option (DistributedOption): Options for distributed
                training.

        Returns:
            bool: True if all steps in the epoch were invalid (i.e., no valid
            updates were made), otherwise False.

        Raises:
            NotImplementedError: If certain options like `accum_grad` or
            `grad_noise` are set to unsupported values.

        Examples:
            >>> options = GANTrainerOptions(generator_first=True,
            ...                              skip_discriminator_prob=0.1)
            >>> result = GANTrainer.train_one_epoch(model, iterator,
            ...                                      optimizers, schedulers,
            ...                                      scaler, reporter,
            ...                                      summary_writer, options,
            ...                                      distributed_option)

        Note:
            This method assumes that the model has a forward method that
            returns a dictionary containing the loss and statistics.

        Todo:
            - Support for `accum_grad > 1` and `grad_noise`.
        """

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        generator_first = options.generator_first
        distributed = distributed_option.distributed
        skip_discriminator_prob = options.skip_discriminator_prob

        # Check unavailable options
        # TODO(kan-bayashi): Support the use of these options
        if accum_grad > 1:
            raise NotImplementedError(
                "accum_grad > 1 is not supported in GAN-based training."
            )
        if grad_noise:
            raise NotImplementedError(
                "grad_noise is not supported in GAN-based training."
            )

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (_, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            turn_start_time = time.perf_counter()
            if generator_first:
                turns = ["generator", "discriminator"]
            else:
                turns = ["discriminator", "generator"]
            for turn in turns:
                # (Jinchuan) skip some updates of discriminator to avoid
                # being over-powered. Synchronized globally.
                if skip_discriminator_prob > 0.0 and turn == "discriminator":
                    if torch.distributed.is_initialized():
                        skip_disc = torch.rand(1)
                        torch.distributed.broadcast(torch.rand(1).cuda(), src=0)
                    else:
                        skip_disc = torch.rand(1)
                    if skip_disc.item() < skip_discriminator_prob:
                        if isinstance(model, DDP):
                            model.module.codec._cache = None
                        elif isinstance(model, torch.nn.Module):
                            model.codec._cache = None
                        else:
                            raise RuntimeError("cannot get model for cache cleaning")
                        continue

                with autocast(scaler is not None):
                    with reporter.measure_time(f"{turn}_forward_time"):
                        retval = model(forward_generator=turn == "generator", **batch)

                        # Note(kamo):
                        # Supporting two patterns for the returned value from the model
                        #   a. dict type
                        if isinstance(retval, dict):
                            loss = retval["loss"]
                            stats = retval["stats"]
                            weight = retval["weight"]
                            optim_idx = retval.get("optim_idx")
                            if optim_idx is not None and not isinstance(optim_idx, int):
                                if not isinstance(optim_idx, torch.Tensor):
                                    raise RuntimeError(
                                        "optim_idx must be int or 1dim torch.Tensor, "
                                        f"but got {type(optim_idx)}"
                                    )
                                if optim_idx.dim() >= 2:
                                    raise RuntimeError(
                                        "optim_idx must be int or 1dim torch.Tensor, "
                                        f"but got {optim_idx.dim()}dim tensor"
                                    )
                                if optim_idx.dim() == 1:
                                    for v in optim_idx:
                                        if v != optim_idx[0]:
                                            raise RuntimeError(
                                                "optim_idx must be 1dim tensor "
                                                "having same values for all entries"
                                            )
                                    optim_idx = optim_idx[0].item()
                                else:
                                    optim_idx = optim_idx.item()

                        # b. tuple or list type
                        else:
                            raise RuntimeError("model output must be dict.")

                    stats = {k: v for k, v in stats.items() if v is not None}
                    if ngpu > 1 or distributed:
                        # Apply weighted averaging for loss and stats
                        loss = (loss * weight.type(loss.dtype)).sum()

                        # if distributed, this method can also apply all_reduce()
                        stats, weight = recursive_average(stats, weight, distributed)

                        # Now weight is summation over all workers
                        loss /= weight

                    if distributed:
                        # NOTE(kamo): Multiply world_size since DistributedDataParallel
                        # automatically normalizes the gradient by world_size.
                        loss *= torch.distributed.get_world_size()

                reporter.register(stats, weight)

                with reporter.measure_time(f"{turn}_backward_time"):
                    if scaler is not None:
                        # Scales loss.  Calls backward() on scaled loss
                        # to create scaled gradients.
                        # Backward passes under autocast are not recommended.
                        # Backward ops run in the same dtype autocast chose
                        # for corresponding forward ops.
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                # TODO(kan-bayashi): Compute grad norm without clipping
                grad_norm = None
                if grad_clip > 0.0:
                    # compute the gradient norm to check if it is normal or not
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                        norm_type=grad_clip_type,
                    )
                    # PyTorch<=1.4, clip_grad_norm_ returns float value
                    if not isinstance(grad_norm, torch.Tensor):
                        grad_norm = torch.tensor(grad_norm)

                if grad_norm is None or torch.isfinite(grad_norm):
                    all_steps_are_invalid = False
                    with reporter.measure_time(f"{turn}_optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                # scaler.step() first unscales the gradients of
                                # the optimizer's assigned params.
                                scaler.step(optimizer)
                                # Updates the scale for next iteration.
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
                else:
                    logging.warning(
                        f"The grad norm is {grad_norm}. " "Skipping updating the model."
                    )
                    # Must invoke scaler.update() if unscale_() is used in the
                    # iteration to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                for iopt, optimizer in enumerate(optimizers):
                    # NOTE(kan-bayashi): In the case of GAN, we need to clear
                    #   the gradient of both optimizers after every update.
                    optimizer.zero_grad()

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    {
                        f"optim{optim_idx}_lr{i}": pg["lr"]
                        for i, pg in enumerate(optimizers[optim_idx].param_groups)
                        if "lr" in pg
                    },
                )
                reporter.register(
                    {f"{turn}_train_time": time.perf_counter() - turn_start_time}
                )
                turn_start_time = time.perf_counter()

            reporter.register({"train_time": time.perf_counter() - start_time})
            start_time = time.perf_counter()

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    @typechecked
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: GANTrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        """Validate one epoch."""
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed
        generator_first = options.generator_first

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for _, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            if generator_first:
                turns = ["generator", "discriminator"]
            else:
                turns = ["discriminator", "generator"]
            for turn in turns:
                retval = model(forward_generator=turn == "generator", **batch)
                if isinstance(retval, dict):
                    stats = retval["stats"]
                    weight = retval["weight"]
                else:
                    _, stats, weight = retval
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for stats.
                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)
                reporter.register(stats, weight)

            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
