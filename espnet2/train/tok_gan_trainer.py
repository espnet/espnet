"""Trainer utilities for adversarial speech-tokenizer learning."""

import argparse
import dataclasses
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsScheduler
from espnet2.tok.tokenizer import SpeechTokenizer
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.gan_trainer import GANTrainer, GANTrainerOptions
from espnet2.train.reporter import SubReporter
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.types import str2bool


@dataclasses.dataclass
class SpeechTokenizerGANTrainerOptions(GANTrainerOptions):
    """Options specific to speech-tokenizer phase transitions."""

    reset_optimizer_on_unfreeze: bool


class SpeechTokenizerGANTrainer(GANTrainer):
    """Alternate main and discriminator updates for speech tokenizers.

    This trainer is independent of the existing codec-oriented GAN trainer.  It
    will reuse cached tokenizer and generator outputs through speech-tokenizer
    model interfaces without assuming that the model owns a ``codec`` member.
    """

    @classmethod
    @typechecked
    def build_options(
        cls, args: argparse.Namespace
    ) -> SpeechTokenizerGANTrainerOptions:
        """Build trainer options including the unfreeze reset policy."""
        return build_dataclass(SpeechTokenizerGANTrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add GAN and speech-tokenizer-specific trainer arguments."""
        super().add_arguments(parser)
        parser.set_defaults(generator_first=True)
        parser.add_argument(
            "--reset_optimizer_on_unfreeze",
            type=str2bool,
            default=False,
            help="Clear all optimizer states when the tokenizer unfreezes.",
        )

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        """Return the underlying model when distributed wrapping is used."""
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    @classmethod
    def _get_tokenizer(cls, model: torch.nn.Module) -> SpeechTokenizer:
        """Locate the shared tokenizer without assuming a codec hierarchy."""
        model = cls._unwrap_model(model)
        if isinstance(model, SpeechTokenizer):
            return model
        tokenizer = getattr(model, "tokenizer", None)
        if not isinstance(tokenizer, SpeechTokenizer):
            raise TypeError(
                "The speech-tokenizer model must expose a SpeechTokenizer as "
                "model.tokenizer"
            )
        return tokenizer

    @classmethod
    def prepare_epoch(
        cls,
        model: torch.nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
        epoch: int,
        reset_optimizer_on_unfreeze: bool = False,
    ) -> bool:
        """Apply tokenizer phase changes before an epoch starts.

        Args:
            model: Speech-tokenizer model, optionally DDP-wrapped.
            optimizers: Optimizers whose first element is the main optimizer.
            epoch: One-based epoch about to be trained.
            reset_optimizer_on_unfreeze: Clear all optimizer states when the
                tokenizer changes from frozen pretraining to fine-tuning.

        Returns:
            Whether the tokenizer crossed its unfreeze boundary.
        """
        if len(optimizers) == 0:
            raise ValueError("At least one main optimizer is required")
        tokenizer = cls._get_tokenizer(model)
        just_unfroze = tokenizer.set_epoch(epoch)
        if just_unfroze and reset_optimizer_on_unfreeze:
            for optimizer in optimizers:
                optimizer.state.clear()
        return just_unfroze

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
        options: SpeechTokenizerGANTrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        """Apply epoch phase changes, then run standard alternating GAN turns."""
        if not options.generator_first:
            raise ValueError(
                "SpeechTokenizerGANTrainer requires generator_first=true so the "
                "discriminator can reuse generator outputs"
            )
        if options.skip_discriminator_prob != 0.0:
            raise ValueError(
                "skip_discriminator_prob is not yet supported by the "
                "speech-tokenizer cache lifecycle"
            )
        cls.prepare_epoch(
            model=model,
            optimizers=optimizers,
            epoch=reporter.epoch,
            reset_optimizer_on_unfreeze=options.reset_optimizer_on_unfreeze,
        )
        return super().train_one_epoch(
            model=model,
            iterator=iterator,
            optimizers=optimizers,
            schedulers=schedulers,
            scaler=scaler,
            reporter=reporter,
            summary_writer=summary_writer,
            options=options,
            distributed_option=distributed_option,
        )
