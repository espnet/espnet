"""Speaker training entrypoints for ESPnet3."""

from __future__ import annotations

from typing import Sequence

from espnet3.systems.spk.task import SpeakerTask


def get_train_parser():
    """Return the speaker-training CLI parser."""
    return SpeakerTask.get_parser()


def train(cmd: Sequence[str] | None = None):
    """Run speaker model training through SpeakerTask."""
    return SpeakerTask.main(cmd=cmd)


def main_train(cmd: Sequence[str] | None = None):
    """CLI-compatible alias for speaker training."""
    return train(cmd=cmd)
