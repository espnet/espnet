"""LibriSpeech 100h voice-conversion dataset module."""

from egs3.librispeech_100.vc.dataset.builder import (
    LibriSpeech100Builder as DatasetBuilder,
)
from egs3.librispeech_100.vc.dataset.dataset import LibriSpeech100Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
