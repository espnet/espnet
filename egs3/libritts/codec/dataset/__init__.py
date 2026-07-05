"""LibriTTS codec dataset module."""

from .builder import LibriTTSCodecBuilder as DatasetBuilder
from .dataset import LibriTTSCodecDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
