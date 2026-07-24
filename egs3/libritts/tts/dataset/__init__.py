"""LibriTTS dataset module."""

from .builder import LibriTTSBuilder as DatasetBuilder
from .dataset import LibriTTSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
