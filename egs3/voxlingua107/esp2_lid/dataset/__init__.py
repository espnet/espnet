"""VoxLingua107 dataset module."""

from .builder import VoxLingua107Builder as DatasetBuilder
from .dataset import VoxLingua107Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
