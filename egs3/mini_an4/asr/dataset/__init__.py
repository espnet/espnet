"""Mini AN4 dataset module."""

from egs3.mini_an4.asr.dataset.builder import MiniAN4Builder as DatasetBuilder
from egs3.mini_an4.asr.dataset.dataset import MiniAN4Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
