"""ESC-50 dataset module."""

from egs3.esc50.cls.dataset.builder import ESC50Builder as DatasetBuilder
from egs3.esc50.cls.dataset.dataset import ESC50Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
