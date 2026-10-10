"""MELD dataset module."""

from egs3.meld.cls.dataset.builder import MELDBuilder as DatasetBuilder
from egs3.meld.cls.dataset.dataset import MELDDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
