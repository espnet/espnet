"""TIMIT dataset module."""

from egs3.timit.pr.dataset.builder import TimitBuilder as DatasetBuilder
from egs3.timit.pr.dataset.dataset import TimitDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
