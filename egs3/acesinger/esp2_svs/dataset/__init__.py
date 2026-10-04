"""ACE-Opencpop dataset module."""

from .builder import ACEOpencpopBuilder as DatasetBuilder
from .dataset import ACEOpencpopDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
