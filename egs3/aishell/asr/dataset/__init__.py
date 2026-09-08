"""AISHELL-1 dataset module."""

from egs3.aishell.asr.dataset.builder import (
    AishellBuilder as DatasetBuilder,
)
from egs3.aishell.asr.dataset.dataset import AishellDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
