"""Mini AN4 SLU dataset module."""

from egs3.mini_an4.esp2_slu.dataset.builder import MiniAn4SluBuilder as DatasetBuilder
from egs3.mini_an4.esp2_slu.dataset.dataset import MiniAn4SluDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
