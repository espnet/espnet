"""Mini AN4 LID dataset module."""

from egs3.mini_an4.esp2_lid.dataset.builder import MiniAn4LIDBuilder as DatasetBuilder
from egs3.mini_an4.esp2_lid.dataset.dataset import MiniAn4LIDDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
