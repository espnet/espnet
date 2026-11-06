"""Mini AN4 classification dataset module."""

from egs3.mini_an4.esp2_cls.dataset.builder import MiniAn4ClsBuilder as DatasetBuilder
from egs3.mini_an4.esp2_cls.dataset.dataset import MiniAn4ClsDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
