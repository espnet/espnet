"""mini_an4 kNN-VC dataset module."""

from egs3.mini_an4.knnvc.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.knnvc.dataset.dataset import MiniAn4KNNVCDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
