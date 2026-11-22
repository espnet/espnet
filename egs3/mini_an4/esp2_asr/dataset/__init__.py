"""Mini AN4 dataset module."""

from egs3.mini_an4.esp2_asr.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.esp2_asr.dataset.dataset import MiniAn4Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
