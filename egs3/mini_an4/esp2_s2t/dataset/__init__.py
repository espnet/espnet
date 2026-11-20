"""Dataset package for the mini AN4 serialized-output-training recipe."""

from egs3.mini_an4.esp2_s2t.dataset.builder import MiniAn4SotBuilder as DatasetBuilder
from egs3.mini_an4.esp2_s2t.dataset.dataset import MiniAn4SotDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
