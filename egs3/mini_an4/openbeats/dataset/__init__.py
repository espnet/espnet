"""Mini AN4 dataset module for BEATs pre-training smoke tests."""

from egs3.mini_an4.asr.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.openbeats.dataset.dataset import MiniAn4OpenBeatsDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
