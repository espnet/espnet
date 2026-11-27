"""AudioSet-2M dataset module for BEATs pre-training."""

from egs3.audioset.beats.dataset.builder import AudioSetBuilder as DatasetBuilder
from egs3.audioset.beats.dataset.dataset import AudioSetDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
