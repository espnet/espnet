"""AudioSet-2M dataset module for BEATs pre-training."""

from egs3.audioset.ssl.dataset.builder import AudioSetBuilder as DatasetBuilder
from egs3.audioset.ssl.dataset.dataset import AudioSetDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
