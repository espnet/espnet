"""VoxCeleb speaker verification dataset module."""

from egs3.voxceleb.spk.dataset.builder import VoxCelebBuilder as DatasetBuilder
from egs3.voxceleb.spk.dataset.dataset import VoxCelebDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
