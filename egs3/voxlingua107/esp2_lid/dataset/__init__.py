"""VoxLingua107 dataset module."""

from egs3.voxlingua107.esp2_lid.dataset.builder import (
    VoxLingua107Builder as DatasetBuilder,
)
from egs3.voxlingua107.esp2_lid.dataset.dataset import VoxLingua107Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
