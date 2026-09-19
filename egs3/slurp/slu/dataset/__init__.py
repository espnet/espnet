"""SLURP dataset module resolved by ``BaseSystem.create_dataset``."""

from egs3.slurp.slu.dataset.builder import SlurpBuilder as DatasetBuilder
from egs3.slurp.slu.dataset.dataset import SlurpDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
