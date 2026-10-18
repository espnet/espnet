"""LibriCSS dataset module."""

from .builder import LibriCSSBuilder as DatasetBuilder
from .dataset import LibriCSSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
