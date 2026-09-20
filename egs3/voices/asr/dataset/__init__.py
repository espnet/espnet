"""Dataset exports used by the ESPnet3 stage runner."""

from .builder import VoicesBuilder as DatasetBuilder
from .dataset import VoicesDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
