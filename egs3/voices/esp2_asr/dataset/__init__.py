"""Dataset exports used by the ESPnet3 stage runner."""

from egs3.voices.esp2_asr.dataset.builder import VoicesBuilder as DatasetBuilder
from egs3.voices.esp2_asr.dataset.dataset import VoicesDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
