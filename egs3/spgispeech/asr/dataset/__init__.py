"""SPGISpeech dataset module."""

from egs3.spgispeech.asr.dataset.builder import SPGISpeechBuilder as DatasetBuilder
from egs3.spgispeech.asr.dataset.dataset import (
    SPGISpeechDataset as Dataset,
    gather_training_text,
)

__all__ = ["Dataset", "DatasetBuilder", "gather_training_text"]
