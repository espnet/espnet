"""SPGISpeech dataset module."""

from egs3.spgispeech.esp2_asr.dataset.builder import SPGISpeechBuilder as DatasetBuilder
from egs3.spgispeech.esp2_asr.dataset.dataset import SPGISpeechDataset as Dataset
from egs3.spgispeech.esp2_asr.dataset.dataset import gather_training_text

__all__ = ["Dataset", "DatasetBuilder", "gather_training_text"]
