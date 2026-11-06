"""LibriSpeech 100h dataset module backed by an omniio dump."""

from egs3.librispeech_100.esp2_asr_omniio.dataset.builder import (
    LibriSpeech100OmniIOBuilder as DatasetBuilder,
)
from egs3.librispeech_100.esp2_asr_omniio.dataset.dataset import (
    LibriSpeech100OmniIODataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
