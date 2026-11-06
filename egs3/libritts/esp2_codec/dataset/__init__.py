"""LibriTTS codec dataset module."""

from egs3.libritts.esp2_codec.dataset.builder import LibriTTSBuilder as DatasetBuilder
from egs3.libritts.esp2_codec.dataset.dataset import LibriTTSCodecDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
