"""Mini AN4 TTS dataset module."""

from egs3.mini_an4.tts.dataset.builder import MiniAn4TTSBuilder as DatasetBuilder
from egs3.mini_an4.tts.dataset.dataset import MiniAn4TTSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
