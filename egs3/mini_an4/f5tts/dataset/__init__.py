"""Mini AN4 F5-TTS dataset module: the ASR recipe's builder, TTS items."""

from egs3.mini_an4.esp2_asr.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.f5tts.dataset.dataset import MiniAn4F5TTSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
