"""Mini AN4 codec dataset module: the ASR recipe's builder, waveform-only items."""

from egs3.mini_an4.esp2_asr.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.esp2_codec.dataset.dataset import MiniAn4CodecDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
