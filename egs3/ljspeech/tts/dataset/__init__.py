"""LJSpeech TTS dataset module."""

from egs3.ljspeech.tts.dataset.builder import LJSpeechTTSBuilder as DatasetBuilder
from egs3.ljspeech.tts.dataset.dataset import LJSpeechTTSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
