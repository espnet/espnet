"""Mini AN4 GAN-TTS dataset module: speaker-labelled manifests, TTS items."""

from egs3.mini_an4.esp2_gan_tts.dataset.builder import (
    MiniAn4GANTTSBuilder as DatasetBuilder,
)
from egs3.mini_an4.esp2_gan_tts.dataset.dataset import MiniAn4GANTTSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
