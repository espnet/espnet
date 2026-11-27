"""AmericasNLP 2022 dataset module."""

from egs3.americasnlp22.esp2_asr.dataset.builder import (
    AmericasNLP22Builder as DatasetBuilder,
)
from egs3.americasnlp22.esp2_asr.dataset.dataset import AmericasNLP22Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
