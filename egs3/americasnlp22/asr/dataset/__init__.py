"""AmericasNLP 2022 dataset module."""

from egs3.americasnlp22.asr.dataset.builder import (
    AmericasNLP22Builder as DatasetBuilder,
)
from egs3.americasnlp22.asr.dataset.dataset import AmericasNLP22Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
