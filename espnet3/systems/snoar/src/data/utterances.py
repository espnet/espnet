"""Whole recordings for the ``infer`` stage (``data_src`` module form)."""

from espnet3.systems.snoar.src.data.noisy import ManifestBuilder, UtteranceDataset

Dataset = UtteranceDataset
DatasetBuilder = ManifestBuilder
