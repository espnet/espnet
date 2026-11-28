"""WHAM! dataset module of the SNOAR recipe.

The recipe reads its manifests with the SNOAR system's own datasets
(``espnet3.systems.snoar.src.data.noisy``); this module builds them.
"""

from egs3.wham.snoar.dataset.builder import WHAMBuilder
from espnet3.systems.snoar.src.data.noisy import NoisyCropDataset

Dataset = NoisyCropDataset
DatasetBuilder = WHAMBuilder
