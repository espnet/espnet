"""URGENT 2025 dataset module of the SNOAR recipe.

Training mixes come from the SNOAR system's on-the-fly URGENT mixer
(``espnet3.systems.snoar.src.data.urgent``); this module's builder writes the
noise license lists and the 16 kHz dev set.
"""

from egs3.urgent25.snoar.dataset.builder import Urgent25Builder
from espnet3.systems.snoar.src.data.urgent import UrgentMixDataset

Dataset = UrgentMixDataset
DatasetBuilder = Urgent25Builder
