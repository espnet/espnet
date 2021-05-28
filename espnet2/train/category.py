from enum import auto
from enum import IntEnum


class UttCategory(IntEnum):
    """List of possible categories of each utterance for mutli-condition training."""

    # single-speaker clean speech
    CLEAN_1SPEAKER = auto()
    # single-speaker real speech (noisy)
    REAL_1SPEAKER = auto()
    # simulated speech (for training enh)
    SIMU_DATA = auto()
