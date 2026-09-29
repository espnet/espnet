"""SPGISpeech read path."""

from __future__ import annotations

from egs3.owsm_v4.owsm.dataset.dataset import OWSMDataset
from egs3.owsm_v4.owsm.dataset.sub_datasets.spgispeech.builder import (
    CACHE_SUBDIR,
    SPLITS,
)


class SPGISpeechDataset(OWSMDataset):
    """OWSM samples for one SPGISpeech split."""

    CACHE_SUBDIR = CACHE_SUBDIR
    SPLITS = SPLITS
