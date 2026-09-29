"""MuST-C read path."""

from __future__ import annotations

from egs3.owsm_v4.owsm.dataset.dataset import OWSMDataset
from egs3.owsm_v4.owsm.dataset.sub_datasets.must_c.builder import (
    CACHE_SUBDIR,
    SPLITS,
)


class MuSTCDataset(OWSMDataset):
    """OWSM samples for one MuST-C split."""

    CACHE_SUBDIR = CACHE_SUBDIR
    SPLITS = SPLITS
