# espnet3/systems/enh/scheduler.py
"""Schedulers for enhancement systems."""
from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau


class ReduceLROnPlateau(_ReduceLROnPlateau):
    """ReduceLROnPlateau that ignores warmup_steps from TEMPLATE base config merge."""

    def __init__(self, optimizer, mode="min", factor=0.5,
                 patience=3, warmup_steps=None, **kwargs):
        super().__init__(optimizer, mode=mode, factor=factor,
                         patience=patience, **kwargs)