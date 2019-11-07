from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, Tuple, Optional

import numpy as np
import torch
from pytypes import typechecked


class Reporter:
    @typechecked
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def get_epoch(self) -> int:
        return self.epoch

    def get_value(self, key: str, key2: str):
        return np.nanmean(self.stats[self.epoch][key][key2])

    def best_epoch_and_value(self, key: str, key2: str, mode: str) \
            -> Tuple[Optional[int], Optional[float]]:
        assert mode in ('min', 'max'), mode

        # iterate from the last epoch
        best_value = None
        best_epoch = None
        for epoch in sorted(self.stats):
            values = self.stats[epoch][key][key2]
            value = np.nanmean(values)
            if best_value is not None:
                best_value = value
                best_epoch = epoch
            else:
                if mode == 'min' and best_value < value:
                    best_value = value
                elif mode == 'min' and best_value > value:
                    best_value = value
        return best_epoch, best_value

    def has_key(self, key: str, key2: str) -> bool:
        return key in self.stats[self.epoch] and key2 in self.stats[self.epoch][key]

    def plot(self):
        # matplotlib and tensorboard
        raise NotImplementedError
    
    def report(self, key: str, stats: Dict[str, Union[float, torch.Tensor]]):
        # key: train or eval
        d = self.stats[self.epoch][key]
        if len(d) != 0 and set(d) != set(stats):
            raise RuntimeError(f'keys mismatching: {set(d)} != {set(stats)}')

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                v = float(v)
            d[k].append(v)

    def total_count(self, key: str):
        count = 0
        for epoch in self.stats:
            d = self.stats[epoch][key]
            if len(d) == 0:
                return 0
            # Get the values from the first key
            values = list(d.values())[0]
            count += len(values)
        return count

    def state_dict(self):
        return {'stats': self.stats, 'epoch': self.epoch}

    def load_state_dict(self, state_dict: dict):
        self.stats = state_dict['stats']
        self.epoch = state_dict['epoch']
