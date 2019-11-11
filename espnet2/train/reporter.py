import itertools
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, Sequence

import numpy as np
import torch
from typeguard import typechecked

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Reporter:
    """

    About the structure of Reporter.stats:
        >>> epoch = 0
        >>> reporter = Reporter('output')
        >>> epoch_stats = reporter.stats[epoch]
        >>> stats = epoch_stats ['train']  # or eval
        >>> acc_list = stats['acc']
        >>> mean_value = np.nanmean(acc_list)

    """
    @typechecked
    def __init__(self, output_dir: Union[str, Path]):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # self.stats: Dict[int, Dict[str, List[float]]]
        self.stats = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def get_epoch(self) -> int:
        return self.epoch

    @typechecked
    def get_value(self, key: str, key2: str, epoch: int = None):
        if epoch is None:
            epoch = self.epoch
        return np.nanmean(self.stats[epoch][key][key2])

    @typechecked
    def best_epoch_and_value(self, key: str, key2: str, mode: str) \
            -> Tuple[Optional[int], Optional[float]]:
        assert mode in ('min', 'max'), mode

        # iterate from the last epoch
        best_value = None
        best_epoch = None
        for epoch in sorted(self.stats):
            values = self.stats[epoch][key][key2]
            value = np.nanmean(values)

            # If at the first iteration:
            if best_value is None:
                best_value = value
                best_epoch = epoch
            else:
                if mode == 'min' and best_value < value:
                    best_value = value
                elif mode == 'min' and best_value > value:
                    best_value = value
        return best_epoch, best_value

    @typechecked
    def has_key(self, key: str, key2: str, epoch: int = None) -> bool:
        if epoch is None:
            epoch = self.epoch
        return key in self.stats[epoch] and key2 in self.stats[epoch][key]

    def plot(self, key: Sequence[str], key2: Sequence[str]):
        assert not isinstance(key, str), f'Input as [{key}]'
        assert not isinstance(key2, str), f'Input as [{key2}]'
        plt.clf()
        for k, k2 in itertools.product(key, key2):
            x = np.arange(1, max(self.stats))
            y = [np.nanmean(self.stats[e][k][k2])
                 if e in self.stats else np.nan for e in x]
            plt.plot(x, y, label=f'{k}-{k2}')
        plt.savefig(self.output_dir / 'plot.png')
        plt.clf()

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
        def to_vanilla_dict(stats: defaultdict) -> dict:
            return {k: to_vanilla_dict(v) if isinstance(v, defaultdict) else v
                    for k, v in stats.items()}
        return {'stats': self.stats, 'epoch': self.epoch}

    def load_state_dict(self, state_dict: dict):
        def to_default_dict(stats: dict) -> defaultdict:
            raise NotImplementedError

        stats = state_dict['stats']
        self.stats = to_default_dict(stats)
        self.epoch = state_dict['epoch']
