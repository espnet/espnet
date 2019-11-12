import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Union, Dict, Tuple, Optional, Sequence

import numpy as np
import torch
from typeguard import typechecked


class Reporter:
    @typechecked
    def __init__(self, epoch: int = 1):
        # These are permanent states
        self.epoch = epoch
        self.total_count = 0
        # Dict[int, Dict[str, Dict[str, float]]]
        # e.g. self.previous_epochs_stats[3]['train']['loss']
        self.previous_epochs_stats = {}

        # Temporal states: Dict[str, Dict[str, List[float]]]
        self.current_epoch_stats = None

    def get_epoch(self) -> int:
        return self.epoch

    def get_total_count(self):
        return self.total_count

    @contextmanager
    def start_epoch(self, epoch: int = None):
        if epoch is not None:
            self.epoch = epoch
        # Refresh the current stat
        self._set_epoch(self.epoch)
        yield self
        self.finish_epoch()
        self.epoch += 1

    def finish_epoch(self):
        # Calc mean of current stats and set it for previous epochs stats
        stats = {}
        for key, d in self.current_epoch_stats.items():
            stats[key] = {}
            for key2, values in d.items():
                stats[key][key2] = np.nanmean(values)

        self.previous_epochs_stats[self.epoch] = stats
        self.current_epoch_stats = None

    def _set_epoch(self, epoch: int):
        self.epoch = epoch
        # Clear the stats for the next epoch if it exists
        self.previous_epochs_stats.pop(epoch, None)
        # Clear current stats
        self.current_epoch_stats = defaultdict(lambda: defaultdict(list))

    def register(self, key: str, stats: Dict[str, Union[float, torch.Tensor]]):
        if self.current_epoch_stats is None:
            raise RuntimeError('Please call start_epoch() before register()')

        self.total_count += 1

        # key: train or eval
        d = self.current_epoch_stats[key]
        if len(d) != 0 and set(d) != set(stats):
            raise RuntimeError(f'keys mismatching: {set(d)} != {set(stats)}')

        for key2, v in stats.items():
            # if the input stats has None value, the key is not registered
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = float(v)
            d[key2].append(v)

    def logging(self, key: str, logger=None,
                level: str = 'INFO', nlatest: int = None):
        if self.current_epoch_stats is None:
            raise RuntimeError('Please call start_epoch() before logging()')
        if logger is None:
            logger = logging
        level = logging.getLevelName(level)

        if nlatest is None:
            nlatest = 0

        message = ''
        for key2, stats in self.current_epoch_stats[key].items():
            values = stats[-nlatest:]
            if len(message) == 0:
                message += (
                    f'{self.epoch}epoch:{key}:' 
                    f'{len(stats) - nlatest}-{len(stats)}batch: ')
            else:
                message += ', '
            v = np.nanmean(values)
            message += f'{key2}={v}'
        logger.log(level, message)

    def best_epoch_and_value(self, key: str, key2: str, mode: str) \
            -> Tuple[Optional[int], Optional[float]]:
        assert mode in ('min', 'max'), mode

        # iterate from the last epoch
        best_value = None
        best_epoch = None
        for epoch in sorted(self.previous_epochs_stats):
            value = self.previous_epochs_stats[epoch][key][key2]

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

    def has_key(self, key: str, key2: str, epoch: int = None) -> bool:
        if epoch is None:
            epoch = max(self.previous_epochs_stats)
        return key in self.previous_epochs_stats[epoch] and \
            key2 in self.previous_epochs_stats[epoch][key]

    def show_stats(self, logger=None, level: str = 'INFO',
                   epoch: int = None):
        if logger is None:
            logger = logging
        if epoch is None:
            epoch = max(self.previous_epochs_stats)
        level = logging.getLevelName(level)

        message = ''
        for key, d in self.previous_epochs_stats[epoch].items():
            _message = ''
            for key2, v in d.items():
                if v is not None:
                    if len(_message) != 0:
                        _message += ', '
                    _message += f'{key2}={v:.3f}'
            if len(_message) != 0:
                if len(message) == 0:
                    message += f'{epoch}epoch results: '
                else:
                    message += ', '
                message += f'[{key}] {_message}'
        logger.log(level, message)

    @typechecked
    def get_value(self, key: str, key2: str, epoch: int = None):
        if epoch is None:
            epoch = max(self.previous_epochs_stats)
        values = self.previous_epochs_stats[epoch][key][key2]
        return np.nanmean(values)

    def get_keys(self, epoch: int = None) -> Tuple[str]:
        if epoch is None:
            epoch = max(self.previous_epochs_stats)
        return tuple(self.previous_epochs_stats[epoch])

    def get_keys2(self, epoch: int = None) -> Tuple[str]:
        if epoch is None:
            epoch = max(self.previous_epochs_stats)
        for d in self.previous_epochs_stats[epoch].values():
            return tuple(d)

    @typechecked
    def plot_stats(self, keys: Sequence[str], key2: str, plt=None):
        if plt is None:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
        plt.clf()

        # str is also Sequence[str]
        if isinstance(keys, str):
            raise TypeError(f'Input as [{keys}]')

        epochs = np.arange(1, max(self.previous_epochs_stats))
        for key in keys:
            y = [np.nanmean(self.previous_epochs_stats[e][key][key2])
                 if e in self.previous_epochs_stats else np.nan
                 for e in epochs]
            assert len(epochs) == len(y), 'Bug?'

            plt.plot(epochs, y, label=key)
        plt.legend()
        plt.title(f'epoch vs {key2}')
        plt.xlabel('epoch')
        plt.ylabel(key2)
        plt.grid()

        return  plt

    def state_dict(self):
        return {'previous_epochs_stats': self.previous_epochs_stats,
                'epoch': self.epoch,
                'total_epoch': self.total_count,
                }

    def load_state_dict(self, state_dict: dict):
        self.total_count = state_dict['total_epoch']
        self.epoch = state_dict['epoch']
        self.previous_epochs_stats = state_dict['previous_epochs_stats']
