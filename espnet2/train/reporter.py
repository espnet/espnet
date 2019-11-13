import datetime
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import Union, Dict, Tuple, Optional, Sequence, Any, List, \
    NamedTuple

import numpy as np
import humanfriendly
import torch
from typeguard import typechecked


class ReportType(Enum):
    average = 'average'
    accumulate = 'accumulate '


class ReportValue(NamedTuple):
    value: Any
    type: ReportType

    @staticmethod
    def to(v) -> 'ReportValue':
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach().numpy()
            if v.ndim in (1, 0):
                raise ValueError(f'must be 1-dim or 0-dim: {v.shape}')

            if v.dtype.kind == 'f':
                return ReportValue(float(v), ReportType.average)
            elif v.dtype.kind == 'i':
                return ReportValue(int(v), ReportType.accumulate)
            else:
                raise TypeError(f'Not supported: {v.dtype}')
        # By default, float value will be averaged
        elif isinstance(v, float):
            return ReportValue(v, ReportType.average)
        # By default, int value will be accumulated
        elif isinstance(v, int):
            return ReportValue(v, ReportType.accumulate)
        # If you need to customize the behaviour
        elif isinstance(v, ReportValue):
            return v
        else:
            raise NotImplementedError(f'Not supported type: {type(v)}')

    @staticmethod
    @typechecked
    def reduce(values: Union[List[float],
                             List[int],
                             List[torch.Tensor],
                             List[np.ndarray],
                             List['ReportValue']]):
        if isinstance(values[0], (torch.Tensor, np.ndarray)):
            if isinstance(values[0], torch.Tensor):
                values = [v.cpu().detach().numpy() for v in values]
            for v in values:
                if v.ndim in (1, 0):
                    raise ValueError(f'must be 1-dim or 0-dim: {v.shape}')

            if values[0].dtype.kind == 'f':
                v = np.nanmean(values)
            elif values[0].dtype.kind == 'f':
                v = np.nansum(values)
            else:
                raise TypeError(f'Not supported: {values[0].dtype}')
        elif isinstance(values[0], float):
            v = np.nanmean([v.value for v in values])
        elif isinstance(values[0], int):
            v = np.nansum([v.value for v in values])
        elif values[0].type == ReportType.average:
            v = np.nanmean([v.value for v in values])
        elif values[0].type == ReportType.accumulate:
            v = np.nansum([v.value for v in values])
        else:
            raise NotImplementedError(f'type={values[0].type}')
        return v


class SubReporter:
    """This class is used in Reporter

    See the docstring of Reporter for the usage.
    """
    def __init__(self, key: str, epoch: int, total_count: int):
        self.key = key
        self.epoch = epoch
        self.start_time = time.perf_counter()
        self.stats = defaultdict(list)
        self.total_count = total_count
        self._finished = False

    def get_total_count(self):
        return self.total_count

    @typechecked
    def register(self,
                 stats: Dict[str, Union[float, int,
                                        torch.Tensor, ReportValue]],
                 not_increment_count: bool = False):
        if self._finished:
            raise RuntimeError('Already finished')
        if not_increment_count:
            self.total_count += 1

        # key: train or eval
        if len(self.stats) != 0 and set(self.stats) != set(stats):
            raise RuntimeError(
                f'keys mismatching: {set(self.stats)} != {set(stats)}')

        for key2, v in stats.items():
            # if the input stats has None value, the key is not registered
            if v is None:
                continue
            r = ReportValue.to(v)
            self.stats[key2].append(r)

    def logging(self, logger=None, level: str = 'INFO', nlatest: int = None):
        if self._finished:
            raise RuntimeError('Already finished')
        if logger is None:
            logger = logging
        level = logging.getLevelName(level)

        if nlatest is None:
            nlatest = 0

        message = ''
        for key2, stats in self.stats.items():
            # values: List[ReportValue]
            values = stats[-nlatest:]
            if len(message) == 0:
                message += (
                    f'{self.epoch}epoch:{self.key}:'
                    f'{len(stats) - nlatest}-{len(stats)}batch: ')
            else:
                message += ', '

            if values[0].type == ReportType.average:
                v = np.nanmean([v.value for v in values])
            elif values[0].type == ReportType.accumulate:
                v = np.nansum([v.value for v in values])
            else:
                raise NotImplementedError(f'type={values[0].type}')

            message += f'{key2}={v}'
        logger.log(level, message)

    def finished(self):
        self._finished = True


class Reporter:
    """

    Examples:

        >>> reporter = Reporter()
        >>> with reporter.start('train') as sub_reporter:
        ...     for batch in iterator:
        ...         stats = dict(
        ...             # If float, the averaged value is reported.
        ...             loss=0.2,
        ...             # If int, the accumulated value is reported.
        ...             count=9,
        ...             # Even if float, you need the summation, then use
        ...             # ReportValue to tell the desired type.
        ...             something=ReportValue(0.4, ReportType.accumulate))
        ...         sub_reporter.register(stats)

    """
    @typechecked
    def __init__(self, epoch: int = 1):
        if epoch <= 0:
            raise RuntimeError(f'epoch must be 1 or more: {epoch}')
        self.epoch = epoch
        # stats: Dict[int, Dict[str, Dict[str, float]]]
        # e.g. self.stats[epoch]['train']['loss']
        self.stats = defaultdict(dict)

    def get_epoch(self) -> int:
        return self.epoch

    def set_epoch(self, epoch: int):
        if epoch <= 0:
            raise RuntimeError(f'epoch must be 1 or more: {epoch}')
        self.epoch = epoch

    @contextmanager
    def start(self, key: str, epoch: int = None) -> SubReporter:
        sub_reporter = self.start_epoch(key, epoch)
        yield sub_reporter
        # Receive the stats from sub_reporter
        self.finish_epoch(sub_reporter)

    def start_epoch(self, key: str, epoch: int = None) -> SubReporter:
        if epoch is not None:
            if epoch <= 0:
                raise RuntimeError(f'epoch must be 1 or more: {epoch}')
            self.epoch = epoch

        if self.epoch - 1 not in self.stats or \
                key not in self.stats[self.epoch - 1]:
            # If the previous epoch doesn't exist for some reason,
            # maybe it's bug, this case also indicates 0-count.
            if self.epoch - 1 != 0:
                logging.warning(
                    f'The stats of the previous epoch={self.epoch - 1}'
                    f'doesn\'t exist.')
            total_count = 0
        else:
            total_count = self.stats[self.epoch - 1][key]['total_count']

        sub_reporter = SubReporter(key, self.epoch, total_count)
        # Clear the stats for the next epoch if it exists
        self.stats.pop(epoch, None)
        return sub_reporter

    def finish_epoch(self, sub_reporter: SubReporter):
        # Note(k): Reporter and SubReporter is tight coupled due to this method

        # Calc mean of current stats and set it for previous epochs stats
        stats = {}
        for key2, values in sub_reporter.stats.items():
            v = ReportValue.reduce(values)
            stats[key2] = v

        if 'time' in stats:
            raise RuntimeError(f'time is reserved: {stats}')
        stats['time'] = datetime.timedelta(
            seconds=time.perf_counter() - sub_reporter.start_time)

        if 'total_count' in stats:
            raise RuntimeError(f'total_count is reserved: {stats}')
        stats['total_count'] = sub_reporter.total_count

        self.stats[self.epoch][sub_reporter.key] = stats
        sub_reporter.finished()

    def best_epoch_and_value(self, key: str, key2: str, mode: str) \
            -> Tuple[Optional[int], Optional[float]]:
        assert mode in ('min', 'max'), mode

        # iterate from the last epoch
        best_value = None
        best_epoch = None
        for epoch in sorted(self.stats):
            value = self.stats[epoch][key][key2]

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
            epoch = max(self.stats)
        return key in self.stats[epoch] and key2 in self.stats[epoch][key]

    def show_stats(self, logger=None, level: str = 'INFO', epoch: int = None):
        if logger is None:
            logger = logging
        if epoch is None:
            epoch = max(self.stats)
        level = logging.getLevelName(level)

        message = ''
        for key, d in self.stats[epoch].items():
            _message = ''
            for key2, v in d.items():
                if v is not None:
                    if len(_message) != 0:
                        _message += ', '
                    if isinstance(v, float):
                        _message += f'{key2}={v:.3f}'
                    elif isinstance(v, datetime.timedelta):
                        _v = humanfriendly.format_timespan(v)
                        _message += f'{key2}={_v}'
                    else:
                        _message += f'{key2}={v}'
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
            epoch = max(self.stats)
        values = self.stats[epoch][key][key2]
        return np.nanmean(values)

    def get_keys(self, epoch: int = None) -> Tuple[str]:
        if epoch is None:
            epoch = max(self.stats)
        return tuple(self.stats[epoch])

    def get_keys2(self, epoch: int = None) -> Tuple[str]:
        if epoch is None:
            epoch = max(self.stats)
        for d in self.stats[epoch].values():
            keys2 = tuple(k for k in d if k not in ('time', 'total_count'))
            return keys2

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

        epochs = np.arange(1, max(self.stats))
        for key in keys:
            y = [np.nanmean(self.stats[e][key][key2])
                 if e in self.stats else np.nan
                 for e in epochs]
            assert len(epochs) == len(y), 'Bug?'

            plt.plot(epochs, y, label=key)
        plt.legend()
        plt.title(f'epoch vs {key2}')
        plt.xlabel('epoch')
        plt.ylabel(key2)
        plt.grid()

        return plt

    def state_dict(self):
        return {'stats': self.stats,
                'epoch': self.epoch}

    def load_state_dict(self, state_dict: dict):
        self.epoch = state_dict['epoch']
        self.stats = state_dict['stats']

