"""Reporter module."""

import dataclasses
import datetime
import logging
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Dict, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
from packaging.version import parse as V
from typeguard import typechecked

Num = Union[float, int, complex, torch.Tensor, np.ndarray]


_reserved = {"time", "total_count"}


@typechecked
def to_reported_value(v: Num, weight: Optional[Num] = None) -> "ReportedValue":
    """
    Convert a value and an optional weight into a reported value.

    This function takes a numerical value and an optional weight and converts
    them into a reported value. The reported value can either be a simple
    average or a weighted average depending on the presence of the weight.

    Args:
        v (Num): The value to be reported. It can be a float, int, complex,
            torch.Tensor, or np.ndarray. The value must be a scalar (0 or 1
            dimension).
        weight (Optional[Num]): An optional weight associated with the value.
            It can also be a float, int, complex, torch.Tensor, or np.ndarray,
            and must be a scalar.

    Returns:
        ReportedValue: An instance of either `Average` or `WeightedAverage`
            based on whether a weight was provided.

    Raises:
        ValueError: If the provided value `v` or weight is not 0 or 1
            dimension.

    Examples:
        >>> to_reported_value(5)
        Average(value=5)

        >>> to_reported_value(5, weight=2)
        WeightedAverage(value=(5, 2))
    """
    if isinstance(v, (torch.Tensor, np.ndarray)):
        if np.prod(v.shape) != 1:
            raise ValueError(f"v must be 0 or 1 dimension: {len(v.shape)}")
        v = v.item()

    if isinstance(weight, (torch.Tensor, np.ndarray)):
        if np.prod(weight.shape) != 1:
            raise ValueError(f"weight must be 0 or 1 dimension: {len(weight.shape)}")
        weight = weight.item()

    if weight is not None:
        retval = WeightedAverage(v, weight)
    else:
        retval = Average(v)
    return retval


@typechecked
def aggregate(values: Sequence["ReportedValue"]) -> Num:
    """
        Aggregate values of reported statistics.

    This function computes the aggregated value from a sequence of
    ReportedValue instances. It supports both Average and WeightedAverage
    types. If the sequence contains invalid values, it will exclude them
    from the calculation. If the input is empty, it raises a warning and
    returns NaN.

    Args:
        values (Sequence[ReportedValue]): A sequence of ReportedValue
        instances (Average or WeightedAverage) to aggregate.

    Returns:
        Num: The aggregated value, which can be of type float, int, complex,
        torch.Tensor, or np.ndarray.

    Raises:
        ValueError: If the input contains different types of ReportedValue
        instances or if the input sequence is empty.
        NotImplementedError: If the type of the values is not supported.

    Examples:
        >>> avg1 = Average(2.0)
        >>> avg2 = Average(4.0)
        >>> result = aggregate([avg1, avg2])
        >>> print(result)
        3.0

        >>> wavg1 = WeightedAverage((3.0, 1.0), 0.5)
        >>> wavg2 = WeightedAverage((7.0, 2.0), 1.5)
        >>> result = aggregate([wavg1, wavg2])
        >>> print(result)
        5.0

        >>> result = aggregate([])
        >>> print(result)
        nan
    """

    for v in values:
        if not isinstance(v, type(values[0])):
            raise ValueError(
                f"Can't use different Reported type together: "
                f"{type(v)} != {type(values[0])}"
            )

    if len(values) == 0:
        warnings.warn("No stats found")
        retval = np.nan

    elif isinstance(values[0], Average):
        retval = np.nanmean([v.value for v in values])

    elif isinstance(values[0], WeightedAverage):
        # Excludes non finite values
        invalid_indices = set()
        for i, v in enumerate(values):
            if not np.isfinite(v.value) or not np.isfinite(v.weight):
                invalid_indices.add(i)
        values = [v for i, v in enumerate(values) if i not in invalid_indices]

        if len(values) != 0:
            # Calc weighed average. Weights are changed to sum-to-1.
            sum_weights = sum(v.weight for i, v in enumerate(values))
            sum_value = sum(v.value * v.weight for i, v in enumerate(values))
            if sum_weights == 0:
                warnings.warn("weight is zero")
                retval = np.nan
            else:
                retval = sum_value / sum_weights
        else:
            warnings.warn("No valid stats found")
            retval = np.nan

    else:
        raise NotImplementedError(f"type={type(values[0])}")
    return retval


def wandb_get_prefix(key: str):
    """
    Get the prefix for logging keys based on their category.

    This function determines the appropriate prefix for a given key based
    on its category, which can be 'valid', 'train', 'attn', or others.
    The prefix helps in organizing the logged data in a structured way.

    Args:
        key (str): The key for which to determine the prefix.

    Returns:
        str: The prefix corresponding to the provided key.

    Examples:
        >>> wandb_get_prefix("valid_loss")
        'valid/'
        >>> wandb_get_prefix("train_accuracy")
        'train/'
        >>> wandb_get_prefix("attn_weights")
        'attn/'
        >>> wandb_get_prefix("other_metric")
        'metrics/'
    """
    if key.startswith("valid"):
        return "valid/"
    if key.startswith("train"):
        return "train/"
    if key.startswith("attn"):
        return "attn/"
    return "metrics/"


class ReportedValue:
    """
        ReportedValue serves as a base class for representing reported values in
    the reporting system of the ESPnet2 framework.

    Attributes:
        None

    Examples:
        >>> avg = Average(value=0.5)
        >>> w_avg = WeightedAverage(value=(0.5, 0.2), weight=0.7)
    """

    pass


@dataclasses.dataclass(frozen=True)
class Average(ReportedValue):
    """
        Average class that represents a simple average value.

    Attributes:
        value (Num): The average value represented by this instance.

    Examples:
        >>> avg = Average(5.0)
        >>> print(avg.value)
        5.0
    """

    value: Num


@dataclasses.dataclass(frozen=True)
class WeightedAverage(ReportedValue):
    """
        Represents a weighted average of reported values.

    Attributes:
        value (Tuple[Num, Num]): A tuple containing the value and its associated weight.
        weight (Num): The weight associated with the reported value.

    Examples:
        >>> weighted_avg = WeightedAverage(value=(3.0, 5.0), weight=0.5)
        >>> print(weighted_avg.value)
        (3.0, 5.0)
        >>> print(weighted_avg.weight)
        0.5
    """

    value: Tuple[Num, Num]
    weight: Num


class SubReporter:
    """
    This class is used in Reporter.

    It is responsible for collecting and managing statistics
    during training iterations for a specific key (e.g., 'train' or 'valid').

    Attributes:
        key (str): The identifier for the current reporting key.
        epoch (int): The current epoch number.
        start_time (float): The time when the reporting for this key started.
        stats (defaultdict): A dictionary to hold statistics collected during
            the epoch.
        _finished (bool): A flag indicating if the reporting for this key is finished.
        total_count (int): The total number of iterations over all epochs.
        count (int): The number of iterations for the current epoch.
        _seen_keys_in_the_step (set): A set to track keys seen in the current step.

    Args:
        key (str): The reporting key.
        epoch (int): The current epoch number.
        total_count (int): The total count of iterations across epochs.

    Methods:
        get_total_count(): Returns the number of iterations over all epochs.
        get_epoch(): Returns the current epoch number.
        next(): Closes up the current step and resets the state for the next step.
        register(stats, weight=None): Registers statistics for the current step.
        log_message(start=None, end=None): Logs a message summarizing statistics.
        tensorboard_add_scalar(summary_writer, start=None): Adds scalar values
            to TensorBoard.
        wandb_log(start=None): Logs statistics to Weights & Biases.
        finished(): Marks the reporting for this key as finished.
        measure_time(name): A context manager to measure and register time.
        measure_iter_time(iterable, name): Measures time for each iteration in
            the given iterable.

    Examples:
        >>> sub_reporter = SubReporter(key='train', epoch=1, total_count=0)
        >>> sub_reporter.register({'loss': 0.2})
        >>> print(sub_reporter.log_message())
        "1epoch:train:1-1batch: loss=0.200"
    """

    @typechecked
    def __init__(self, key: str, epoch: int, total_count: int):
        self.key = key
        self.epoch = epoch
        self.start_time = time.perf_counter()
        self.stats = defaultdict(list)
        self._finished = False
        self.total_count = total_count
        self.count = 0
        self._seen_keys_in_the_step = set()

    def get_total_count(self) -> int:
        """
        Returns the number of iterations over all epochs.

        This method provides the total count of iterations that have been
        registered across all epochs within the SubReporter. It is useful for
        tracking the number of completed iterations during training or evaluation
        phases.

        Returns:
            int: The total count of iterations.

        Examples:
            >>> sub_reporter = SubReporter(key='train', epoch=1, total_count=5)
            >>> sub_reporter.get_total_count()
            5
        """
        return self.total_count

    def get_epoch(self) -> int:
        """
        Get the current epoch number.

        Returns:
            int: The current epoch number.

        Examples:
            >>> reporter = Reporter(epoch=5)
            >>> reporter.get_epoch()
            5
        """
        return self.epoch

    def next(self):
        """
                This class is used in Reporter.

        See the docstring of Reporter for the usage.

        Attributes:
            key (str): The key for the reported values.
            epoch (int): The current epoch number.
            start_time (float): The time when the reporting started.
            stats (defaultdict): A dictionary holding the statistics collected.
            _finished (bool): A flag indicating whether the reporting is finished.
            total_count (int): The total number of counts across epochs.
            count (int): The current count for the ongoing epoch.
            _seen_keys_in_the_step (set): A set to track seen keys during the step.

        Args:
            key (str): A string representing the key for the current report.
            epoch (int): An integer representing the current epoch.
            total_count (int): An integer representing the total count of iterations.

        Examples:
            >>> reporter = Reporter()
            >>> with reporter.observe('train') as sub_reporter:
            ...     for batch in iterator:
            ...         stats = dict(loss=0.2)
            ...         sub_reporter.register(stats)

        Note:
            This class facilitates the collection and logging of statistics during
            the training process.
        """
        for key, stats_list in self.stats.items():
            if key not in self._seen_keys_in_the_step:
                # Fill nan value if the key is not registered in this step
                if isinstance(stats_list[0], WeightedAverage):
                    stats_list.append(to_reported_value(np.nan, 0))
                elif isinstance(stats_list[0], Average):
                    stats_list.append(to_reported_value(np.nan))
                else:
                    raise NotImplementedError(f"type={type(stats_list[0])}")

            assert len(stats_list) == self.count, (len(stats_list), self.count)

        self._seen_keys_in_the_step = set()

    @typechecked
    def register(
        self,
        stats: Dict[str, Optional[Union[Num, Dict[str, Num]]]],
        weight: Optional[Num] = None,
    ) -> None:
        """
        Register statistics for the current step.

        This method allows you to log various statistics during a training step.
        If the key is being registered for the first time in the current step,
        it will initialize the statistics list for that key with NaN values for
        previous counts.

        Args:
            stats: A dictionary where the keys are statistic names and the values
                are either a numeric value or another dictionary containing
                additional metrics.
            weight: An optional weight for the registered statistics. If provided,
                it will be used in the calculation of weighted averages.

        Raises:
            RuntimeError: If the registration is attempted after the step has
                been marked as finished.
            RuntimeError: If a reserved key is used.
            RuntimeError: If a key is registered more than once in the same step.

        Examples:
            >>> sub_reporter = SubReporter('train', 1, 0)
            >>> sub_reporter.register({'loss': 0.2, 'accuracy': 0.95})
            >>> sub_reporter.register({'loss': 0.15}, weight=0.5)

        Note:
            The `stats` dictionary can contain nested dictionaries, allowing for
            more complex logging structures.
        """
        if self._finished:
            raise RuntimeError("Already finished")
        if len(self._seen_keys_in_the_step) == 0:
            # Increment count as the first register in this step
            self.total_count += 1
            self.count += 1

        for key2, v in stats.items():
            if key2 in _reserved:
                raise RuntimeError(f"{key2} is reserved.")
            if key2 in self._seen_keys_in_the_step:
                raise RuntimeError(f"{key2} is registered twice.")
            if v is None:
                v = np.nan
            r = to_reported_value(v, weight)

            if key2 not in self.stats:
                # If it's the first time to register the key,
                # append nan values in front of the the value
                # to make it same length to the other stats
                # e.g.
                # stat A: [0.4, 0.3, 0.5]
                # stat B: [nan, nan, 0.2]
                nan = to_reported_value(np.nan, None if weight is None else 0)
                self.stats[key2].extend(
                    r if i == self.count - 1 else nan for i in range(self.count)
                )
            else:
                self.stats[key2].append(r)
            self._seen_keys_in_the_step.add(key2)

    def log_message(self, start: int = None, end: int = None) -> str:
        """
        Generates a log message summarizing the statistics for the current epoch.

        The message includes statistics collected during the current epoch,
        formatted as a string. The user can specify the range of batches
        to include in the log message.

        Args:
            start (int, optional): The starting batch index for the log message.
                If None, defaults to 0. If negative, it is treated as an
                offset from the total count.
            end (int, optional): The ending batch index for the log message.
                If None, defaults to the current count of batches.

        Returns:
            str: A formatted string containing the epoch, key, batch range,
                and aggregated statistics.

        Raises:
            RuntimeError: If the reporter has already finished logging.

        Examples:
            >>> sub_reporter = SubReporter("train", 1, 10)
            >>> sub_reporter.register({"loss": 0.5})
            >>> message = sub_reporter.log_message(0, 1)
            >>> print(message)  # Output: "1epoch:train:1-1batch: loss=0.500"
        """
        if self._finished:
            raise RuntimeError("Already finished")
        if start is None:
            start = 0
        if start < 0:
            start = self.count + start
        if end is None:
            end = self.count

        if self.count == 0 or start == end:
            return ""

        message = f"{self.epoch}epoch:{self.key}:" f"{start + 1}-{end}batch: "

        for idx, (key2, stats_list) in enumerate(self.stats.items()):
            assert len(stats_list) == self.count, (len(stats_list), self.count)
            # values: List[ReportValue]
            values = stats_list[start:end]
            if idx != 0 and idx != len(stats_list):
                message += ", "

            v = aggregate(values)
            if abs(v) > 1.0e3:
                message += f"{key2}={v:.3e}"
            elif abs(v) > 1.0e-3:
                message += f"{key2}={v:.3f}"
            else:
                message += f"{key2}={v:.3e}"
        return message

    def tensorboard_add_scalar(self, summary_writer, start: int = None):
        """
        Logs scalar values to TensorBoard.

        This method aggregates the reported values from the current
        statistics and logs them to TensorBoard using the provided
        summary writer. The scalar values are logged with the key
        corresponding to the statistic name.

        Args:
            summary_writer: The TensorBoard summary writer used to log
                the scalar values.
            start: Optional; the starting index for logging. If not
                provided, it defaults to 0. If negative, it counts
                backwards from the current count.

        Examples:
            >>> from tensorboardX import SummaryWriter
            >>> writer = SummaryWriter()
            >>> sub_reporter.tensorboard_add_scalar(writer)

        Raises:
            AssertionError: If the lengths of the statistics list
                do not match the current count.

        Note:
            This method assumes that the statistics have been registered
            prior to logging. Ensure that the `register` method is called
            to populate the statistics before invoking this method.
        """
        if start is None:
            start = 0
        if start < 0:
            start = self.count + start

        for key2, stats_list in self.stats.items():
            assert len(stats_list) == self.count, (len(stats_list), self.count)
            # values: List[ReportValue]
            values = stats_list[start:]
            v = aggregate(values)
            summary_writer.add_scalar(f"{key2}", v, self.total_count)

    def wandb_log(self, start: int = None):
        """
        Logs the current statistics to Weights & Biases (wandb).

        This method aggregates the statistics collected during the reporting
        period and logs them to the Weights & Biases dashboard. The statistics
        are prefixed based on their type (e.g., training, validation) and the
        total iteration count is also included in the log.

        Args:
            start (int, optional): The starting index for the statistics to log.
                If None, logging starts from index 0. If negative, it counts
                backwards from the current count.

        Examples:
            >>> sub_reporter = SubReporter('train', epoch=1, total_count=10)
            >>> # Register some statistics
            >>> sub_reporter.register({'loss': 0.5, 'accuracy': 0.8})
            >>> # Log statistics to wandb
            >>> sub_reporter.wandb_log()

        Raises:
            AssertionError: If the length of the stats list does not match the
            current count.
        """
        import wandb

        if start is None:
            start = 0
        if start < 0:
            start = self.count + start

        d = {}
        for key2, stats_list in self.stats.items():
            assert len(stats_list) == self.count, (len(stats_list), self.count)
            # values: List[ReportValue]
            values = stats_list[start:]
            v = aggregate(values)
            d[wandb_get_prefix(key2) + key2] = v
        d["iteration"] = self.total_count
        wandb.log(d)

    def finished(self) -> None:
        """
                This class is used in Reporter.

        See the docstring of Reporter for the usage.

        Attributes:
            key (str): The key identifier for the reporter.
            epoch (int): The current epoch number.
            start_time (float): The time when the reporting started.
            stats (defaultdict): A dictionary that holds the statistics for reporting.
            _finished (bool): A flag indicating if the reporting is finished.
            total_count (int): The total number of counts recorded.
            count (int): The count of registered statistics in the current step.
            _seen_keys_in_the_step (set): A set of keys that have been seen in the current step.

        Args:
            key (str): The key identifier for the reporter.
            epoch (int): The current epoch number.
            total_count (int): The total number of counts recorded.

        Examples:
            >>> sub_reporter = SubReporter("train", 1, 0)
            >>> sub_reporter.get_total_count()
            0
            >>> sub_reporter.register({"loss": 0.5}, weight=1.0)
            >>> message = sub_reporter.log_message()
            >>> print(message)
            "1epoch:train:1-1batch: loss=0.500"

        Raises:
            RuntimeError: If an attempt is made to register statistics after finishing.
        """
        self._finished = True

    @contextmanager
    def measure_time(self, name: str):
        """
                Measures the time taken for a block of code to execute.

        This context manager yields the start time and registers the elapsed time
        once the block is exited. It is particularly useful for tracking the duration
        of specific operations within the reporting framework.

        Args:
            name (str): A descriptive name for the timing measurement.

        Yields:
            float: The start time in seconds since the epoch.

        Examples:
            >>> with sub_reporter.measure_time("data_loading"):
            ...     load_data()  # Some function to load data
            >>> print(sub_reporter.stats)  # Will show the recorded time under the key "data_loading"

        Note:
            Ensure that this context manager is used within a valid instance of
            SubReporter.
        """
        start = time.perf_counter()
        yield start
        t = time.perf_counter() - start
        self.register({name: t})

    def measure_iter_time(self, iterable, name: str):
        """
        Measures the time taken for each iteration over the given iterable.

        This method will yield each item from the iterable while registering the
        time taken for each iteration using the specified name. The time taken
        is recorded in the statistics of the SubReporter.

        Args:
            iterable (iterable): An iterable object to iterate over.
            name (str): The name under which to register the time taken for
                each iteration.

        Yields:
            The next item from the iterable.

        Examples:
            >>> sub_reporter = SubReporter("example", 1, 0)
            >>> for item in sub_reporter.measure_iter_time(range(3), "iteration_time"):
            ...     print(item)

        Note:
            If the iterable is empty, the function will not yield any items.

        Raises:
            RuntimeError: If the sub-reporter has already finished.
        """
        iterator = iter(iterable)
        while True:
            try:
                start = time.perf_counter()
                retval = next(iterator)
                t = time.perf_counter() - start
                self.register({name: t})
                yield retval
            except StopIteration:
                break


class Reporter:
    """
        Reporter class.

    This class is responsible for reporting training and evaluation statistics
    during the training process of a machine learning model. It keeps track of
    various metrics over multiple epochs and allows for organized logging and
    visualization of these metrics.

    Attributes:
        epoch (int): The current epoch number.
        stats (dict): A dictionary to store statistics for each epoch and key.

    Examples:
        >>> reporter = Reporter()
        >>> with reporter.observe('train') as sub_reporter:
        ...     for batch in iterator:
        ...         stats = dict(loss=0.2)
        ...         sub_reporter.register(stats)

    Methods:
        get_epoch() -> int:
            Returns the current epoch number.

        set_epoch(epoch: int) -> None:
            Sets the current epoch number.

        observe(key: str, epoch: Optional[int] = None) -> ContextManager[SubReporter]:
            Context manager to observe metrics for a specific key.

        start_epoch(key: str, epoch: Optional[int] = None) -> SubReporter:
            Initializes a new SubReporter for the specified key and epoch.

        finish_epoch(sub_reporter: SubReporter) -> None:
            Finalizes and records the statistics collected in the given SubReporter.

        sort_epochs_and_values(key: str, key2: str, mode: str) -> List[Tuple[int, float]]:
            Returns a list of epochs and their corresponding values sorted by mode.

        check_early_stopping(patience: int, key1: str, key2: str, mode: str,
                             epoch: Optional[int] = None, logger: Optional[logging.Logger] = None) -> bool:
            Checks if early stopping criteria are met based on the specified key.

        has(key: str, key2: str, epoch: Optional[int] = None) -> bool:
            Checks if the specified keys exist in the statistics for the given epoch.

        log_message(epoch: Optional[int] = None) -> str:
            Generates a formatted log message for the specified epoch.

        get_value(key: str, key2: str, epoch: Optional[int] = None):
            Retrieves the value for the specified keys and epoch.

        get_keys(epoch: Optional[int] = None) -> Tuple[str, ...]:
            Returns the first-level keys (e.g., 'train', 'eval') for the specified epoch.

        get_keys2(key: str, epoch: Optional[int] = None) -> Tuple[str, ...]:
            Returns the second-level keys (e.g., 'loss', 'acc') for the specified key and epoch.

        get_all_keys(epoch: Optional[int] = None) -> Tuple[Tuple[str, str], ...]:
            Returns all key pairs for the specified epoch.

        matplotlib_plot(output_dir: Union[str, Path]) -> None:
            Plots statistics using Matplotlib and saves the images to the specified directory.

        tensorboard_add_scalar(summary_writer, epoch: Optional[int] = None,
                               key1: Optional[str] = None) -> None:
            Adds scalar values to TensorBoard for visualization.

        wandb_log(epoch: Optional[int] = None) -> None:
            Logs metrics to Weights & Biases for tracking.

        state_dict() -> dict:
            Returns the state of the Reporter as a dictionary.

        load_state_dict(state_dict: dict) -> None:
            Loads the state of the Reporter from a given dictionary.
    """

    @typechecked
    def __init__(self, epoch: int = 0):
        if epoch < 0:
            raise ValueError(f"epoch must be 0 or more: {epoch}")
        self.epoch = epoch
        # stats: Dict[int, Dict[str, Dict[str, float]]]
        # e.g. self.stats[epoch]['train']['loss']
        self.stats = {}

    def get_epoch(self) -> int:
        """
                Returns the current epoch number.

        This method retrieves the epoch number that the reporter is currently in.

        Returns:
            int: The current epoch number.

        Examples:
            >>> reporter = Reporter(epoch=5)
            >>> current_epoch = reporter.get_epoch()
            >>> print(current_epoch)
            5
        """
        return self.epoch

    def set_epoch(self, epoch: int) -> None:
        """
            Sets the current epoch of the reporter.

        This method updates the epoch counter to the specified value. The epoch
        must be a non-negative integer. If a negative value is provided, a
        ValueError will be raised.

        Args:
            epoch (int): The new epoch number to set. Must be 0 or more.

        Raises:
            ValueError: If the provided epoch is less than 0.

        Examples:
            >>> reporter = Reporter()
            >>> reporter.set_epoch(1)
            >>> reporter.get_epoch()
            1
            >>> reporter.set_epoch(-1)  # This will raise a ValueError
        """
        if epoch < 0:
            raise ValueError(f"epoch must be 0 or more: {epoch}")
        self.epoch = epoch

    @contextmanager
    def observe(self, key: str, epoch: int = None) -> ContextManager[SubReporter]:
        """
            Observe a specific key during a training or evaluation epoch.

        This context manager allows for the registration of statistics
        for a specific key (e.g., 'train' or 'eval') during an epoch.
        It will yield a SubReporter instance that can be used to
        collect statistics, which will be finalized once the context
        manager exits.

        Args:
            key (str): The key for which statistics are being collected.
            epoch (int, optional): The epoch number. If None, the current
                epoch will be used.

        Yields:
            SubReporter: An instance of SubReporter to register statistics.

        Raises:
            ValueError: If the provided epoch is negative.

        Examples:
            >>> reporter = Reporter()
            >>> with reporter.observe('train') as sub_reporter:
            ...     for batch in iterator:
            ...         stats = dict(loss=0.2)
            ...         sub_reporter.register(stats)

        Note:
            The statistics collected during this observation will be
            finalized and stored in the main Reporter instance once
            the context manager exits.
        """
        sub_reporter = self.start_epoch(key, epoch)
        yield sub_reporter
        # Receive the stats from sub_reporter
        self.finish_epoch(sub_reporter)

    def start_epoch(self, key: str, epoch: int = None) -> SubReporter:
        """
            Starts the reporting for a new epoch.

        This method initializes a new `SubReporter` for tracking statistics
        during the epoch specified by the `key` parameter. If an epoch is
        specified, it will be set as the current epoch for the `Reporter`.

        Args:
            key (str): A string identifier for the reporting context, such as
                'train' or 'eval'.
            epoch (int, optional): The epoch number to start. If not provided,
                the current epoch will be used.

        Returns:
            SubReporter: An instance of `SubReporter` for the current epoch
            reporting.

        Raises:
            ValueError: If `epoch` is less than 0.
            RuntimeError: If the previous epoch's statistics are missing or if
                the epoch is not properly incremented.

        Examples:
            >>> reporter = Reporter()
            >>> sub_reporter = reporter.start_epoch('train', 1)
            >>> sub_reporter.key
            'train'
            >>> sub_reporter.epoch
            1

        Note:
            It is essential to call `finish_epoch` after using the
            `SubReporter` to finalize and store the statistics for the epoch.

        Todo:
            Add validation for `key` to ensure it matches expected formats.
        """
        if epoch is not None:
            if epoch < 0:
                raise ValueError(f"epoch must be 0 or more: {epoch}")
            self.epoch = epoch

        if self.epoch - 1 not in self.stats or key not in self.stats[self.epoch - 1]:
            # If the previous epoch doesn't exist for some reason,
            # maybe due to bug, this case also indicates 0-count.
            if self.epoch - 1 != 0:
                warnings.warn(
                    f"The stats of the previous epoch={self.epoch - 1}"
                    f"doesn't exist."
                )
            total_count = 0
        else:
            total_count = self.stats[self.epoch - 1][key]["total_count"]

        sub_reporter = SubReporter(key, self.epoch, total_count)
        # Clear the stats for the next epoch if it exists
        self.stats.pop(epoch, None)
        return sub_reporter

    def finish_epoch(self, sub_reporter: SubReporter) -> None:
        """
            Finalizes the statistics for the current epoch and updates the report.

        This method calculates the mean values of the statistics collected
        during the epoch and stores them in the overall stats dictionary.
        It also records the elapsed time for the epoch and handles GPU
        memory metrics if applicable.

        Args:
            sub_reporter (SubReporter): The sub-reporter instance that contains
                the statistics for the current epoch.

        Raises:
            RuntimeError: If the epoch of the sub-reporter does not match the
                current epoch of the reporter.

        Examples:
            >>> reporter = Reporter()
            >>> with reporter.observe('train') as sub_reporter:
            ...     # Register statistics during training
            ...     sub_reporter.register({'loss': 0.5})
            >>> reporter.finish_epoch(sub_reporter)

        Note:
            This method is intended to be called after the observation of
            the epoch has been completed, and it ensures that the statistics
            are properly aggregated and stored.

        Todo:
            - Consider adding more metrics to track or log.
        """
        if self.epoch != sub_reporter.epoch:
            raise RuntimeError(
                f"Don't change epoch during observation: "
                f"{self.epoch} != {sub_reporter.epoch}"
            )

        # Calc mean of current stats and set it as previous epochs stats
        stats = {}
        for key2, values in sub_reporter.stats.items():
            v = aggregate(values)
            stats[key2] = v

        stats["time"] = datetime.timedelta(
            seconds=time.perf_counter() - sub_reporter.start_time
        )
        stats["total_count"] = sub_reporter.total_count
        if V(torch.__version__) >= V("1.4.0"):
            if torch.cuda.is_initialized():
                stats["gpu_max_cached_mem_GB"] = (
                    torch.cuda.max_memory_reserved() / 2**30
                )
        else:
            if torch.cuda.is_available() and torch.cuda.max_memory_cached() > 0:
                stats["gpu_cached_mem_GB"] = torch.cuda.max_memory_cached() / 2**30

        self.stats.setdefault(self.epoch, {})[sub_reporter.key] = stats
        sub_reporter.finished()

    def sort_epochs_and_values(
        self, key: str, key2: str, mode: str
    ) -> List[Tuple[int, float]]:
        """
        Return the epoch which resulted in the best value.

        This method sorts the epochs based on the specified metric and mode,
        returning the epochs along with their corresponding values.

        Args:
            key (str): The primary key to access the statistics (e.g., 'train' or 'eval').
            key2 (str): The secondary key to specify the metric (e.g., 'loss' or 'accuracy').
            mode (str): The mode for sorting; must be either 'min' or 'max'.
                        'min' returns the epochs with the smallest values,
                        while 'max' returns the epochs with the largest values.

        Returns:
            List[Tuple[int, float]]: A list of tuples, each containing an epoch
            number and its corresponding value, sorted according to the specified mode.

        Raises:
            ValueError: If the mode is not 'min' or 'max'.
            KeyError: If the specified key or key2 is not found in the statistics.

        Examples:
            >>> val = reporter.sort_epochs_and_values('eval', 'loss', 'min')
            >>> e_1best, v_1best = val[0]
            >>> e_2best, v_2best = val[1]

        Note:
            Ensure that the statistics for the specified keys have been registered
            before calling this method.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must min or max: {mode}")
        if not self.has(key, key2):
            raise KeyError(f"{key}.{key2} is not found: {self.get_all_keys()}")

        # iterate from the last epoch
        values = [(e, self.stats[e][key][key2]) for e in self.stats]

        if mode == "min":
            values = sorted(values, key=lambda x: x[1])
        else:
            values = sorted(values, key=lambda x: -x[1])
        return values

    def sort_epochs(self, key: str, key2: str, mode: str) -> List[int]:
        """
            Sort and return the epochs based on the specified metric.

        This method retrieves the epochs associated with the specified
        key and key2, and sorts them in either ascending or descending
        order based on the mode specified. The mode can be 'min' for
        finding the epochs with the minimum value of the specified metric
        or 'max' for finding the epochs with the maximum value.

        Args:
            key (str): The primary key representing the type of data (e.g.,
                'train', 'eval').
            key2 (str): The secondary key representing the specific metric
                to evaluate (e.g., 'loss', 'accuracy').
            mode (str): Specifies the sorting order. Should be either 'min'
                or 'max'.

        Returns:
            List[int]: A list of epochs sorted based on the specified
            metric.

        Raises:
            ValueError: If the mode is not 'min' or 'max'.
            KeyError: If the specified key or key2 does not exist in the
            recorded statistics.

        Examples:
            >>> reporter = Reporter()
            >>> # Assuming reporter has recorded stats for 'train' and 'loss'
            >>> sorted_epochs = reporter.sort_epochs('train', 'loss', 'min')
            >>> print(sorted_epochs)  # Output: [1, 2, 3] (example output)
        """
        return [e for e, v in self.sort_epochs_and_values(key, key2, mode)]

    def sort_values(self, key: str, key2: str, mode: str) -> List[float]:
        """
        Return a list of values sorted by the specified key and mode.

        This method retrieves values for a specified key (e.g., 'train')
        and a second key (e.g., 'loss') from the stats, sorts them
        according to the specified mode ('min' or 'max'), and returns
        the sorted values as a list.

        Args:
            key (str): The primary key for which to retrieve values.
            key2 (str): The secondary key for which to retrieve values.
            mode (str): The sorting mode, either 'min' or 'max'.

        Returns:
            List[float]: A list of values sorted according to the specified
            mode.

        Raises:
            ValueError: If mode is not 'min' or 'max'.
            KeyError: If the specified key or key2 is not found in the
            stats.

        Examples:
            >>> reporter = Reporter()
            >>> reporter.sort_values('train', 'loss', 'min')
            [0.1, 0.2, 0.3]

            >>> reporter.sort_values('eval', 'accuracy', 'max')
            [0.95, 0.93, 0.90]
        """
        return [v for e, v in self.sort_epochs_and_values(key, key2, mode)]

    def get_best_epoch(self, key: str, key2: str, mode: str, nbest: int = 0) -> int:
        """
        Retrieve the best epoch based on a specified metric.

        This method sorts the epochs based on the specified metric and
        returns the epoch corresponding to the best value. The best value
        can be either the minimum or maximum, depending on the specified
        mode. The `nbest` argument allows you to retrieve the nth best
        epoch (e.g., 0 for the best, 1 for the second best, etc.).

        Args:
            key: The main key for the metric (e.g., 'train', 'eval').
            key2: The sub-key for the specific metric (e.g., 'loss', 'acc').
            mode: The mode for selecting the best value; it should be either
                'min' or 'max'.
            nbest: The index of the best epoch to retrieve. Defaults to 0,
                which retrieves the best epoch.

        Returns:
            The epoch number that corresponds to the best value based on the
            specified criteria.

        Examples:
            >>> best_epoch = reporter.get_best_epoch('eval', 'loss', 'min')
            >>> second_best_epoch = reporter.get_best_epoch('train', 'acc', 'max', nbest=1)

        Raises:
            ValueError: If `mode` is not 'min' or 'max'.
            KeyError: If the specified key or sub-key does not exist in the
                reported statistics.
        """
        return self.sort_epochs(key, key2, mode)[nbest]

    def check_early_stopping(
        self,
        patience: int,
        key1: str,
        key2: str,
        mode: str,
        epoch: int = None,
        logger=None,
    ) -> bool:
        if logger is None:
            logger = logging
        if epoch is None:
            epoch = self.get_epoch()

        best_epoch = self.get_best_epoch(key1, key2, mode)
        if epoch - best_epoch > patience:
            logger.info(
                f"[Early stopping] {key1}.{key2} has not been "
                f"improved {epoch - best_epoch} epochs continuously. "
                f"The training was stopped at {epoch}epoch"
            )
            return True
        else:
            return False

    def has(self, key: str, key2: str, epoch: int = None) -> bool:
        """
            Check if the specified key and key2 exist in the stats for a given epoch.

        This method verifies whether the specified statistics are recorded for the
        current epoch or a specific epoch. It checks if both the primary key and
        the secondary key exist in the stats dictionary.

        Args:
            key (str): The primary key, typically representing a phase (e.g., 'train').
            key2 (str): The secondary key, usually representing a specific metric
                (e.g., 'loss').
            epoch (int, optional): The epoch to check. If None, the current epoch
                is used. Defaults to None.

        Returns:
            bool: True if the keys exist in the stats for the specified epoch,
                otherwise False.

        Examples:
            >>> reporter = Reporter()
            >>> reporter.has('train', 'loss')  # Check for the current epoch
            False
            >>> with reporter.observe('train') as sub_reporter:
            ...     sub_reporter.register({'loss': 0.2})
            >>> reporter.has('train', 'loss')  # Check after registration
            True

        Raises:
            KeyError: If the specified keys do not exist in the stats for the epoch.
        """
        if epoch is None:
            epoch = self.get_epoch()
        return (
            epoch in self.stats
            and key in self.stats[epoch]
            and key2 in self.stats[epoch][key]
        )

    def log_message(self, epoch: int = None) -> str:
        """
        Generate a formatted log message for the current epoch.

        This method constructs a string message that summarizes the
        reported values for the specified range of batches within the
        current epoch. The message includes statistics for each key
        that has been registered during the epoch.

        Args:
            start (int, optional): The starting index of the batch range.
                If None, defaults to 0. If negative, counts from the end.
            end (int, optional): The ending index of the batch range.
                If None, defaults to the current count of batches.

        Returns:
            str: A formatted log message summarizing the statistics
                for the specified range of batches.

        Raises:
            RuntimeError: If the reporting process has already been
                finished.

        Examples:
            >>> sub_reporter = SubReporter("train", 1, 10)
            >>> sub_reporter.register({"loss": 0.5})
            >>> print(sub_reporter.log_message(0, 1))
            "1epoch:train:1-1batch: loss=0.500"

            >>> sub_reporter.register({"loss": 0.3})
            >>> print(sub_reporter.log_message(0, 2))
            "1epoch:train:1-2batch: loss=0.400"

        Note:
            The start and end indices must be within the bounds of the
            registered statistics.
        """
        if epoch is None:
            epoch = self.get_epoch()

        message = ""
        for key, d in self.stats[epoch].items():
            _message = ""
            for key2, v in d.items():
                if v is not None:
                    if len(_message) != 0:
                        _message += ", "
                    if isinstance(v, float):
                        if abs(v) > 1.0e3:
                            _message += f"{key2}={v:.3e}"
                        elif abs(v) > 1.0e-3:
                            _message += f"{key2}={v:.3f}"
                        else:
                            _message += f"{key2}={v:.3e}"
                    elif isinstance(v, datetime.timedelta):
                        _v = humanfriendly.format_timespan(v)
                        _message += f"{key2}={_v}"
                    else:
                        _message += f"{key2}={v}"
            if len(_message) != 0:
                if len(message) == 0:
                    message += f"{epoch}epoch results: "
                else:
                    message += ", "
                message += f"[{key}] {_message}"
        return message

    def get_value(self, key: str, key2: str, epoch: int = None):
        """
                Retrieves the value associated with a specified key and key2 for a given epoch.

        Args:
            key (str): The primary key representing a specific metric category,
                such as 'train' or 'eval'.
            key2 (str): The secondary key representing a specific metric within
                the category, such as 'loss' or 'accuracy'.
            epoch (int, optional): The epoch number to retrieve the value from. If
                not provided, the current epoch is used.

        Returns:
            The value associated with the specified key and key2 for the given
            epoch.

        Raises:
            KeyError: If the specified key or key2 does not exist in the
                statistics.

        Examples:
            >>> reporter = Reporter()
            >>> reporter.register({'loss': 0.2})
            >>> value = reporter.get_value('train', 'loss')
            >>> print(value)  # Output: 0.2

        Note:
            Ensure that the statistics for the specified key and key2 have been
            registered before attempting to retrieve the value.
        """
        if not self.has(key, key2):
            raise KeyError(f"{key}.{key2} is not found in stats: {self.get_all_keys()}")
        if epoch is None:
            epoch = self.get_epoch()
        return self.stats[epoch][key][key2]

    def get_keys(self, epoch: int = None) -> Tuple[str, ...]:
        """
        Returns keys1 e.g. train, eval.

        Args:
            epoch (int, optional): The epoch number for which to retrieve the keys.
                If None, the current epoch is used.

        Returns:
            Tuple[str, ...]: A tuple of keys representing the first-level
                keys (e.g., 'train', 'eval') in the stats dictionary for
                the specified epoch.

        Examples:
            >>> reporter = Reporter()
            >>> reporter.start_epoch('train')
            >>> reporter.register({'loss': 0.2})
            >>> reporter.get_keys()
            ('train',)

            >>> reporter.start_epoch('eval')
            >>> reporter.register({'loss': 0.1})
            >>> reporter.get_keys()
            ('train', 'eval')
        """
        if epoch is None:
            epoch = self.get_epoch()
        return tuple(self.stats[epoch])

    def get_keys2(self, key: str, epoch: int = None) -> Tuple[str, ...]:
        """
            Returns keys2 e.g. loss, acc.

        This method retrieves the second-level keys associated with a specified
        first-level key (e.g., 'train' or 'eval') for a given epoch. The keys
        are filtered to exclude reserved keys like 'time' and 'total_count'.

        Args:
            key (str): The first-level key to retrieve second-level keys from.
            epoch (int, optional): The epoch number to retrieve keys for. If None,
                the current epoch is used.

        Returns:
            Tuple[str, ...]: A tuple of second-level keys corresponding to the
            specified first-level key for the specified epoch.

        Examples:
            >>> reporter = Reporter()
            >>> with reporter.observe('train') as sub_reporter:
            ...     sub_reporter.register({'loss': 0.2, 'acc': 0.8})
            >>> reporter.get_keys2('train')
            ('loss', 'acc')

        Raises:
            KeyError: If the specified key does not exist in the stats for the
            specified epoch.
        """
        if epoch is None:
            epoch = self.get_epoch()
        d = self.stats[epoch][key]
        keys2 = tuple(k for k in d if k not in ("time", "total_count"))
        return keys2

    def get_all_keys(self, epoch: int = None) -> Tuple[Tuple[str, str], ...]:
        """
            Returns all keys from the current epoch's statistics.

        The returned keys are tuples containing two elements:
        the first element corresponds to the main key (e.g., 'train', 'eval'),
        and the second element corresponds to the specific metric (e.g., 'loss', 'acc').

        Args:
            epoch (int, optional): The epoch from which to retrieve the keys. If None,
                the current epoch is used. Defaults to None.

        Returns:
            Tuple[Tuple[str, str], ...]: A tuple of tuples containing all key pairs
            from the specified epoch.

        Examples:
            >>> reporter = Reporter()
            >>> with reporter.observe('train') as sub_reporter:
            ...     sub_reporter.register({'loss': 0.2})
            ...     sub_reporter.next()
            >>> reporter.get_all_keys()
            (('train', 'loss'),)

        Note:
            If the specified epoch has no recorded statistics, an empty tuple will be
            returned.
        """
        if epoch is None:
            epoch = self.get_epoch()
        all_keys = []
        for key in self.stats[epoch]:
            for key2 in self.stats[epoch][key]:
                all_keys.append((key, key2))
        return tuple(all_keys)

    def matplotlib_plot(self, output_dir: Union[str, Path]):
        """
                Plot stats using Matplotlib and save images.

        This method generates plots for each metric collected during training or
        evaluation and saves them as PNG images in the specified output directory.
        The images are named according to the metric (key2) being plotted.

        Args:
            output_dir (Union[str, Path]): The directory where the plots will be saved.

        Examples:
            >>> reporter = Reporter()
            >>> # Assume stats have been registered in reporter
            >>> reporter.matplotlib_plot('output/plots')

        Note:
            Ensure that the Matplotlib library is properly installed and available
            in the environment where this function is executed.
        """
        keys2 = set.union(*[set(self.get_keys2(k)) for k in self.get_keys()])
        for key2 in keys2:
            keys = [k for k in self.get_keys() if key2 in self.get_keys2(k)]
            plt = self._plot_stats(keys, key2)
            p = output_dir / f"{key2}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p)

    @typechecked
    def _plot_stats(self, keys: Sequence[str], key2: str):
        # str is also Sequence[str]
        if isinstance(keys, str):
            raise TypeError(f"Input as [{keys}]")

        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        plt.clf()

        epochs = np.arange(1, self.get_epoch() + 1)
        for key in keys:
            y = [
                (
                    self.stats[e][key][key2]
                    if e in self.stats
                    and key in self.stats[e]
                    and key2 in self.stats[e][key]
                    else np.nan
                )
                for e in epochs
            ]
            assert len(epochs) == len(y), "Bug?"

            plt.plot(epochs, y, label=key, marker="x")
        plt.legend()
        plt.title(f"{key2} vs epoch")
        # Force integer tick for x-axis
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel("epoch")
        plt.ylabel(key2)
        plt.grid()

        return plt

    def tensorboard_add_scalar(
        self, summary_writer, epoch: int = None, key1: Optional[str] = None
    ):
        """
            Adds scalar values to TensorBoard for visualization.

        This method logs scalar statistics to the provided TensorBoard
        summary writer. The statistics are recorded for the current epoch
        or a specified epoch.

        Args:
            summary_writer: A TensorBoard summary writer instance used to log
                scalar values.
            start (int, optional): The starting index from which to log the
                statistics. If None, it defaults to 0. If negative, it will
                start from `self.count + start`.

        Examples:
            >>> from torch.utils.tensorboard import SummaryWriter
            >>> writer = SummaryWriter()
            >>> reporter = Reporter()
            >>> with reporter.observe('train') as sub_reporter:
            ...     for batch in iterator:
            ...         stats = dict(loss=0.2)
            ...         sub_reporter.register(stats)
            ...     sub_reporter.tensorboard_add_scalar(writer)
            >>> writer.close()

        Raises:
            AssertionError: If the length of the statistics list does not match
                the expected count.
        """
        if epoch is None:
            epoch = self.get_epoch()
            total_count = self.stats[epoch]["train"]["total_count"]
            if key1 == "train":
                summary_writer.add_scalar("iter_epoch", epoch, total_count)

        if key1 is not None:
            key1_iterator = tuple([key1])
        else:
            key1_iterator = self.get_keys(epoch)

        for key1 in key1_iterator:
            for key2 in self.get_keys2(key1):
                summary_writer.add_scalar(
                    f"{key2}", self.stats[epoch][key1][key2], total_count
                )

    def wandb_log(self, epoch: int = None):
        """
            Logs the reported values to Weights and Biases (wandb).

        This method aggregates the statistics collected in the current epoch and logs
        them to the Weights and Biases dashboard. The logged values include metrics
        prefixed with their respective categories (e.g., 'train/', 'valid/') and the
        total count of iterations.

        Args:
            start (int, optional): The starting index for the statistics to log.
                If None, logging will start from the beginning. If negative, it will
                start from the end of the statistics.

        Raises:
            AssertionError: If the lengths of the statistics lists do not match the
                current count of reported values.

        Examples:
            >>> reporter = Reporter()
            >>> with reporter.observe('train') as sub_reporter:
            ...     sub_reporter.register({'loss': 0.5})
            ...     sub_reporter.wandb_log()

        Note:
            Ensure that wandb is properly initialized before calling this method.
        """
        import wandb

        if epoch is None:
            epoch = self.get_epoch()

        d = {}
        for key1 in self.get_keys(epoch):
            for key2 in self.stats[epoch][key1]:
                if key2 in ("time", "total_count"):
                    continue
                key = f"{key1}_{key2}_epoch"
                d[wandb_get_prefix(key) + key] = self.stats[epoch][key1][key2]
        d["epoch"] = epoch
        wandb.log(d)

    def state_dict(self):
        """
                Returns the current state of the Reporter instance.

        The state is represented as a dictionary containing the following keys:
        - 'stats': A dictionary of recorded statistics for each epoch and key.
        - 'epoch': The current epoch number.

        This method is useful for saving the current state of the Reporter, allowing
        for resuming from a checkpoint later.

        Examples:
            >>> reporter = Reporter()
            >>> # Simulate some training and logging
            >>> with reporter.observe('train') as sub_reporter:
            ...     sub_reporter.register({'loss': 0.5})
            >>> state = reporter.state_dict()
            >>> print(state['epoch'])  # Output: 0
            >>> print(state['stats'])  # Output: {'train': {'loss': 0.5, ...}}

        Returns:
            dict: A dictionary containing the state of the Reporter.
        """
        return {"stats": self.stats, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        """
                Loads the state dictionary into the Reporter instance.

        This method updates the internal state of the Reporter instance with the
        provided state dictionary. It expects the state dictionary to contain the
        epoch and stats of the Reporter. This is typically used for restoring the
        state of a model during training or evaluation.

        Args:
            state_dict (dict): A dictionary containing the state of the Reporter.
                It should have the following keys:
                    - "epoch" (int): The current epoch of the Reporter.
                    - "stats" (dict): The statistics collected during training,
                      indexed by epoch and other relevant keys.

        Examples:
            >>> reporter = Reporter()
            >>> state = {"epoch": 5, "stats": {0: {"train": {"loss": 0.1}}}}
            >>> reporter.load_state_dict(state)
            >>> print(reporter.get_epoch())  # Output: 5
            >>> print(reporter.stats)  # Output: {0: {'train': {'loss': 0.1}}}
        """
        self.epoch = state_dict["epoch"]
        self.stats = state_dict["stats"]
