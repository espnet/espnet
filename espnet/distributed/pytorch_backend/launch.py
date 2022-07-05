#
# SPDX-FileCopyrightText:
#   Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""This is a helper module for distributed training.

The code uses an official implementation of
distributed data parallel launcher as just a reference.
https://github.com/pytorch/pytorch/blob/v1.8.2/torch/distributed/launch.py
One main difference is this code focuses on
launching simple function with given arguments.
"""

import multiprocessing
import os
import signal
import socket
import time

if hasattr(signal, "valid_signals"):
    _signalno_name_map = {
        s.value: s.name for s in signal.valid_signals() if isinstance(s, signal.Signals)
    }
else:
    # TODO(lazykyama): It should be deprecated
    # once Python 3.7 is removed from supported platform.
    _signalno_name_map = dict(
        [
            (1, "SIGHUP"),
            (2, "SIGINT"),
            (3, "SIGQUIT"),
            (4, "SIGILL"),
            (5, "SIGTRAP"),
            (6, "SIGABRT"),
            (7, "SIGBUS"),
            (8, "SIGFPE"),
            (9, "SIGKILL"),
            (10, "SIGUSR1"),
            (11, "SIGSEGV"),
            (12, "SIGUSR2"),
            (13, "SIGPIPE"),
            (14, "SIGALRM"),
            (15, "SIGTERM"),
            (17, "SIGCHLD"),
            (18, "SIGCONT"),
            (19, "SIGSTOP"),
            (20, "SIGTSTP"),
            (21, "SIGTTIN"),
            (22, "SIGTTOU"),
            (23, "SIGURG"),
            (24, "SIGXCPU"),
            (25, "SIGXFSZ"),
            (26, "SIGVTALRM"),
            (27, "SIGPROF"),
            (28, "SIGWINCH"),
            (29, "SIGIO"),
            (30, "SIGPWR"),
            (31, "SIGSYS"),
            (34, "SIGRTMIN"),
            (64, "SIGRTMAX"),
        ]
    )


class WorkerError(multiprocessing.ProcessError):
    """An error happened within each worker."""

    def __init__(self, *, msg, exitcode, worker_id):
        """Initialize error class."""
        super(WorkerError, self).__init__(msg)
        self._exitcode = exitcode
        self._worker_id = worker_id

    def __str__(self):
        """Construct and return a special error message."""
        return f"worker[{self._worker_id}] failed with exitcode={self._exitcode}"

    @property
    def exitcode(self):
        """Return exitcode from worker process."""
        return self._exitcode

    @property
    def worker_id(self):
        """Return worker ID related to a process causes this error."""
        return self._worker_id


class MainProcessError(multiprocessing.ProcessError):
    """An error happened from main process."""

    def __init__(self, *, signal_no):
        """Initialize error class."""
        msg = (
            f"{_signalno_name_map[signal_no]} received, "
            f"exiting due to {signal.strsignal(signal_no)}."
        )
        super(MainProcessError, self).__init__(msg)
        self._signal_no = signal_no
        self._msg = msg

    def __str__(self):
        """Return a custom error message."""
        return self._msg

    @property
    def signal_no(self):
        """Return signal number which stops main process."""
        return self._signal_no


def set_start_method(method):
    """Set multiprocess start method."""
    assert method in ("fork", "spawn", "forkserver")
    return multiprocessing.set_start_method(method)


def free_port():
    """Find free port using bind().

    There are some interval between finding this port and using it
    and the other process might catch the port by that time.
    Thus it is not guaranteed that the port is really empty.

    """
    # This method is copied from ESPnet v2's utility below.
    # https://github.com/espnet/espnet/blob/43ce0c69fb32961235534b348700dc6c74ad5792/espnet2/train/distributed_utils.py#L187-L198
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _kill_processes(processes):
    # TODO(lazykyama): This implementation can't stop all processes
    # which have grandchildren processes launched
    # within each child process directly forked from this script.
    # Need improvement for more safe termination.
    for p in processes:
        try:
            # NOTE: multiprocessing.Process.kill() was introduced in 3.7.
            # https://docs.python.org/3.7/library/multiprocessing.html#multiprocessing.Process.kill
            if not hasattr(p, "kill"):
                p.terminate()
            else:
                p.kill()
        except Exception:  # noqa: E722
            # NOTE: Ignore any exception happens during killing a process
            # because this intends to send kill signal to *all* processes.
            pass


def launch(func, args, nprocs, master_addr="localhost", master_port=None):
    """Launch processes with a given function and given arguments.

    .. note:: Current implementaiton supports only single node case.

    """
    if master_port is None:
        master_port = free_port()

    # Set PyTorch distributed related environmental variables
    # NOTE: in contrast to subprocess.Popen,
    # explicit environment variables can not be specified.
    # It's necessary to add additional variables to
    # current environment variable list.
    original_env = os.environ.copy()
    # TODO(lazykyama): multi-node support
    os.environ["WORLD_SIZE"] = str(nprocs)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    processes = []
    for local_rank in range(nprocs):
        # Each process's rank
        # TODO(lazykyama): multi-node support
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

        process = multiprocessing.Process(target=func, args=(args,))
        process.start()
        processes.append(process)

    # Set signal handler to capture signals sent to main process,
    # and ensure that all children processes will be terminated.
    def _handler(signal_no, _):
        _kill_processes(processes)
        raise MainProcessError(signal_no=signal_no)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # Recovery environment variables.
    os.environ.clear()
    os.environ.update(original_env)

    # Monitor all workers.
    worker_error = None
    finished_process_ids = set()
    while len(processes) > len(finished_process_ids):
        for localrank, p in enumerate(processes):
            if p.pid in finished_process_ids:
                # Skip rest of checks becuase
                # this process has been already finished.
                continue

            if p.is_alive():
                # This process is still running.
                continue
            elif p.exitcode == 0:
                # This process properly finished.
                finished_process_ids.add(p.pid)
            else:
                # An error happens in one process.
                # Will try to terminate all other processes.
                worker_error = WorkerError(
                    msg=(f"{func.__name__} failed with error code: {p.exitcode}"),
                    exitcode=p.exitcode,
                    worker_id=localrank,
                )
                break
        if worker_error is not None:
            # Go out of this while loop to terminate all processes.
            break
        time.sleep(1.0)

    if worker_error is not None:
        # Trying to stop all workers.
        _kill_processes(processes)
        raise worker_error
