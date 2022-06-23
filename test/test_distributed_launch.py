# coding: utf-8
#
# SPDX-FileCopyrightText:
#   Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


import argparse
import itertools
import os
import sys
from multiprocessing import Queue

import pytest

from espnet.distributed.pytorch_backend.launch import WorkerError, launch


@pytest.mark.parametrize("nprocs", [1, 2])
@pytest.mark.execution_timeout(4.0)
def test_simple_function_ok(nprocs):
    args = None

    def simple_func(args):
        # NOP
        return 0

    launch(simple_func, args, nprocs)


@pytest.mark.parametrize("nprocs", [1, 2])
@pytest.mark.execution_timeout(4.0)
def test_simple_function_ok_with_args(nprocs):
    args = argparse.Namespace(**{f"param{v}": v for v in range(10)})

    def simple_func(args):
        args_dict = vars(args)
        for v in range(10):
            key = f"param{v}"
            assert key in args_dict
            assert args_dict[key] == v
        return 0

    launch(simple_func, args, nprocs)


@pytest.mark.parametrize("nprocs", [1, 2])
@pytest.mark.execution_timeout(4.0)
def test_simple_function_ok_with_right_envvar(nprocs):
    queue = Queue()

    def simple_func(queue):
        worldsize = os.environ.get("WORLD_SIZE", None)
        rank = os.environ.get("RANK", None)
        localrank = os.environ.get("LOCAL_RANK", None)
        assert worldsize is not None
        assert rank is not None
        assert localrank is not None
        queue.put(
            {
                "worldsize": int(worldsize),
                "rank": int(rank),
                "localrank": int(localrank),
            }
        )
        return 0

    launch(simple_func, queue, nprocs)

    results = [queue.get() for _ in range(nprocs)]
    pids = set(range(nprocs))
    for r in results:
        worldsize = r["worldsize"]
        rank = r["rank"]
        localrank = r["localrank"]
        assert worldsize == nprocs
        assert rank in pids
        assert localrank in pids
        pids.remove(rank)
    assert len(pids) == 0
    assert queue.empty()


@pytest.mark.parametrize(
    "nprocs, exitcode",
    [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    ],
)
@pytest.mark.execution_timeout(4.0)
def test_worker_exits_nonzero_code_ng(nprocs, exitcode):
    for combination in itertools.product(range(2), repeat=nprocs):
        n_activated = sum(combination)
        if n_activated != 1 and n_activated != nprocs:
            # skip.
            continue
        if n_activated == 1:
            exit_idx = combination.index(1)
        else:
            exit_idx = None
        args = None

        def simple_func(args):
            # NOP
            rank = os.environ.get("RANK", None)
            assert rank is not None
            rank = int(rank)
            if n_activated == 1 and rank != exit_idx:
                return
            sys.exit(exitcode)

        with pytest.raises(WorkerError) as excinfo:
            launch(simple_func, args, nprocs)
        assert excinfo.value.exitcode == exitcode
        if n_activated == 1:
            assert excinfo.value.worker_id == exit_idx


@pytest.mark.parametrize("nprocs", [1, 2])
@pytest.mark.execution_timeout(4.0)
def test_worker_raises_exception_ng(nprocs):
    for combination in itertools.product(range(2), repeat=nprocs):
        n_activated = sum(combination)
        if n_activated != 1 and n_activated != nprocs:
            # skip.
            continue
        if n_activated == 1:
            exit_idx = combination.index(1)
        else:
            exit_idx = None
        args = None

        def simple_func(args):
            # NOP
            rank = os.environ.get("RANK", None)
            assert rank is not None
            rank = int(rank)
            if n_activated == 1 and rank != exit_idx:
                return
            raise RuntimeError("error")

        with pytest.raises(WorkerError) as excinfo:
            launch(simple_func, args, nprocs)
        assert excinfo.value.exitcode == 1
        if n_activated == 1:
            assert excinfo.value.worker_id == exit_idx
