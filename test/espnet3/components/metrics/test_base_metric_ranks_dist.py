"""Two real gloo processes gather metric states and agree with one run."""

import os
import socket
import sys
from datetime import timedelta
from test.espnet3.components.metrics.test_base_metric_ranks import (
    ROWS,
    Counts,
    Scores,
    one_shot,
)

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world_size, port, rows_per_rank, queue):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        start = sum(rows_per_rank[:rank])
        mine = ROWS[start : start + rows_per_rank[rank]]
        results = {}
        for metric_cls in (Counts, Scores):
            metric = metric_cls()
            if mine:
                metric.update(ref=mine)
            gathered = [None] * world_size
            dist.all_gather_object(gathered, metric.state())
            metric.load_state(metric.merge_states(gathered))
            results[metric_cls.__name__] = (metric.compute(), metric.state())
        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("rows_per_rank", [(5, 5), (9, 1), (10, 0)])
def test_two_gloo_ranks_equal_one_run(rows_per_rank):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, 2, port, rows_per_rank, queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    try:
        got = dict(queue.get(timeout=60) for _ in procs)
        for proc in procs:
            proc.join(10)
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
    for metric_cls in (Counts, Scores):
        expected = one_shot(metric_cls).compute()
        name = metric_cls.__name__
        assert got[0][name][0] == got[1][name][0] == expected
        assert got[0][name][1] == got[1][name][1]
