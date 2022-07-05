import argparse
import unittest.mock
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor

import pytest

from espnet2.tasks.abs_task import AbsTask
from espnet2.train.distributed_utils import (
    DistributedOption,
    free_port,
    resolve_distributed_mode,
)
from espnet2.utils.build_dataclass import build_dataclass


@pytest.fixture()
def dist_init_method(tmp_path):
    return f"file://{tmp_path}/init"


def _init(option):
    option.init_options()
    option.init_torch_distributed()


def test_default_work():
    parser = AbsTask.get_parser()
    args = parser.parse_args([])
    resolve_distributed_mode(args)
    option = build_dataclass(DistributedOption, args)
    option.init_options()
    option.init_torch_distributed()


def test_resolve_distributed_mode1(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=False,
        dist_world_size=2,
        dist_rank=None,
        ngpu=2,
        local_rank=0,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    with pytest.raises(RuntimeError):
        resolve_distributed_mode(args)


def test_resolve_distributed_mode2(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=False,
        dist_world_size=2,
        dist_rank=0,
        ngpu=2,
        local_rank=None,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    with pytest.raises(RuntimeError):
        resolve_distributed_mode(args)


def test_resolve_distributed_mode3(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=False,
        dist_world_size=None,
        dist_rank=None,
        ngpu=2,
        local_rank=None,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    resolve_distributed_mode(args)


def test_resolve_distributed_mode4(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=False,
        dist_world_size=2,
        dist_rank=0,
        ngpu=2,
        local_rank=1,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    resolve_distributed_mode(args)
    assert args.distributed


def test_resolve_distributed_mode5(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=False,
        dist_world_size=2,
        dist_rank=0,
        ngpu=2,
        local_rank=1,
        dist_launcher="slurm",
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    with pytest.raises(RuntimeError):
        resolve_distributed_mode(args)


def test_resolve_distributed_mode6(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=2,
        dist_rank=None,
        ngpu=1,
        local_rank=None,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    with pytest.raises(RuntimeError):
        resolve_distributed_mode(args)


def test_resolve_distributed_mode7(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=2,
        dist_rank=0,
        ngpu=1,
        local_rank=None,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    resolve_distributed_mode(args)
    assert args.distributed
    assert not args.multiprocessing_distributed


def test_resolve_distributed_mode9(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=1,
        dist_rank=None,
        ngpu=2,
        local_rank=None,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    resolve_distributed_mode(args)
    assert args.distributed
    assert args.multiprocessing_distributed


def test_resolve_distributed_mode10(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=None,
        dist_rank=None,
        ngpu=1,
        local_rank=None,
        dist_launcher=None,
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    resolve_distributed_mode(args)
    assert not args.distributed
    assert not args.multiprocessing_distributed


@pytest.mark.skipif(True, reason="sometimes hangup?")
def test_init_cpu(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=2,
        dist_rank=None,
        ngpu=0,
        local_rank=None,
        dist_launcher=None,
        distributed=True,
        dist_backend="gloo",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    args.dist_rank = 0
    option = build_dataclass(DistributedOption, args)
    args.dist_rank = 1
    option2 = build_dataclass(DistributedOption, args)
    with ProcessPoolExecutor(max_workers=2) as e:
        fn = e.submit(_init, option)
        fn2 = e.submit(_init, option2)
        fn.result()
        fn2.result()


@pytest.mark.skipif(True, reason="sometimes hangup?")
def test_init_cpu2():
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=2,
        dist_rank=None,
        ngpu=0,
        local_rank=None,
        dist_launcher=None,
        distributed=True,
        dist_backend="gloo",
        dist_init_method="env://",
        dist_master_addr=None,
        dist_master_port=free_port(),
    )
    args.dist_rank = 0
    option = build_dataclass(DistributedOption, args)
    args.dist_rank = 1
    option2 = build_dataclass(DistributedOption, args)
    with ProcessPoolExecutor(max_workers=2) as e:
        fn = e.submit(_init, option)
        fn2 = e.submit(_init, option2)
        with pytest.raises(RuntimeError):
            fn.result()
            fn2.result()


@pytest.mark.skipif(True, reason="sometimes hangup?")
def test_init_cpu3():
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=2,
        dist_rank=None,
        ngpu=0,
        local_rank=None,
        dist_launcher=None,
        distributed=True,
        dist_backend="gloo",
        dist_init_method="env://",
        dist_master_addr="localhost",
        dist_master_port=None,
    )
    args.dist_rank = 0
    option = build_dataclass(DistributedOption, args)
    args.dist_rank = 1
    option2 = build_dataclass(DistributedOption, args)
    with ThreadPoolExecutor(max_workers=2) as e:
        fn = e.submit(_init, option)
        fn2 = e.submit(_init, option2)
        with pytest.raises(RuntimeError):
            fn.result()
            fn2.result()


@pytest.mark.skipif(True, reason="sometimes hangup?")
def test_init_cpu4():
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=2,
        dist_rank=None,
        ngpu=0,
        local_rank=None,
        dist_launcher=None,
        distributed=True,
        dist_backend="gloo",
        dist_init_method="env://",
        dist_master_addr="localhost",
        dist_master_port=free_port(),
    )
    args.dist_rank = 0
    option = build_dataclass(DistributedOption, args)
    args.dist_rank = 1
    option2 = build_dataclass(DistributedOption, args)
    with ProcessPoolExecutor(max_workers=2) as e:
        fn = e.submit(_init, option)
        fn2 = e.submit(_init, option2)
        fn.result()
        fn2.result()


@pytest.mark.skipif(True, reason="sometimes hangup?")
def test_init_cpu5():
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=2,
        dist_rank=None,
        ngpu=0,
        local_rank=None,
        dist_launcher=None,
        distributed=True,
        dist_backend="gloo",
        dist_init_method="env://",
        dist_master_addr="localhost",
        dist_master_port=free_port(),
    )
    args.dist_rank = 0
    option = build_dataclass(DistributedOption, args)
    args.dist_rank = 1
    option2 = build_dataclass(DistributedOption, args)
    with ProcessPoolExecutor(max_workers=2) as e:
        fn = e.submit(_init, option)
        fn2 = e.submit(_init, option2)
        fn.result()
        fn2.result()


def test_resolve_distributed_mode_slurm1(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=False,
        dist_world_size=None,
        dist_rank=None,
        ngpu=2,
        local_rank=None,
        dist_launcher="slurm",
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    with unittest.mock.patch.dict(
        "os.environ",
        dict(
            SLURM_PROCID="0",
            SLURM_NTASKS="2",
            SLURM_STEP_NUM_NODES="2",
            SLURM_STEP_NODELIST="host1",
            SLURM_NODEID="0",
            SLURM_LOCALID="0",
            CUDA_VISIBLE_DEVICES="0,1",
        ),
    ):
        resolve_distributed_mode(args)


def test_resolve_distributed_mode_slurm2(dist_init_method):
    args = argparse.Namespace(
        multiprocessing_distributed=False,
        dist_world_size=None,
        dist_rank=None,
        ngpu=2,
        local_rank=None,
        dist_launcher="slurm",
        dist_backend="nccl",
        dist_init_method=dist_init_method,
        dist_master_addr=None,
        dist_master_port=None,
    )
    with unittest.mock.patch.dict(
        "os.environ",
        dict(
            SLURM_PROCID="0",
            SLURM_NTASKS="2",
            SLURM_STEP_NUM_NODES="1",
            SLURM_STEP_NODELIST="host1",
            SLURM_NODEID="0",
            SLURM_LOCALID="0",
            CUDA_VISIBLE_DEVICES="0,1",
        ),
    ):
        with pytest.raises(RuntimeError):
            resolve_distributed_mode(args)


def test_resolve_distributed_mode_slurm3():
    args = argparse.Namespace(
        multiprocessing_distributed=True,
        dist_world_size=None,
        dist_rank=None,
        ngpu=1,
        local_rank=None,
        dist_launcher="slurm",
        dist_backend="nccl",
        dist_init_method="env://",
        dist_master_addr=None,
        dist_master_port=10000,
    )
    env = dict(
        SLURM_PROCID="0",
        SLURM_NTASKS="1",
        SLURM_STEP_NUM_NODES="1",
        SLURM_STEP_NODELIST="localhost",
        SLURM_NODEID="0",
        CUDA_VISIBLE_DEVICES="0,1",
    )

    e = ProcessPoolExecutor(max_workers=2)
    with unittest.mock.patch.dict("os.environ", dict(env, SLURM_LOCALID="0")):
        resolve_distributed_mode(args)
        option = build_dataclass(DistributedOption, args)
        fn = e.submit(_init, option)

    with unittest.mock.patch.dict("os.environ", dict(env, SLURM_LOCALID="0")):
        option2 = build_dataclass(DistributedOption, args)
        fn2 = e.submit(_init, option2)

    fn.result()
    fn2.result()
