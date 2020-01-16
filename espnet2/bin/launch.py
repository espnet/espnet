#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import subprocess
import shutil
import shlex
import uuid
from pathlib import Path

from espnet.utils.cli_utils import get_commandline_args
from espnet2.utils.types import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        description="Launch distributed process with appropriate options. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cmd",
        help="The path of cmd script of Kaldi: run.pl. queue.pl, or slurm.pl",
        default="utils/run.pl",
    )
    parser.add_argument(
        "--log", help="The path of log file used by cmd", default="run.log",
    )
    parser.add_argument(
        "--ngpu", type=int, default=1, help="The number of GPUs per node"
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="The number of nodes")
    parser.add_argument(
        "--multiprocessing_distributed",
        type=str2bool,
        default=True,
        help="Distributed method is used when single-node mode.",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Specify the port number of master"
        "Master is a host machine has RANK0 process.",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=None,
        help="Specify the address s of master. "
        "Master is a host machine has RANK0 process.",
    )
    parser.add_argument(
        "--init_file_prefix",
        type=str,
        default=f"{os.getcwd()}/.dist_init_",
        help="The file name prefix for init_file, which is used for "
        "'Shared-file system initialization'. "
        "This option is used when --port is not specified",
    )
    parser.add_argument("args", type=str, nargs="+")
    return parser


def main(cmd=None):
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = get_parser()
    args = parser.parse_args(cmd)
    args.cmd = shlex.split(args.cmd)

    if shutil.which(args.cmd[0]) is None:
        raise RuntimeError(
            f"The first args of --cmd should be a script path. e.g. utils/run.pl: "
            f"{args.cmd[0]}"
        )

    # Specify init_method:
    #   See: https://pytorch.org/docs/stable/distributed.html#initialization
    if args.num_nodes <= 1:
        # Automatically set init_method if num_node=1
        init_method = None
    else:
        if args.master_port is None:
            # Try "shared-file system initialization" if master_port is not specified
            # Give random name to avoid reusing previous file
            init_file = args.init_file_prefix + str(uuid.uuid4())
            Path(init_file).parent.mkdir(exist_ok=True, parents=True)
            init_method = ["--dist_init_method", f"file://{init_file}"]
        else:
            init_method = ["--dist_master_port", str(args.master_port)]

        # This can be omitted if slurm mode
        if args.master_addr is not None:
            init_method = ["--dist_master_addr", str(args.master_addf)]

    processes = []
    if args.num_nodes <= 1:
        if args.ngpu > 1:
            if args.multiprocessing_distributed:
                # NOTE:
                #   If multiprocessing_distributed=true,
                # -> Distributed mode, which is multi-process and Multi-GPUs.
                #    and TCP initializetion is used if single-node case:
                #      e.g. init_method="tcp://localhost:20000"
                logging.info(f"single-node with ngpu={args.ngpu} on distributed mode")
            else:
                # NOTE:
                #   If multiprocessing_distributed=false
                # -> "DataParallel" mode, which is single-process
                #    and Multi-GPUs with threading.
                # See:
                # https://discuss.pytorch.org/t/why-torch-nn-parallel-distributeddataparallel-runs-faster-than-torch-nn-dataparallel-on-single-machine-with-multi-gpu/32977/2
                logging.info(f"single-node with ngpu={args.ngpu} using DataParallel")

        # Using cmd as it is simply
        cmd = (
            args.cmd
            # arguments for ${cmd}
            + ["--gpu", str(args.ngpu), args.log]
            # arguments for *_train.py
            + args.args
            + [
                "--ngpu",
                str(args.ngpu),
                "--multiprocessing_distributed",
                str(args.multiprocessing_distributed),
            ]
        )
        process = subprocess.Popen(cmd)
        processes.append(process)
        logfile = args.log

    # If Slurm
    elif Path(args.cmd[0]).name == "slurm.pl":
        # WARNING:
        #   If the Cgroups option of slurm is enabled, you must take this way,
        #   if not so, the last pattern can be also used.
        # NOTE:
        #   Assume same number of GPUs for each nodes.

        logging.info(f"{args.num_nodes}-nodes and ngpu={args.ngpu} using srun")
        cmd = (
            args.cmd
            # arguments for ${cmd}
            + [
                "--gpu",
                str(args.ngpu),
                "--num_threads",
                str(max(args.ngpu, 1)),
                "--num_nodes",
                str(args.num_nodes),
                args.log,
                "srun",
            ]
            # arguments for *_train.py
            + args.args
            + [
                "--ngpu",
                str(args.ngpu),
                "--multiprocessing_distributed",
                "true",
                "--dist_launcher",
                "slurm",
            ]
            + init_method
        )
        if args.ngpu == 0:
            # Gloo supports both GPU and CPU mode.
            #   See: https://pytorch.org/docs/stable/distributed.html
            cmd += ["--dist_backend", "gloo"]
        process = subprocess.Popen(cmd)
        processes.append(process)
        logfile = args.log

    elif Path(args.cmd[0]).name == "run.pl":
        raise RuntimeError("run.pl can't submit jobs to the other nodes")

    else:
        # WARNING:
        #   Assuming that $CUDA_VISIBLE_DEVICES is automatically set by the system
        # WARNING:
        #   If the allocated devices are isolated from each other,
        #   e.g. by using Cgroups,
        #   then the internal communication within a node may be failed.

        world_size = args.num_nodes * max(args.ngpu, 1)
        logging.info(f"World_size={world_size}")

        for rank in range(world_size):
            cmd = (
                args.cmd
                # arguments for ${cmd}
                + [
                    "--gpu",
                    str(min(args.ngpu, 1)),
                    Path(args.log).stem + f".{rank}" + Path(args.log).suffix,
                ]
                # arguments for *_train.py
                + args.args
                + [
                    "--ngpu",
                    "1",
                    "--multiprocessing_distributed",
                    "false",
                    "--dist_rank",
                    str(rank),
                    "--dist_world_size",
                    str(world_size),
                ]
                + init_method
            )
            if args.ngpu == 0:
                # Gloo supports both GPU and CPU mode.
                #   See: https://pytorch.org/docs/stable/distributed.html
                cmd += ["--dist_backend", "gloo"]
            process = subprocess.Popen(cmd)
            processes.append(process)

        logfile = (Path(args.log).stem + ".*" + Path(args.log).suffix,)

    logging.info(f"log file: {logfile}")

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    main()
