#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

from espnet.utils.cli_utils import get_commandline_args
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


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
        "--log",
        help="The path of log file used by cmd",
        default="run.log",
    )
    parser.add_argument(
        "--max_num_log_files",
        help="The maximum number of log-files to be kept",
        default=1000,
    )
    parser.add_argument(
        "--ngpu", type=int, default=1, help="The number of GPUs per node"
    )
    egroup = parser.add_mutually_exclusive_group()
    egroup.add_argument("--num_nodes", type=int, default=1, help="The number of nodes")
    egroup.add_argument(
        "--host",
        type=str,
        default=None,
        help="Directly specify the host names.  The job are submitted via SSH. "
        "Multiple host names can be specified by splitting by comma. e.g. host1,host2"
        " You can also the device id after the host name with ':'. e.g. "
        "host1:0:2:3,host2:0:2. If the device ids are specified in this way, "
        "the value of --ngpu is ignored.",
    )
    parser.add_argument(
        "--envfile",
        type=str_or_none,
        default="path.sh",
        help="Source the shell script before executing command. "
        "This option is used when --host is specified.",
    )

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
        default=".dist_init_",
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

    if args.host is None and shutil.which(args.cmd[0]) is None:
        raise RuntimeError(
            f"The first args of --cmd should be a script path. e.g. utils/run.pl: "
            f"{args.cmd[0]}"
        )

    # Specify init_method:
    #   See: https://pytorch.org/docs/stable/distributed.html#initialization
    if args.host is None and args.num_nodes <= 1:
        # Automatically set init_method if num_node=1
        init_method = None
    else:
        if args.master_port is None:
            # Try "shared-file system initialization" if master_port is not specified
            # Give random name to avoid reusing previous file
            init_file = args.init_file_prefix + str(uuid.uuid4())
            init_file = Path(init_file).absolute()
            Path(init_file).parent.mkdir(exist_ok=True, parents=True)
            init_method = ["--dist_init_method", f"file://{init_file}"]
        else:
            init_method = ["--dist_master_port", str(args.master_port)]

            # This can be omitted if slurm mode
            if args.master_addr is not None:
                init_method += ["--dist_master_addr", args.master_addr]
            elif args.host is not None:
                init_method += [
                    "--dist_master_addr",
                    args.host.split(",")[0].split(":")[0],
                ]

    # Log-rotation
    for i in range(args.max_num_log_files - 1, -1, -1):
        if i == 0:
            p = Path(args.log)
            pn = p.parent / (p.stem + ".1" + p.suffix)
        else:
            _p = Path(args.log)
            p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
            pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

        if p.exists():
            if i == args.max_num_log_files - 1:
                p.unlink()
            else:
                shutil.move(p, pn)

    processes = []
    # Submit command via SSH
    if args.host is not None:
        hosts = []
        ids_list = []
        # e.g. args.host = "host1:0:2,host2:0:1"
        for host in args.host.split(","):
            # e.g host = "host1:0:2"
            sps = host.split(":")
            host = sps[0]
            if len(sps) > 1:
                ids = [int(x) for x in sps[1:]]
            else:
                ids = list(range(args.ngpu))
            hosts.append(host)
            ids_list.append(ids)

        world_size = sum(max(len(x), 1) for x in ids_list)
        logging.info(f"{len(hosts)}nodes with world_size={world_size} via SSH")

        if args.envfile is not None:
            env = f"source {args.envfile}"
        else:
            env = ""

        if args.log != "-":
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            f = Path(args.log).open("w", encoding="utf-8")
        else:
            # Output to stdout/stderr
            f = None

        rank = 0
        for host, ids in zip(hosts, ids_list):
            ngpu = 1 if len(ids) > 0 else 0
            ids = ids if len(ids) > 0 else ["none"]

            for local_rank in ids:
                cmd = (
                    args.args
                    + [
                        "--ngpu",
                        str(ngpu),
                        "--multiprocessing_distributed",
                        "false",
                        "--local_rank",
                        str(local_rank),
                        "--dist_rank",
                        str(rank),
                        "--dist_world_size",
                        str(world_size),
                    ]
                    + init_method
                )
                if ngpu == 0:
                    # Gloo supports both GPU and CPU mode.
                    #   See: https://pytorch.org/docs/stable/distributed.html
                    cmd += ["--dist_backend", "gloo"]

                heredoc = f"""<< EOF
set -euo pipefail
cd {os.getcwd()}
{env}
{" ".join([c if len(c) != 0 else "''" for c in cmd])}
EOF
"""

                # FIXME(kamo): The process will be alive
                #  even if this program is stopped because we don't set -t here,
                #  i.e. not assigning pty,
                #  and the program is not killed when SSH connection is closed.
                process = subprocess.Popen(
                    ["ssh", host, "bash", heredoc],
                    stdout=f,
                    stderr=f,
                )

                processes.append(process)

                rank += 1

    # If Single node
    elif args.num_nodes <= 1:
        if args.ngpu > 1:
            if args.multiprocessing_distributed:
                # NOTE:
                #   If multiprocessing_distributed=true,
                # -> Distributed mode, which is multi-process and Multi-GPUs.
                #    and TCP initializetion is used if single-node case:
                #      e.g. init_method="tcp://localhost:20000"
                logging.info(f"single-node with {args.ngpu}gpu on distributed mode")
            else:
                # NOTE:
                #   If multiprocessing_distributed=false
                # -> "DataParallel" mode, which is single-process
                #    and Multi-GPUs with threading.
                # See:
                # https://discuss.pytorch.org/t/why-torch-nn-parallel-distributeddataparallel-runs-faster-than-torch-nn-dataparallel-on-single-machine-with-multi-gpu/32977/2
                logging.info(f"single-node with {args.ngpu}gpu using DataParallel")

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

    elif Path(args.cmd[0]).name == "run.pl":
        raise RuntimeError("run.pl doesn't support submitting to the other nodes.")

    elif Path(args.cmd[0]).name == "ssh.pl":
        raise RuntimeError("Use --host option instead of ssh.pl")

    # If Slurm
    elif Path(args.cmd[0]).name == "slurm.pl":
        logging.info(f"{args.num_nodes}nodes and {args.ngpu}gpu-per-node using srun")
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
                # Inherit all environment variable from parent process
                "--export=ALL",
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

    else:
        # This pattern can also works with Slurm.

        logging.info(f"{args.num_nodes}nodes and {args.ngpu}gpu-per-node using mpirun")
        cmd = (
            args.cmd
            # arguments for ${cmd}
            + [
                "--gpu",
                str(args.ngpu),
                "--num_threads",
                str(max(args.ngpu, 1)),
                # Make sure scheduler setting, i.e. conf/queue.conf
                # so that --num_nodes requires 1process-per-node
                "--num_nodes",
                str(args.num_nodes),
                args.log,
                "mpirun",
                # -np option can be omitted with Torque/PBS
                "-np",
                str(args.num_nodes),
            ]
            # arguments for *_train.py
            + args.args
            + [
                "--ngpu",
                str(args.ngpu),
                "--multiprocessing_distributed",
                "true",
                "--dist_launcher",
                "mpi",
            ]
            + init_method
        )
        if args.ngpu == 0:
            # Gloo supports both GPU and CPU mode.
            #   See: https://pytorch.org/docs/stable/distributed.html
            cmd += ["--dist_backend", "gloo"]
        process = subprocess.Popen(cmd)
        processes.append(process)

    logging.info(f"log file: {args.log}")

    failed = False
    while any(p.returncode is None for p in processes):
        for process in processes:
            # If any process is failed, try to kill the other processes too
            if failed and process.returncode is not None:
                process.kill()
            else:
                try:
                    process.wait(0.5)
                except subprocess.TimeoutExpired:
                    pass

                if process.returncode is not None and process.returncode != 0:
                    failed = True

    for process in processes:
        if process.returncode != 0:
            print(
                subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd),
                file=sys.stderr,
            )
            p = Path(args.log)
            if p.exists():
                with p.open() as f:
                    lines = list(f)
                raise RuntimeError(
                    f"\n################### The last 1000 lines of {args.log} "
                    f"###################\n" + "".join(lines[-1000:])
                )
            else:
                raise RuntimeError


if __name__ == "__main__":
    main()
