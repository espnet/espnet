import dataclasses
import logging
import os
import socket
from typing import Optional

import torch
import torch.distributed


@dataclasses.dataclass
class DistributedOption:
    """
        Dataclass to manage distributed training options in PyTorch.

    Attributes:
        distributed (bool): Flag to enable distributed training. Default is False.
        dist_backend (str): The backend to use for distributed training. Options
            include "nccl", "mpi", "gloo", or "tcp". Default is "nccl".
        dist_init_method (str): Method for initializing the process group. If
            "env://", it uses environment variables for configuration. Default is
            "env://".
        dist_world_size (Optional[int]): Total number of processes participating
            in the job. Default is None.
        dist_rank (Optional[int]): Rank of the current process. Default is None.
        local_rank (Optional[int]): Rank of the current process on the node.
            Default is None.
        ngpu (int): Number of GPUs available for training. Default is 0.
        dist_master_addr (Optional[str]): Address of the master process. Default
            is None.
        dist_master_port (Optional[int]): Port of the master process. Default is
            None.
        dist_launcher (Optional[str]): The launcher used to start the distributed
            job. Default is None.
        multiprocessing_distributed (bool): Flag indicating if the training is
            using multiprocessing. Default is True.

    Methods:
        init_options: Initializes distributed training options based on the
            specified attributes.
        init_torch_distributed: Initializes the PyTorch distributed process
            group.
        init_deepspeed: Initializes DeepSpeed distributed training.

    Raises:
        RuntimeError: If required environment variables are not set or if
            inconsistencies in ranks or world size are detected.
        ValueError: If trying to initialize DeepSpeed without initializing
            PyTorch distributed first.

    Examples:
        >>> options = DistributedOption(distributed=True, ngpu=2)
        >>> options.init_options()
        >>> options.init_torch_distributed()
        >>> options.init_deepspeed()

    Note:
        This class is designed to be used in distributed training scenarios,
        particularly with PyTorch and DeepSpeed.

    Todo:
        Add more validation checks for distributed settings in the future.
    """

    # Enable distributed Training
    distributed: bool = False
    # torch.distributed.Backend: "nccl", "mpi", "gloo", or "tcp"
    dist_backend: str = "nccl"
    # if init_method="env://",
    # env values of "MASTER_PORT", "MASTER_ADDR", "WORLD_SIZE", and "RANK" are referred.
    dist_init_method: str = "env://"
    dist_world_size: Optional[int] = None
    dist_rank: Optional[int] = None
    local_rank: Optional[int] = None
    ngpu: int = 0
    dist_master_addr: Optional[str] = None
    dist_master_port: Optional[int] = None
    dist_launcher: Optional[str] = None
    multiprocessing_distributed: bool = True

    def init_options(self):
        """
            Initialize the options for distributed training.

        This method configures the distributed training settings based on
        the specified attributes. It verifies that the necessary environment
        variables are set and assigns values to the `dist_rank`,
        `dist_world_size`, and `local_rank` attributes. It also checks
        for potential issues such as exceeding the number of visible devices.

        If the `dist_init_method` is set to "env://", the method will attempt
        to retrieve the master address and port from the environment variables
        or use the specified values. If both the master address and port are
        provided, it will set the `dist_init_method` to a TCP URL format.

        Raises:
            RuntimeError: If required environment variables or attributes are
            not set correctly or if the rank exceeds the world size.

        Examples:
            # Example 1: Using default environment variables
            options = DistributedOption(distributed=True)
            options.init_options()

            # Example 2: Custom master address and port
            options = DistributedOption(
                distributed=True,
                dist_master_addr="192.168.1.1",
                dist_master_port=12345
            )
            options.init_options()
        """
        if self.distributed:
            if self.dist_init_method == "env://":
                if get_master_addr(self.dist_master_addr, self.dist_launcher) is None:
                    raise RuntimeError(
                        "--dist_master_addr or MASTER_ADDR must be set "
                        "if --dist_init_method == 'env://'"
                    )
                if get_master_port(self.dist_master_port) is None:
                    raise RuntimeError(
                        "--dist_master_port or MASTER_PORT must be set "
                        "if --dist_init_port == 'env://'"
                    )

            # About priority order:
            # If --dist_* is specified:
            #    Use the value of --dist_rank and overwrite it environ just in case.
            # elif environ is set:
            #    Use the value of environ and set it to self
            self.dist_rank = get_rank(self.dist_rank, self.dist_launcher)
            self.dist_world_size = get_world_size(
                self.dist_world_size, self.dist_launcher
            )
            self.local_rank = get_local_rank(self.local_rank, self.dist_launcher)

            if self.local_rank is not None:
                if self.ngpu > 1:
                    raise RuntimeError(f"Assuming 1GPU in this case: ngpu={self.ngpu}")
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    cvd = os.environ["CUDA_VISIBLE_DEVICES"]
                    if self.local_rank >= len(cvd.split(",")):
                        raise RuntimeError(
                            f"LOCAL_RANK={self.local_rank} is bigger "
                            f"than the number of visible devices: {cvd}"
                        )

            if (
                self.dist_rank is not None
                and self.dist_world_size is not None
                and self.dist_rank >= self.dist_world_size
            ):
                raise RuntimeError(
                    f"RANK >= WORLD_SIZE: {self.dist_rank} >= {self.dist_world_size}"
                )

            if self.dist_init_method == "env://":
                self.dist_master_addr = get_master_addr(
                    self.dist_master_addr, self.dist_launcher
                )
                self.dist_master_port = get_master_port(self.dist_master_port)
                if (
                    self.dist_master_addr is not None
                    and self.dist_master_port is not None
                ):
                    self.dist_init_method = (
                        f"tcp://{self.dist_master_addr}:{self.dist_master_port}"
                    )

    def init_torch_distributed(self):
        """
            Initializes the PyTorch distributed environment.

        This method sets up the distributed training environment using
        PyTorch's `torch.distributed` module. It checks if distributed
        training is enabled and initializes the process group based on
        the specified backend, initialization method, world size, and
        rank.

        It also configures the CUDA device if multiple GPUs are being
        used and the local rank is specified.

        Note:
            This method should be called after the distributed options
            have been set up correctly, typically after calling
            `init_options`.

        Raises:
            ValueError: If the distributed environment is not properly
                initialized or if the rank is greater than or equal to
                the world size.

        Examples:
            >>> dist_option = DistributedOption(distributed=True)
            >>> dist_option.init_options()
            >>> dist_option.init_torch_distributed()

        See Also:
            PyTorch documentation on distributed training:
            https://pytorch.org/docs/stable/distributed.html
        """
        if self.distributed:
            # See:
            # https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html
            os.environ.setdefault("NCCL_DEBUG", "INFO")

            # See:
            # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
            os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

            torch.distributed.init_process_group(
                backend=self.dist_backend,
                init_method=self.dist_init_method,
                world_size=self.dist_world_size,
                rank=self.dist_rank,
            )

            # About distributed model:
            # if self.local_rank is not None and ngpu == 1
            #    => Distributed with n-Process and n-GPU
            # if self.local_rank is None and ngpu >= 1
            #    => Distributed with 1-Process and n-GPU
            if self.local_rank is not None and self.ngpu > 0:
                torch.cuda.set_device(self.local_rank)

    def init_deepspeed(self):
        """
            Initialize DeepSpeed for distributed training.

        This method sets up DeepSpeed for distributed training by first ensuring
        that PyTorch's distributed backend is initialized. It checks that the
        necessary environment variables are set and raises appropriate errors
        if they are not. The method also logs a warning if the `OMP_NUM_THREADS`
        environment variable is set to 1, suggesting that this may not be sufficient
        for optimal performance with DeepSpeed.

        Raises:
            ImportError: If the DeepSpeed package cannot be imported.
            ValueError: If PyTorch distributed is not initialized before
            initializing DeepSpeed.

        Examples:
            >>> distributed_options = DistributedOption(distributed=True)
            >>> distributed_options.init_options()
            >>> distributed_options.init_torch_distributed()
            >>> distributed_options.init_deepspeed()

        Note:
            Ensure that the environment variables for distributed training,
            such as `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`, are properly set
            before calling this method.
        """
        try:
            import deepspeed
        except ImportError:
            raise

        if not torch.distributed.is_initialized():
            raise ValueError(
                "Should initailize torch distributed before initializing deepspeed"
            )

        # NOTE(Jinchuan): init torch distributed backend first. Then
        # deepspeed will find that backend automatically.
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.dist_rank)
        os.environ["WORLD_SIZE"] = str(self.dist_world_size)
        if int(os.environ["OMP_NUM_THREADS"]) == 1:
            logging.warning(
                "\n=================================================================\n"
                "Found OMP_NUM_THREADS=1 in environment variables. "
                "With some advanced features, DeepSpeed may have heavy cpu workload "
                "so that OMP_NUM_THREADS=1 is not sufficient. "
                "Try to increase it in your path.sh \n"
                "================================================================="
            )

        deepspeed.init_distributed()


def resolve_distributed_mode(args):
    """
    Resolve the distributed training mode based on the provided arguments.

    This function sets the `args.distributed` attribute based on the configuration
    of the distributed training environment. It checks various conditions to determine
    if the training should run in distributed mode, such as the number of nodes,
    GPUs, and the launcher being used.

    Args:
        args: An object containing the training arguments, which should include:
            - `args.multiprocessing_distributed`: A boolean indicating if
              multiprocessing distributed mode is enabled.
            - `args.dist_world_size`: An integer specifying the total number of
              processes participating in the job.
            - `args.ngpu`: An integer representing the number of GPUs available.
            - `args.dist_rank`: An optional integer representing the rank of the
              current process.
            - `args.local_rank`: An optional integer representing the local rank
              of the current process.
            - `args.dist_launcher`: A string indicating the launcher being used
              (e.g., "slurm", "mpi").

    Raises:
        RuntimeError: If the conditions for distributed training are not met,
        such as missing required arguments or misconfiguration.

    Examples:
        >>> class Args:
        ...     def __init__(self):
        ...         self.multiprocessing_distributed = True
        ...         self.dist_world_size = 4
        ...         self.ngpu = 2
        ...         self.dist_rank = None
        ...         self.local_rank = None
        ...         self.dist_launcher = "slurm"
        ...
        >>> args = Args()
        >>> resolve_distributed_mode(args)
        >>> print(args.distributed)
        True

    Note:
        This function modifies the `args` object in place. After calling this
        function, `args.distributed`, `args.local_rank`, and other related
        attributes will be set according to the determined mode.
    """
    # Note that args.distributed is set by only this function.
    # and ArgumentParser doesn't have such option

    if args.multiprocessing_distributed:
        num_nodes = get_num_nodes(args.dist_world_size, args.dist_launcher)
        # a. multi-node
        if num_nodes > 1:
            args.distributed = True
        # b. single-node and multi-gpu with multiprocessing_distributed mode
        elif args.ngpu > 1:
            args.distributed = True
        # c. single-node and single-gpu
        else:
            args.distributed = False

        if args.ngpu <= 1:
            # Disable multiprocessing_distributed mode if 1process per node or cpu mode
            args.multiprocessing_distributed = False
        if args.ngpu == 1:
            # If the number of GPUs equals to 1 with multiprocessing_distributed mode,
            # LOCAL_RANK is always 0
            args.local_rank = 0

        if num_nodes > 1 and get_node_rank(args.dist_rank, args.dist_launcher) is None:
            raise RuntimeError(
                "--dist_rank or RANK must be set "
                "if --multiprocessing_distributed == true"
            )

        # Note that RANK, LOCAL_RANK, and WORLD_SIZE is automatically set,
        # so we don't need to check here
    else:
        # d. multiprocess and multi-gpu with external launcher
        #    e.g. torch.distributed.launch
        if get_world_size(args.dist_world_size, args.dist_launcher) > 1:
            args.distributed = True
        # e. single-process
        else:
            args.distributed = False

        if args.distributed and args.ngpu > 0:
            if get_local_rank(args.local_rank, args.dist_launcher) is None:
                raise RuntimeError(
                    "--local_rank or LOCAL_RANK must be set "
                    "if --multiprocessing_distributed == false"
                )
        if args.distributed:
            if get_node_rank(args.dist_rank, args.dist_launcher) is None:
                raise RuntimeError(
                    "--dist_rank or RANK must be set "
                    "if --multiprocessing_distributed == false"
                )
    if args.distributed and args.dist_launcher == "slurm" and not is_in_slurm_step():
        raise RuntimeError("Launch by 'srun' command if --dist_launcher='slurm'")


def is_in_slurm_job() -> bool:
    """
    Check if the current process is running within a SLURM job.

    This function determines if the environment variables indicative of a SLURM
    job are set. It checks for the presence of the "SLURM_PROCID" and
    "SLURM_NTASKS" environment variables, which are typically set when a job
    is launched using SLURM.

    Returns:
        bool: True if the process is running in a SLURM job, False otherwise.

    Examples:
        >>> os.environ["SLURM_PROCID"] = "0"
        >>> os.environ["SLURM_NTASKS"] = "4"
        >>> is_in_slurm_job()
        True

        >>> os.environ.pop("SLURM_PROCID")
        >>> is_in_slurm_job()
        False
    """
    return "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ


def is_in_slurm_step() -> bool:
    """
    Check if the current process is running in a SLURM step.

    This function verifies whether the current process is part of a SLURM
    job step by checking the presence of specific environment variables
    related to SLURM. It returns `True` if the process is indeed in a
    SLURM step, and `False` otherwise.

    Returns:
        bool: True if the process is in a SLURM step, False otherwise.

    Examples:
        >>> os.environ['SLURM_PROCID'] = '0'
        >>> os.environ['SLURM_NTASKS'] = '2'
        >>> os.environ['SLURM_STEP_NUM_NODES'] = '1'
        >>> os.environ['SLURM_STEP_NODELIST'] = 'node[1-2]'
        >>> is_in_slurm_step()
        True

        >>> del os.environ['SLURM_STEP_NUM_NODES']
        >>> is_in_slurm_step()
        False
    """
    return (
        is_in_slurm_job()
        and "SLURM_STEP_NUM_NODES" in os.environ
        and "SLURM_STEP_NODELIST" in os.environ
    )


def _int_or_none(x: Optional[str]) -> Optional[int]:
    if x is None:
        return x
    return int(x)


def free_port():
    """
    Find a free port using bind().

    This function attempts to find a free port by creating a socket and
    binding it to an available port. However, there is a possibility that
    the port may be occupied by another process after it has been found,
    as there is an interval between finding the port and using it. Thus,
    it is not guaranteed that the port is truly free at the time of usage.

    Returns:
        int: A free port number.

    Examples:
        >>> port = free_port()
        >>> print(port)  # Outputs a free port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def get_rank(prior=None, launcher: Optional[str] = None) -> Optional[int]:
    """
    Get the rank of the current process in a distributed setting.

    This function retrieves the rank of the current process. The rank is
    defined as an integer that uniquely identifies a process within a
    distributed training setup. If the `prior` argument is provided,
    it will be used as the rank. If `prior` is not specified, the function
    checks the environment variables or uses the specified launcher
    (e.g., "slurm", "mpi") to determine the rank.

    Args:
        prior (Optional[int]): An optional prior rank to use if provided.
        launcher (Optional[str]): The launcher used to start the process
            (e.g., "slurm", "mpi").

    Returns:
        Optional[int]: The rank of the current process, or None if it
        cannot be determined.

    Raises:
        RuntimeError: If the process is launched with an unsupported
        launcher or if the required environment variables are not set.

    Examples:
        >>> os.environ["RANK"] = "2"
        >>> get_rank()  # Returns 2

        >>> os.environ["SLURM_PROCID"] = "0"
        >>> get_rank(launcher="slurm")  # Returns 0

        >>> get_rank(1)  # Returns 1

    Note:
        If `prior` is None and no environment variable is set, the
        function will return None.
    """
    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")
            prior = os.environ["SLURM_PROCID"]
        elif launcher == "mpi":
            raise RuntimeError(
                "launcher=mpi is used for 'multiprocessing-distributed' mode"
            )
        elif launcher is not None:
            raise RuntimeError(f"launcher='{launcher}' is not supported")

    if prior is not None:
        return int(prior)
    else:
        # prior is None and RANK is None -> RANK = None
        return _int_or_none(os.environ.get("RANK"))


def get_world_size(prior=None, launcher: Optional[str] = None) -> int:
    """
    Get the world size for distributed training.

    The world size refers to the total number of processes participating in
    the distributed training. If the `prior` argument is provided, it will
    be used as the world size. Otherwise, the function will attempt to read
    the world size from environment variables based on the launcher type.

    Args:
        prior (Optional[int]): The world size to use if specified.
        launcher (Optional[str]): The type of launcher used, e.g., "slurm",
            "mpi", or None.

    Returns:
        int: The world size, which defaults to 1 if no valid value can
        be determined.

    Raises:
        RuntimeError: If the specified launcher is not supported or if
        the process is not launched correctly for the specified launcher.

    Examples:
        >>> get_world_size()  # Assuming WORLD_SIZE=4 in the environment
        4

        >>> get_world_size(prior=2)
        2

        >>> get_world_size(launcher='slurm')  # Assuming SLURM_NTASKS=3
        3
    """
    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")
            prior = int(os.environ["SLURM_NTASKS"])
        elif launcher == "mpi":
            raise RuntimeError(
                "launcher=mpi is used for 'multiprocessing-distributed' mode"
            )
        elif launcher is not None:
            raise RuntimeError(f"launcher='{launcher}' is not supported")

    if prior is not None:
        return int(prior)
    else:
        # prior is None and WORLD_SIZE is None -> WORLD_SIZE = 1
        return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank(prior=None, launcher: Optional[str] = None) -> Optional[int]:
    """
        Get the local rank of the process.

    The local rank corresponds to the GPU device ID when using distributed training.
    This function retrieves the local rank based on the provided prior value or
    environment variables. It is essential for configuring distributed training,
    particularly when multiple processes are launched across multiple GPUs.

    Args:
        prior (Optional[int]): The local rank to use if provided. If None, the
            function will check environment variables for the local rank.

        launcher (Optional[str]): The type of launcher used to start the process.
            Supported values include "slurm" and "mpi". If None, it will only
            check environment variables.

    Returns:
        Optional[int]: The local rank of the current process, or None if it cannot
            be determined.

    Raises:
        RuntimeError: If the launcher is "slurm" and the process is not launched
            by 'srun'.
        RuntimeError: If the launcher is "mpi" which is used for
            'multiprocessing-distributed' mode.
        RuntimeError: If an unsupported launcher is specified.

    Examples:
        # Example usage in a distributed training setup
        local_rank = get_local_rank()
        if local_rank is not None:
            print(f"Local rank is: {local_rank}")
        else:
            print("Local rank could not be determined.")

    Note:
        The LOCAL_RANK environment variable should be set when using distributed
        training frameworks like PyTorch or when launching via SLURM.
    """
    # LOCAL_RANK is same as GPU device id

    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")

            prior = int(os.environ["SLURM_LOCALID"])
        elif launcher == "mpi":
            raise RuntimeError(
                "launcher=mpi is used for 'multiprocessing-distributed' mode"
            )
        elif launcher is not None:
            raise RuntimeError(f"launcher='{launcher}' is not supported")

    if prior is not None:
        return int(prior)

    elif "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        # There are two possibility:
        # - "CUDA_VISIBLE_DEVICES" is set to multiple GPU ids. e.g. "0.1,2"
        #   => This intends to specify multiple devices to to be used exactly
        #      and local_rank information is possibly insufficient.
        # - "CUDA_VISIBLE_DEVICES" is set to an id. e.g. "1"
        #   => This could be used for LOCAL_RANK
        cvd = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(cvd) == 1 and "LOCAL_RANK" not in os.environ:
            # If CUDA_VISIBLE_DEVICES is set and LOCAL_RANK is not set,
            # then use it as LOCAL_RANK.

            # Unset CUDA_VISIBLE_DEVICES
            # because the other device must be visible to communicate
            return int(os.environ.pop("CUDA_VISIBLE_DEVICES"))
        else:
            return None
    else:
        return None


def get_master_addr(prior=None, launcher: Optional[str] = None) -> Optional[str]:
    """
    Retrieve the master address for distributed training.

    This function determines the master address based on the provided
    `prior` value or by checking the environment variables. If the
    `launcher` is set to "slurm", it will extract the address from the
    SLURM node list.

    Args:
        prior (Optional[str]): The address to use if provided. If None, the
            function checks the environment variable `MASTER_ADDR`.
        launcher (Optional[str]): The type of launcher used for distributed
            training (e.g., "slurm"). If "slurm", it will get the address
            from the SLURM environment.

    Returns:
        Optional[str]: The master address as a string, or None if it cannot
        be determined.

    Raises:
        RuntimeError: If the function is called with "slurm" as the
            launcher but not within a valid SLURM step.

    Examples:
        >>> # Example usage when not using a launcher
        >>> addr = get_master_addr()
        >>> print(addr)  # Might print the address set in MASTER_ADDR

        >>> # Example usage with SLURM launcher
        >>> addr = get_master_addr(launcher='slurm')
        >>> print(addr)  # Prints the master address from SLURM node list
    """
    if prior is None:
        if launcher == "slurm":
            if not is_in_slurm_step():
                raise RuntimeError("This process seems not to be launched by 'srun'")

            # e.g nodelist = foo[1-10],bar[3-8] or foo4,bar[2-10]
            nodelist = os.environ["SLURM_STEP_NODELIST"]
            prior = nodelist.split(",")[0].split("-")[0].replace("[", "")

    if prior is not None:
        return str(prior)
    else:
        return os.environ.get("MASTER_ADDR")


def get_master_port(prior=None) -> Optional[int]:
    """
        Get the master port for distributed training.

    This function retrieves the master port to be used in a distributed training
    setup. If a prior value is provided, it will be returned; otherwise, the
    function checks the environment variable `MASTER_PORT`. If `MASTER_PORT` is
    not set, it returns `None`.

    Args:
        prior (Optional[int]): The prior port value. If not provided, the function
        will look for the `MASTER_PORT` environment variable.

    Returns:
        Optional[int]: The master port if it is provided or found in the environment,
        otherwise `None`.

    Examples:
        >>> # Assuming the environment variable is set
        >>> os.environ["MASTER_PORT"] = "12345"
        >>> get_master_port()  # Returns 12345

        >>> # Without prior value and no environment variable set
        >>> get_master_port()  # Returns None

        >>> # With a prior value
        >>> get_master_port(8080)  # Returns 8080
    """
    if prior is not None:
        return prior
    else:
        return _int_or_none(os.environ.get("MASTER_PORT"))


def get_node_rank(prior=None, launcher: Optional[str] = None) -> Optional[int]:
    """
    Get Node Rank.

    This function is used for "multiprocessing distributed" mode. The initial
    RANK equals the Node ID in this case, and the real Rank is set as
    (nGPU * NodeID) + LOCAL_RANK in `torch.distributed`.

    Args:
        prior (Optional[int]): The prior rank to return if provided.
        launcher (Optional[str]): The launcher type, e.g., "slurm" or "mpi".

    Returns:
        Optional[int]: The node rank or None if not determined.

    Raises:
        RuntimeError: If not launched by 'srun' when using 'slurm' launcher.
        RuntimeError: If `ntasks_per_node` does not equal `SLURM_NTASKS`.
        RuntimeError: If an unsupported launcher is specified.

    Examples:
        >>> get_node_rank()  # Assuming proper environment variables are set
        0  # returns the node rank for the current process.

    Note:
        This function assumes that `ntasks_per_node` is 1. If this assumption
        is violated, the behavior may be undefined.
    """
    if prior is not None:
        return prior
    elif launcher == "slurm":
        if not is_in_slurm_step():
            raise RuntimeError("This process seems not to be launched by 'srun'")

        # Assume ntasks_per_node == 1
        if os.environ["SLURM_STEP_NUM_NODES"] != os.environ["SLURM_NTASKS"]:
            raise RuntimeError(
                "Run with --ntasks_per_node=1 if mutliprocessing_distributed=true"
            )
        return int(os.environ["SLURM_NODEID"])
    elif launcher == "mpi":
        # Use mpi4py only for initialization and not using for communication
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        # Assume ntasks_per_node == 1 (We can't check whether it is or not)
        return comm.Get_rank()
    elif launcher is not None:
        raise RuntimeError(f"launcher='{launcher}' is not supported")
    else:
        return _int_or_none(os.environ.get("RANK"))


def get_num_nodes(prior=None, launcher: Optional[str] = None) -> Optional[int]:
    """
        Get the number of nodes.

    Use for "multiprocessing distributed" mode. The RANK equals to the Node ID in
    this case and the real Rank is set as (nGPU * NodeID) + LOCAL_RANK in
    torch.distributed.

    This function determines the number of nodes participating in the
    distributed training setup. It checks the launcher type (e.g., slurm, mpi)
    to retrieve the appropriate environment variables or uses the provided
    parameter if available.

    Args:
        prior (Optional[int]): An optional prior value for the number of nodes.
            If provided, this value will be returned directly without checking
            the environment.

        launcher (Optional[str]): The launcher type used to start the process.
            This can be "slurm", "mpi", or None. If None, it defaults to
            checking the WORLD_SIZE environment variable.

    Returns:
        Optional[int]: The number of nodes participating in the distributed
        training. Returns 1 if no nodes are found and no prior is provided.

    Raises:
        RuntimeError: If the launcher is "slurm" and the environment is not
        set up correctly, or if the launcher is not supported.

    Examples:
        >>> get_num_nodes()
        1

        >>> get_num_nodes(prior=3)
        3

        >>> get_num_nodes(launcher="slurm")
        5  # Assuming SLURM_STEP_NUM_NODES is set to 5 in the environment.
    """
    if prior is not None:
        return prior
    elif launcher == "slurm":
        if not is_in_slurm_step():
            raise RuntimeError("This process seems not to be launched by 'srun'")

        # Assume ntasks_per_node == 1
        if os.environ["SLURM_STEP_NUM_NODES"] != os.environ["SLURM_NTASKS"]:
            raise RuntimeError(
                "Run with --ntasks_per_node=1 if mutliprocessing_distributed=true"
            )
        return int(os.environ["SLURM_STEP_NUM_NODES"])
    elif launcher == "mpi":
        # Use mpi4py only for initialization and not using for communication
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        # Assume ntasks_per_node == 1 (We can't check whether it is or not)
        return comm.Get_size()
    elif launcher is not None:
        raise RuntimeError(f"launcher='{launcher}' is not supported")
    else:
        # prior is None -> NUM_NODES = 1
        return int(os.environ.get("WORLD_SIZE", 1))
