import dataclasses

import torch


@dataclasses.dataclass
class DistributedOption:
    # Enable distributed Training
    distributed: bool = False
    # torch.distributed.Backend: "nccl", "mpi", "gloo", or "tcp"
    dist_backend: str = "nccl"
    # if init_method="env://",
    # env values of "MASTER_PORT", "MASTER_ADDR", "WORLD_SIZE", and "RANK" are referred.
    dist_init_method: str = "env://"
    dist_world_size: int = -1
    dist_rank: int = -1

    def init(self):
        if self.distributed:
            torch.distributed.init_process_group(
                backend=self.dist_backend,
                init_method=self.dist_init_method,
                world_size=self.dist_world_size,
                rank=self.dist_rank,
            )
            self.dist_world_size = torch.distributed.get_world_size()
            self.dist_rank = torch.distributed.get_rank()
