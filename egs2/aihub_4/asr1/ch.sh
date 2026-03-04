cat > /tmp/nccl_test.py <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

x = torch.ones(1, device="cuda")
dist.all_reduce(x)

print(f"rank {rank} local_rank {local_rank} ok {x.item()}")

dist.destroy_process_group()
PY

torchrun --nproc_per_node=2 /tmp/nccl_test.py