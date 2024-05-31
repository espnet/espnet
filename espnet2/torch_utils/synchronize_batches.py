#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import torch
import torch.distributed as dist

@torch.no_grad()
def synchronize_sharded_batches(batches):
    """
    Synchronize shareded batches across all GPU rank.
    Duplicate some batches so that #batches across all GPU ranks are the same.
    """
    if not torch.cuda.is_available() or not dist.is_initialized():
        return batches
    
    n_batches = len(batches)
    n_batches_tensor = torch.Tensor([n_batches]).long().cuda()
    n_batches_list = [n_batches_tensor for _ in range(dist.get_world_size())]
    dist.all_gather(n_batches_list, n_batches_tensor)
    tgt_n_batches = max([t.cpu().item() for t in n_batches_list])

    if tgt_n_batches > n_batches:
        batches = batches + batches[-(tgt_n_batches - n_batches):]
        logging.info(f"Synchronize sharded dataset across all process")
        logging.info(f"#Batches: {n_batches} -> {tgt_n_batches}")

    return batches