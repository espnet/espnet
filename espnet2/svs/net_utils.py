#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
# Copyright 2024 Yuxun Tang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, Optional, Tuple

import torch

def pad_and_concat(tensor_list, pad_id=0):
    """pad a list of torch.Tensor with shape [B_n, T_n, ...]
    in T dimension and then concat in B dimension
    """

    size_list = [t.size() for t in tensor_list]
    concat_size = sum([size[0] for size in size_list])
    pad_size = max([size[1] for size in size_list])
    assert all([size[2:] == size_list[0][2:] for size in size_list])

    retval = (
        torch.ones(
            tuple([concat_size, pad_size] + list(tensor_list[0].size()[2:])),
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )
        * pad_id
    )

    count = 0
    for t in tensor_list:
        B, T = t.size()[:2]
        retval[count : count + B, :T] = t
        count += B

    return retval

