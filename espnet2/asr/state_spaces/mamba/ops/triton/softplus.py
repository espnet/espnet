# Copyright (c) 2023, Tri Dao, Albert Gu.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code is based on Vision Mamba [1] (https://github.com/hustvl/Vim)
#
# [1] Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
#     https://arxiv.org/abs/2401.09417


import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")


if TRITON3:
    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)
else:
    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))