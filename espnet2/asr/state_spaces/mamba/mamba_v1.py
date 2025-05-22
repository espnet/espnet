# Copyright (c) 2023, Tri Dao, Albert Gu.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code is based on Vision Mamba [1] (https://github.com/hustvl/Vim)
#
# [1] Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
#     https://arxiv.org/abs/2401.09417


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

try:
    from espnet2.asr.state_spaces.mamba.ops.selective_scan_interface import (
        mamba_inner_fn,
        mamba_inner_fn_no_out_proj,
        selective_scan_fn,
    )
except ImportError:
    selective_scan_fn, mamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from espnet2.asr.state_spaces.mamba.ops.triton.selective_scan_update import (
        selective_state_update,
    )
except ImportError:
    selective_state_update = None

try:
    from espnet2.asr.state_spaces.mamba.ops.triton.layer_norm import (
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn,
    )
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from espnet2.asr.state_spaces.s4 import OptimModule


class TransposedDropout1D(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = torch.nn.Dropout1d(p)

    def forward(self, x):
        assert x.dim() == 3, x.dim()
        x = self.dropout(x.transpose(1, 2))
        return x.transpose(1, 2)


class Mamba(OptimModule):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bidirectional: bool = False,
        bidirectional_mask: bool = True,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        dropout_type: str = "dropout1d",
        projection_dropout_rate: float = 0.0,
        use_causal_conv1d: bool = True,
        ssm_lr: Optional[float] = None,
        prefix_bidir: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bidirectional = bidirectional
        self.bidirectional_mask = bidirectional_mask
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.prefix_bidir = prefix_bidir
        if dropout_type == "dropout":
            self.dropout = nn.Dropout(projection_dropout_rate)
        elif dropout_type == "dropout1d":
            self.dropout = TransposedDropout1D(projection_dropout_rate)
        else:
            raise NotImplementedError

        self.use_causal_conv1d = use_causal_conv1d

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if ssm_lr is not None:
            # No wight decay will be applied in the optimizer hooks
            self.register("A_log", A_log, lr=ssm_lr)
        else:
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        if self.bidirectional:
            self.conv1d_bwd = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_bwd = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )
            self.dt_proj_bwd = nn.Linear(
                self.dt_rank, self.d_inner, bias=True, **factory_kwargs
            )

            # Initialize special dt projection to preserve variance at initialization
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_proj_bwd.bias._no_reinit = True

            # S4D real initialization
            A_bwd = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_bwd_log = torch.log(A_bwd)  # Keep A_log in fp32
            if ssm_lr is not None:
                # No wight decay will be applied in the optimizer hooks
                self.register("A_bwd_log", A_bwd_log, lr=ssm_lr)
            else:
                self.A_bwd_log = nn.Parameter(A_bwd_log)
                self.A_bwd_log._no_weight_decay = True

            # D "skip" parameter
            self.D_bwd = nn.Parameter(
                torch.ones(self.d_inner, device=device)
            )  # Keep in fp32
            self.D_bwd._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(
        self, hidden_states, mask=None, inference_params=None, flip_fn=None, **kwargs
    ):
        """
        hidden_states: (B, L, D)
        mask: (B, 1, L)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            if not self.bidirectional:
                if self.prefix_bidir:
                    conv_state, ssm_state, conv_state_pb, ssm_state_pb = (
                        self._get_states_from_cache_pb(inference_params, batch)
                    )
                    if kwargs["direction"] == "backward":
                        conv_state, ssm_state = conv_state_pb, ssm_state_pb
                else:
                    conv_state, ssm_state = self._get_states_from_cache(
                        inference_params, batch
                    )
                if inference_params.seqlen_offset > 0:
                    # The states are updated inplace
                    out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                    return out
            else:
                conv_state, ssm_state, conv_state_pb, ssm_state_pb = (
                    self._get_states_from_cache_pb(inference_params, batch)
                )
                if inference_params.seqlen_offset > 0:
                    out, _, _, _, _ = self.bi_mamba_forward_step(
                        hidden_states,
                        conv_state,
                        ssm_state,
                        conv_state_pb,
                        ssm_state_pb,
                        mask,
                    )
                    return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        xz = self.dropout(xz)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path and inference_params is None
        ):  # Doesn't support outputting the states
            if not self.bidirectional:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out = self.dropout(out)
            else:
                A_bwd = -torch.exp(self.A_bwd_log.float())

                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

                # NOTE (Y. Masuyama): The mask is applied just before the backward model
                # This is for the case masking out not only padded tokens but also text tokens
                if self.bidirectional_mask:
                    xz = xz * mask

                if flip_fn is None:
                    _xz = torch.flip(xz, dims=[-1])
                else:
                    _xz = flip_fn[0](xz)

                out_bwd = mamba_inner_fn_no_out_proj(
                    _xz,
                    self.conv1d_bwd.weight,
                    self.conv1d_bwd.bias,
                    self.x_proj_bwd.weight,
                    self.dt_proj_bwd.weight,
                    A_bwd,
                    None,
                    None,
                    self.D_bwd.float(),
                    delta_bias=self.dt_proj_bwd.bias.float(),
                    delta_softplus=True,
                )

                if flip_fn is None:
                    out_bwd = torch.flip(out_bwd, dims=[-1])
                else:
                    out_bwd = flip_fn[1](out_bwd)

                out = F.linear(
                    rearrange(out + out_bwd, "b d l -> b l d"),
                    self.out_proj.weight,
                    self.out_proj.bias,
                )
                out = self.dropout(out)
        else:
            """
            NOTE (Y. Masuyama): LM decoding calls this part to handle prefix speech tokens
            """
            if not self.bidirectional:
                x, z = xz.chunk(2, dim=1)

                if self.use_causal_conv1d:
                    # Compute short convolution
                    if conv_state is not None:
                        # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                        # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                        conv_state.copy_(
                            F.pad(x, (self.d_conv - x.shape[-1], 0))
                        )  # Update state (B D W)
                    if causal_conv1d_fn is None:
                        x = self.act(self.conv1d(x)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x = causal_conv1d_fn(
                            x=x,
                            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            bias=self.conv1d.bias,
                            activation=self.activation,
                        )

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(
                    x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")
                out = self.dropout(self.out_proj(y))

            else:
                # Forward block
                x, z = xz.chunk(2, dim=1)

                if self.use_causal_conv1d:
                    # Compute short convolution
                    if conv_state is not None:
                        # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                        # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                        conv_state.copy_(
                            F.pad(x, (self.d_conv - x.shape[-1], 0))
                        )  # Update state (B D W)
                    if causal_conv1d_fn is None:
                        x = self.act(self.conv1d(x)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x = causal_conv1d_fn(
                            x=x,
                            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            bias=self.conv1d.bias,
                            activation=self.activation,
                        )
                """
                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                """
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(
                    x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )

                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")

                if self.bidirectional_mask:
                    xz = xz * mask

                if flip_fn is None:
                    _xz = torch.flip(xz, dims=[-1])
                else:
                    _xz = flip_fn[0](xz)

                x, z = _xz.chunk(2, dim=1)
                A_bwd = -torch.exp(self.A_bwd_log.float())

                if self.use_causal_conv1d:
                    # Compute short convolution
                    if conv_state_pb is not None:
                        # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                        # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                        conv_state_pb.copy_(
                            F.pad(x, (self.d_conv - x.shape[-1], 0))
                        )  # Update state (B D W)
                    if causal_conv1d_fn is None:
                        x = self.act(self.conv1d_bwd(x)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x = causal_conv1d_fn(
                            x=x,
                            weight=rearrange(self.conv1d_bwd.weight, "d 1 w -> d w"),
                            bias=self.conv1d_bwd.bias,
                            activation=self.activation,
                        )
                """
                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                """
                x_dbl = self.x_proj_bwd(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(
                    x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                dt = self.dt_proj_bwd.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y_bwd = selective_scan_fn(
                    x,
                    dt,
                    A_bwd,
                    B,
                    C,
                    self.D_bwd.float(),
                    z=z,
                    delta_bias=self.dt_proj_bwd.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state_pb is not None,
                )

                if ssm_state_pb is not None:
                    y_bwd, last_state = y_bwd
                    ssm_state_pb.copy_(last_state)

                if flip_fn is None:
                    y_bwd = torch.flip(y_bwd, dims=[-1])
                else:
                    y_bwd = flip_fn[1](y_bwd)

                y_bwd = rearrange(y_bwd, "b d l -> b l d")

                out = self.out_proj(y + y_bwd)
                out = self.dropout(out)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, (
            "Only support decoding with 1 token at a time for now"
        )
        assert not self.bidirectional, "Only support Unidirectional Mamba"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def bi_mamba_forward_step(
        self, hidden_states, conv_state, ssm_state, conv_state_pb, ssm_state_pb, mask
    ):
        assert hidden_states.shape[1] == 1, (
            "Only support decoding with 1 token at a time for now"
        )
        assert self.bidirectional, "Only support Bidirectional Mamba"
        assert causal_conv1d_update is not None
        assert selective_state_update is not None

        # Shared process
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)

        # Forward Mamba
        x, z = xz.chunk(2, dim=-1)  # (B D)
        x = causal_conv1d_update(
            x,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = selective_state_update(
            ssm_state,
            x,
            dt,
            A,
            B,
            C,
            self.D,
            z=z,
            dt_bias=self.dt_proj.bias,
            dt_softplus=True,
        )

        if not mask:
            out = self.out_proj(y)
            return out.unsqueeze(1), conv_state, ssm_state, conv_state_pb, ssm_state_pb

        # Backward Mamba, but it processes the input text token in a similar manner to Forward one
        x, z = xz.chunk(2, dim=-1)  # (B D)
        x = causal_conv1d_update(
            x,
            conv_state_pb,
            rearrange(self.conv1d_bwd.weight, "d 1 w -> d w"),
            self.conv1d_bwd.bias,
            self.activation,
        )
        x_db = self.x_proj_bwd(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj_bwd.weight)  # (B d_inner)
        A_bwd = -torch.exp(self.A_bwd_log.float())
        y_bwd = selective_state_update(
            ssm_state_pb,
            x,
            dt,
            A_bwd,
            B,
            C,
            self.D_bwd,
            z=z,
            dt_bias=self.dt_proj_bwd.bias,
            dt_softplus=True,
        )

        out = self.out_proj(y + y_bwd)
        return out.unsqueeze(1), conv_state, ssm_state, conv_state_pb, ssm_state_pb

    def allocate_inference_cache(
        self, batch_size, max_seqlen=None, dtype=None, **kwargs
    ):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        if not self.bidirectional:
            return conv_state, ssm_state

        else:
            conv_state_pb = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=device,
                dtype=conv_dtype,
            )
            ssm_state_pb = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=device,
                dtype=ssm_dtype,
            )
            return conv_state, ssm_state, conv_state_pb, ssm_state_pb

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def _get_states_from_cache_pb(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            conv_state_pb = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state_pb = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
                conv_state_pb,
                ssm_state_pb,
            )
        else:
            assert len(inference_params.key_value_memory_dict[self.layer_idx]) == 4, (
                len(inference_params.key_value_memory_dict[self.layer_idx])
            )
            conv_state, ssm_state, conv_state_pb, ssm_state_pb = (
                inference_params.key_value_memory_dict[self.layer_idx]
            )
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
                conv_state_pb.zero_()
                ssm_state_pb._zero_()
        return conv_state, ssm_state, conv_state_pb, ssm_state_pb


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), (
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        inference_params=None,
        flip_fn=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            mask: the mask for the input sequence (optional).
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(
            hidden_states, mask=mask, inference_params=inference_params, flip_fn=flip_fn
        )
        return hidden_states, residual

    def allocate_inference_cache(
        self, batch_size, max_seqlen=None, dtype=None, **kwargs
    ):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
