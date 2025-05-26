# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code is modified from the original script inspired by Vision Mamba [1] and Mamba-ND [2]
#
# [1] Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
#     https://arxiv.org/abs/2401.09417
# [2] Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data
#     https://arxiv.org/abs/2402.05892


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from espnet2.asr.state_spaces.mamba.ops.triton.selective_state_update import (
        selective_state_update,
    )
except ImportError:
    selective_state_update = None

from espnet2.asr.state_spaces.mamba.distributed.distributed_utils import (
    all_reduce,
    reduce_scatter,
)
from espnet2.asr.state_spaces.mamba.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from espnet2.asr.state_spaces.mamba.ops.triton.layernorm_gated import (
    RMSNorm as RMSNormGated,
)
from espnet2.asr.state_spaces.mamba.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)
from espnet2.asr.state_spaces.s4 import OptimModule


class MambaV2(OptimModule):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        projection_dropout_rate=0.0,
        bidirectional: bool = False,
        bidirectional_mask: bool = True,
        prefix_bidir: bool = False,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.bidirectional = bidirectional
        self.bidirectional_mask = bidirectional_mask
        self.prefix_bidir = prefix_bidir

        self.dropout = nn.Dropout(projection_dropout_rate)

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(
                self.d_model, d_in_proj, bias=bias, **factory_kwargs
            )
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model,
                d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        if self.process_group is None:
            self.out_proj = nn.Linear(
                self.d_inner, self.d_model, bias=bias, **factory_kwargs
            )
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size,
                self.d_model,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        if self.bidirectional:
            self.conv1d_bwd = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(
                    self.conv1d_bwd.weight, -self.conv_init, self.conv_init
                )
            self.dt_bias_bwd = nn.Parameter(inv_dt)
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias_bwd._no_weight_decay = True

            assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
            A_bwd = torch.empty(
                self.nheads, dtype=torch.float32, device=device
            ).uniform_(*A_init_range)
            A_log_bwd = torch.log(A_bwd).to(dtype=dtype)
            self.A_log_bwd = nn.Parameter(A_log_bwd)
            self.A_log_bwd._no_weight_decay = True

            # D "skip" parameter
            self.D_bwd = nn.Parameter(
                torch.ones(
                    self.d_ssm if self.D_has_hdim else self.nheads, device=device
                )
            )
            self.D_bwd._no_weight_decay = True

    def forward(
        self,
        u,
        seqlen=None,
        seq_idx=None,
        inference_params=None,
        mask=None,
        flip_fn=None,
        **kwargs,
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        if self.bidirectional:
            return self.forward_pb(
                u,
                seqlen=seqlen,
                seq_idx=seq_idx,
                inference_params=inference_params,
                mask=mask,
                flip_fn=flip_fn,
                **kwargs,
            )

        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
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
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.dropout(
            self.in_proj(u)
        )  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (
                zxbcdt.shape[-1]
                - 2 * self.d_ssm
                - 2 * self.ngroups * self.d_state
                - self.nheads
            ) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [
                    d_mlp,
                    d_mlp,
                    self.d_ssm,
                    self.d_ssm + 2 * self.ngroups * self.d_state,
                    self.nheads,
                ],
                dim=-1,
            )
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(
                    F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                )  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            x, B, C = torch.split(
                xBC,
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1,
            )
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
                if not self.rmsnorm
                else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return self.dropout(out)

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, (
            "Only support decoding with 1 token at a time for now"
        )
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, (
                "Only support ngroups=1 for this inference code path"
            )
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(
                dtype=torch.float32
            )
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state,
                x_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def forward_pb(
        self,
        u,
        seqlen=None,
        seq_idx=None,
        inference_params=None,
        mask=None,
        flip_fn=None,
        **kwargs,
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        assert self.bidirectional
        assert not self.use_mem_eff_path, (
            "Currently not supported for bidirectional Mamba"
        )
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state, conv_state_pb, ssm_state_pb = None, None, None, None
        if inference_params is not None:
            conv_state, ssm_state, conv_state_pb, ssm_state_pb = (
                self._get_states_from_cache_pb(inference_params, batch)
            )
            if inference_params.seqlen_offset > 0:
                out, _, _, _, _ = self.bi_mamba_forward_step(
                    u, conv_state, ssm_state, conv_state_pb, ssm_state_pb, mask
                )
                return out

        zxbcdt = self.dropout(
            self.in_proj(u)
        )  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        A_bwd = -torch.exp(self.A_log_bwd)  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)

        else:
            d_mlp = (
                zxbcdt.shape[-1]
                - 2 * self.d_ssm
                - 2 * self.ngroups * self.d_state
                - self.nheads
            ) // 2
            assert d_mlp == 0, "Currently not supported for bidirectional Mamba"
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [
                    d_mlp,
                    d_mlp,
                    self.d_ssm,
                    self.d_ssm + 2 * self.ngroups * self.d_state,
                    self.nheads,
                ],
                dim=-1,
            )

            # forward path
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(
                    F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                )  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            x, B, C = torch.split(
                xBC,
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1,
            )
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
                if not self.rmsnorm
                else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")

            # backward path
            # NOTE (Y. Masuyama): The mask is applied just before the backward model
            # This is for the case masking out not only padded tokens but also text tokens
            if self.bidirectional_mask:
                zxbcdt = zxbcdt * mask.transpose(-2, -1)

            if flip_fn is None:
                _zxbcdt = torch.flip(zxbcdt, dims=[-2])
            else:
                _zxbcdt = flip_fn[0](zxbcdt.transpose(-2, -1)).transpose(-2, -1)

            z0, x0, z, xBC, dt = torch.split(
                _zxbcdt,
                [
                    d_mlp,
                    d_mlp,
                    self.d_ssm,
                    self.d_ssm + 2 * self.ngroups * self.d_state,
                    self.nheads,
                ],
                dim=-1,
            )

            if conv_state_pb is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state_pb.copy_(
                    F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                )  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d_bwd(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d_bwd.weight, "d 1 w -> d w"),
                    bias=self.conv1d_bwd.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            x, B, C = torch.split(
                xBC,
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1,
            )
            y_bwd = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A_bwd,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D_bwd, "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.D_bwd,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
                if not self.rmsnorm
                else None,
                dt_bias=self.dt_bias_bwd,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state_pb is not None,
            )
            if ssm_state_pb is not None:
                y_bwd, last_state = y_bwd
                ssm_state_pb.copy_(last_state)
            y_bwd = rearrange(y_bwd, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y_bwd = self.norm(y_bwd, z)
            if d_mlp > 0:
                y_bwd = torch.cat([F.silu(z0) * x0, y_bwd], dim=-1)
            if seqlen_og is not None:
                y_bwd = rearrange(y_bwd, "b l d -> (b l) d")

            if flip_fn is None:
                y_bwd = torch.flip(y_bwd, dims=[-2])
            else:
                y_bwd = flip_fn[1](y_bwd.transpose(-2, -1)).transpose(-2, -1)

            out = self.out_proj(y + y_bwd)
        return self.dropout(out)

    def bi_mamba_forward_step(
        self, hidden_states, conv_state, ssm_state, conv_state_pb, ssm_state_pb, mask
    ):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, (
            "Only support decoding with 1 token at a time for now"
        )
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, (
                "Only support ngroups=1 for this inference code path"
            )
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(
                dtype=torch.float32
            )
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state,
                x_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        if not mask:
            out = self.out_proj(y)
            return out.unsqueeze(1), conv_state, ssm_state, conv_state_pb, ssm_state_pb

        # backward path
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state_pb.copy_(
                torch.roll(conv_state_pb, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state_pb[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state_pb * rearrange(self.conv1d_bwd.weight, "d 1 w -> d w"),
                dim=-1,
            )  # (B D)
            if self.conv1d_bwd.bias is not None:
                xBC = xBC + self.conv1d_bwd.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state_pb,
                rearrange(self.conv1d_bwd.weight, "d 1 w -> d w"),
                self.conv1d_bwd.bias,
                self.activation,
            )

        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log_bwd.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, (
                "Only support ngroups=1 for this inference code path"
            )
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias_bwd.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state_pb.copy_(ssm_state_pb * rearrange(dA, "b h -> b h 1 1") + dBx)
            y_bwd = torch.einsum("bhpn,bn->bhp", ssm_state_pb.to(dtype), C)
            y_bwd = y_bwd + rearrange(self.D_bwd.to(dtype), "h -> h 1") * x
            y_bwd = rearrange(y_bwd, "b h p -> b (h p)")
            if not self.rmsnorm:
                y_bwd = y_bwd * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(
                dtype=torch.float32
            )
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias_bwd, "h -> h p", p=self.headdim)
            D = repeat(self.D_bwd, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y_bwd = selective_state_update(
                ssm_state_pb,
                x_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            y_bwd = rearrange(y_bwd, "b h p -> b (h p)")
        if self.rmsnorm:
            y_bwd = self.norm(y_bwd, z)
        if d_mlp > 0:
            y_bwd = torch.cat([F.silu(z0) * x0, y_bwd], dim=-1)

        out = self.out_proj(y + y_bwd)
        return out.unsqueeze(1), conv_state, ssm_state, conv_state_pb, ssm_state_pb

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.conv1d.weight.shape[0],
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        if not self.bidirectional:
            return conv_state, ssm_state

        else:
            conv_state_pb = torch.zeros(
                batch_size,
                self.conv1d_bwd.weight.shape[0],
                self.d_conv,
                device=device,
                dtype=conv_dtype,
            )
            ssm_state_pb = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
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
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
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
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            conv_state_pb = torch.zeros(
                batch_size,
                self.conv1d_bwd.weight.shape[0],
                self.d_conv,
                device=self.conv1d_bwd.weight.device,
                dtype=self.conv1d_bwd.weight.dtype,
            )
            ssm_state_pb = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
                conv_state_pb,
                ssm_state_pb,
            )
        else:
            conv_state, ssm_state, conv_state_pb, ssm_state_pb = (
                inference_params.key_value_memory_dict[self.layer_idx]
            )

        return conv_state, ssm_state, conv_state_pb, ssm_state_pb
