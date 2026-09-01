# Portions of this file (the LoRALayer, Embedding and Linear classes) are
# adapted from microsoft/LoRA (https://github.com/microsoft/LoRA):
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License (MIT).
# The DoRA / PiSSA / SVFT / SSVD backends are implemented from the
# respective papers (see the references below).
"""Parameter-efficient fine-tuning (PEFT) layers used by ESPnet.

Self-contained PyTorch implementations of LoRA and its variants, ported into
ESPnet so the framework does not depend on an external ``loralib`` install.

Backends provided:
    * ``LoRALayer``    -- common base mixin (rank, alpha, dropout, merge flag).
    * ``Embedding``    -- LoRA-adapted ``nn.Embedding``.
    * ``Linear``       -- vanilla LoRA (Hu et al., 2021).
    * ``DoraLinear``   -- DoRA: Weight-decomposed LoRA (Liu et al., 2024).
    * ``PiSSALinear``  -- PiSSA: Principal Singular value adaptation
                         (Meng et al., 2024).
    * ``SVFTLinear``   -- Singular Vector Fine-Tuning (Lingam et al., 2024).
    * ``SSVDLinear``   -- Structured SVD (Wang et al., ASRU 2025).

References:
    LoRA   : https://arxiv.org/abs/2106.09685
    DoRA   : https://arxiv.org/abs/2402.09353
    PiSSA  : https://arxiv.org/abs/2404.02948
    SVFT   : https://arxiv.org/abs/2405.19597
    SSVD   : https://arxiv.org/abs/2509.02830

The ``LoRALayer`` / ``Embedding`` / ``Linear`` classes are adapted from
microsoft/LoRA (MIT License). The DoRA / PiSSA / SVFT / SSVD backends are
implemented from the corresponding papers.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    """Mixin holding the shared LoRA hyper-parameters."""

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    """LoRA adapter for ``nn.Embedding``."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x,
                self.lora_A.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    """Vanilla LoRA adapter for ``nn.Linear`` (Hu et al., 2021)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            return result
        return F.linear(x, T(self.weight), bias=self.bias)


class DoraLinear(nn.Linear, LoRALayer):
    """DoRA: weight-decomposed low-rank adaptation (Liu et al., 2024).

    Implemented from the paper (https://arxiv.org/abs/2402.09353): the
    weight is decomposed as ``W = m * V / ||V||_c`` with a trainable
    magnitude vector ``m`` and a LoRA update on the direction ``V``; the
    norm is detached from the gradient as described in the paper.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        if r <= 0:
            raise ValueError("DoraLinear requires r > 0.")
        # Persistent: at inference the checkpoint already holds the *trained*
        # magnitude, so apply_m() must not overwrite it with the row norms of
        # the loaded weight. A plain Python bool would not survive the
        # checkpoint round trip.
        self.register_buffer("m_initialized", torch.tensor(False, dtype=torch.bool))
        self.lora_m = nn.Parameter(torch.ones((out_features, 1)))
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def apply_m(self):
        if not bool(self.m_initialized):
            self.lora_m.data = (
                torch.linalg.norm(self.weight, dim=1).unsqueeze(1).detach()
            )
            self.m_initialized.fill_(True)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Switch train/eval mode. DoRA never merges into ``self.weight``.

        The DoRA update is ``m * (W + BA) / ||W + BA||_row``. Unlike the
        additive LoRA/PiSSA updates it cannot be inverted from the adapter
        parameters alone -- ``||W + BA||_row`` is lost once the merged weight
        overwrites ``W`` -- so a merge here would destroy the pretrained
        weight on the very first ``model.eval()`` of the training loop (ESPnet
        validates after every epoch) and corrupt it further on every
        subsequent epoch. ``forward`` recomputes the decomposition on each
        call anyway, so the only cost of never merging is one extra rank-``r``
        matmul at inference.
        """
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        self.apply_m()
        if self.r > 0 and not self.merged:
            v_adapted = self.weight + (self.lora_B @ self.lora_A) * self.scaling
            m_over_norm = (
                self.lora_m.transpose(0, 1).view(-1)
                / (torch.linalg.norm(v_adapted, dim=1)).detach()
            )
            org_result = F.linear(x, T(self.weight), bias=self.bias)
            dropout_x = self.lora_dropout(x)
            result = org_result + (
                (m_over_norm - 1) * F.linear(dropout_x, T(self.weight))
            ).to(org_result.dtype)
            result += (
                m_over_norm
                * (
                    self.lora_dropout(x)
                    @ self.lora_A.transpose(0, 1)
                    @ self.lora_B.transpose(0, 1)
                )
            ) * self.scaling
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
        return result


class PiSSALinear(nn.Linear, LoRALayer):
    """PiSSA: Principal Singular value adaptation (Meng et al., 2024).

    Implemented from the paper (https://arxiv.org/abs/2404.02948).

    Trains the top-``r`` SVD factors of the pretrained weight while keeping
    the rest of the spectrum frozen. The frozen ``A0/B0`` buffers store the
    original factors so the update is exact: at init the effective weight
    equals the pretrained weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        if r <= 0:
            raise ValueError("PiSSALinear requires r > 0.")
        self.fan_in_fan_out = fan_in_fan_out
        self.register_buffer("pissa_initialized", torch.tensor(False, dtype=torch.bool))

        self.lora_A = nn.Parameter(
            self.weight.new_zeros((r, in_features)), requires_grad=True
        )
        self.lora_B = nn.Parameter(
            self.weight.new_zeros((out_features, r)), requires_grad=True
        )
        self.scaling = self.lora_alpha / self.r
        self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.register_buffer("A0", torch.empty(r, in_features))
        self.register_buffer("B0", torch.empty(out_features, r))
        # NOTE: no eager _pissa_factorize() here. `replace_module` copies the
        # pretrained weight in *after* construction, and `init_param` is loaded
        # later still (see `AbsTask.build_model`), so factorizing now would
        # take the SVD of the random `nn.Linear.reset_parameters` weight and
        # PiSSA would train a random -- not principal -- subspace.

    def _pissa_factorize(self):
        if bool(self.pissa_initialized):
            return
        U, S, Vh = torch.linalg.svd(self.weight, full_matrices=False)
        U_r = U[:, : self.r]
        S_r = S[: self.r]
        V_r = Vh[: self.r, :].T
        sqrtS = torch.sqrt(S_r)
        self.lora_A.data = sqrtS.unsqueeze(1) * V_r.T
        self.lora_B.data = U_r * sqrtS.unsqueeze(0)
        self.A0.data = sqrtS.unsqueeze(1) * V_r.T
        self.B0.data = U_r * sqrtS.unsqueeze(0)
        self.pissa_initialized.fill_(True)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

        if not bool(self.pissa_initialized):
            # `create_lora_adapter` calls `model.eval()` right after module
            # replacement, before `init_param` is loaded, so A0/B0 are still
            # empty here. Track the flag only -- checkpoints are saved from
            # eval (merged) state, so `merged` must end up True exactly as it
            # does for vanilla LoRA -- but leave the weight alone.
            self.merged = not mode
            return

        A = torch.cat([self.lora_A, self.A0], dim=0)
        B = torch.cat([self.lora_B, -self.B0], dim=1)

        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(B @ A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(B @ A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        self._pissa_factorize()
        A = torch.cat([self.lora_A, self.A0], dim=0)
        B = torch.cat([self.lora_B, -self.B0], dim=1)
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)
            ) * self.scaling
            return result
        return F.linear(x, T(self.weight), bias=self.bias)


class SVFTLinear(nn.Linear, LoRALayer):
    """SVFT: train a banded matrix in the SVD basis (Lingam et al., 2024).

    Implemented from the paper (https://arxiv.org/abs/2405.19597).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        off_diag: Optional[int] = None,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        self.fan_in_fan_out = fan_in_fan_out
        self.register_buffer("svd_initialized", torch.tensor(False, dtype=torch.bool))

        self.r_svft = min(out_features, in_features)
        self.off_diag = r if off_diag is None else off_diag
        if self.off_diag < 0:
            raise ValueError(f"off_diag must be >= 0, got {self.off_diag}.")

        row_idx = []
        col_idx = []
        for d in range(-self.off_diag, self.off_diag + 1):
            i = torch.arange(self.r_svft - abs(d))
            j = i + d
            if d < 0:
                i, j = j, i
            row_idx.append(i)
            col_idx.append(j)
        self.register_buffer("band_row", torch.cat(row_idx))
        self.register_buffer("band_col", torch.cat(col_idx))
        self.num_banded_params = len(self.band_row)

        self.m_entries = nn.Parameter(torch.zeros(self.num_banded_params))
        self.register_buffer("u", torch.empty(out_features, self.r_svft))
        self.register_buffer("v", torch.empty(self.r_svft, in_features))
        self.register_buffer("s_pre", torch.empty(self.r_svft))
        self.gate = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.weight.requires_grad = False
        self.reset_parameters()

    def apply_svd(self):
        if not bool(self.svd_initialized):
            u, s, v = torch.linalg.svd(self.weight, full_matrices=False)
            self.u.data = u.clone().detach().contiguous()
            self.v.data = v.clone().detach().contiguous()
            self.s_pre.data = s.clone().detach().contiguous()
            self.svd_initialized.fill_(True)

    def construct_M(self):
        M = torch.zeros(self.r_svft, self.r_svft, device=self.m_entries.device)
        M[self.band_row, self.band_col] = self.m_entries
        return M

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

        if not bool(self.svd_initialized):
            # `create_lora_adapter` calls `model.eval()` right after module
            # replacement, before `init_param` is loaded, so u/s_pre/v are
            # still `torch.empty` here. Track the flag only -- checkpoints are
            # saved from eval (merged) state, so `merged` must end up True
            # exactly as it does for vanilla LoRA -- but leave the weight
            # alone.
            self.merged = not mode
            return

        if not self.merge_weights:
            return
        if mode:
            if self.merged:
                # Restore the pretrained weight: it is exactly the frozen
                # factorization u @ diag(s_pre) @ v.
                self.weight.data = T(self.u @ torch.diag(self.s_pre) @ self.v)
                self.merged = False
        else:
            if not self.merged:
                M = self.construct_M() * torch.sigmoid(self.gate)
                self.weight.data = T(self.u @ (torch.diag(self.s_pre) + M) @ self.v)
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        self.apply_svd()
        if not self.merged:
            M = self.construct_M() * torch.sigmoid(self.gate)
            return F.linear(
                x, T(self.u @ (torch.diag(self.s_pre) + M) @ self.v), bias=self.bias
            )
        return F.linear(x, T(self.weight), bias=self.bias)


class SSVDLinear(nn.Linear, LoRALayer):
    """SSVD: Structured SVD (Wang et al., ASRU 2025).

    Implemented from the paper (https://arxiv.org/abs/2509.02830).

    The pretrained weight is factorised as ``W = U diag(s_pre) V``. SSVD
    trains a trainable singular-value delta ``s`` (top-``k`` entries) plus a
    Cayley-style rotation of the top-``k`` rows of ``V``. ``k`` is set either
    via ``rotation_ratio`` (fraction of ``min(in,out)``) or via ``r``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        rotation_ratio: Optional[float] = None,
        off_diag: int = 0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        self.fan_in_fan_out = fan_in_fan_out
        self.register_buffer("svd_initialized", torch.tensor(False, dtype=torch.bool))
        self.in_features = in_features
        self.out_features = out_features

        self.r_ssvd = min(out_features, in_features)
        if rotation_ratio is not None:
            if not (0.0 < rotation_ratio <= 1.0):
                raise ValueError(
                    f"rotation_ratio must be in (0, 1], got {rotation_ratio}."
                )
            self.k_trainable = int(self.r_ssvd * rotation_ratio)
            if self.k_trainable < 1:
                raise ValueError(
                    f"rotation_ratio={rotation_ratio} yields k_trainable=0 for "
                    f"r_ssvd={self.r_ssvd}. Increase rotation_ratio."
                )
        elif r > 0:
            self.k_trainable = r
        else:
            raise ValueError(
                "SSVDLinear requires rotation_ratio or r > 0 to determine "
                "k_trainable."
            )
        if self.k_trainable > self.r_ssvd:
            raise ValueError(
                f"k_trainable ({self.k_trainable}) must be <= "
                f"min(out_features, in_features) ({self.r_ssvd})."
            )

        if self.out_features >= self.in_features:
            self.register_buffer("u", torch.empty(out_features, self.r_ssvd))
            self.register_buffer("v", torch.empty(self.r_ssvd, in_features))
        else:
            self.register_buffer("u", torch.empty(in_features, self.r_ssvd))
            self.register_buffer("v", torch.empty(self.r_ssvd, out_features))

        self.register_buffer("s_pre", torch.empty(self.r_ssvd))
        self.s = nn.Parameter(torch.zeros(self.k_trainable))
        self.gate = nn.Parameter(torch.empty(1).zero_(), requires_grad=True)
        self.K_vec = nn.Parameter(
            torch.zeros((self.k_trainable * (self.k_trainable - 1)) // 2),
            requires_grad=True,
        )
        self.register_buffer(
            "K_triu_idx",
            torch.triu_indices(self.k_trainable, self.k_trainable, offset=1),
        )

        self.weight.requires_grad = False
        self.reset_parameters()

    def apply_svd(self):
        if bool(self.svd_initialized):
            return
        if self.out_features >= self.in_features:
            u, s, v = torch.linalg.svd(self.weight, full_matrices=False)
        else:
            u, s, v = torch.linalg.svd(self.weight.T, full_matrices=False)
        self.u.data = u.detach()
        self.v.data = v.detach()
        self.s_pre.data = s.detach()
        # `s` stays zero-initialized: get_sigma() applies `s * sigmoid(gate)`
        # and sigmoid(0) == 0.5, so a random `s` would move the layer away
        # from the pretrained weight before a single training step.
        self.svd_initialized.fill_(True)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def get_sigma(self):
        delta = F.pad(
            self.s * F.sigmoid(self.gate), (0, self.r_ssvd - self.k_trainable)
        )
        return self.s_pre + delta

    def apply_rotation(self):
        Y = self.v.clone()
        k = self.k_trainable
        idx = self.K_triu_idx
        device = Y.device
        dtype = Y.dtype

        # Build the skew-symmetric step of the Cayley transform.
        A = torch.eye(k, device=device, dtype=dtype)
        A[idx[0], idx[1]] -= 2 * self.K_vec
        A[idx[1], idx[0]] += 2 * self.K_vec

        Y_top = Y[:k, :]
        Y_rest = Y[k:, :]
        Y_top_new = A @ Y_top
        return torch.cat([Y_top_new, Y_rest], dim=0)

    def _compose(self, sigma: torch.Tensor, rotated_v: torch.Tensor):
        """Rebuild the (out_features, in_features) weight from the factors."""
        w = self.u @ torch.diag(sigma) @ rotated_v
        if self.out_features < self.in_features:
            # apply_svd() factorized W.T in this orientation.
            w = w.T
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

        if not bool(self.svd_initialized):
            # `create_lora_adapter` calls `model.eval()` right after module
            # replacement, before `init_param` is loaded, so u/s_pre/v are
            # still `torch.empty` here. Track the flag only -- checkpoints are
            # saved from eval (merged) state, so `merged` must end up True
            # exactly as it does for vanilla LoRA -- but leave the weight
            # alone.
            self.merged = not mode
            return

        if not self.merge_weights:
            return
        if mode:
            if self.merged:
                # Restore the pretrained weight: it is exactly the frozen
                # factorization u @ diag(s_pre) @ v (no rotation).
                self.weight.data = self._compose(self.s_pre, self.v)
                self.merged = False
        else:
            if not self.merged:
                self.weight.data = self._compose(
                    self.get_sigma(), self.apply_rotation()
                )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        self.apply_svd()
        if not self.merged:
            return F.linear(
                x,
                self._compose(self.get_sigma(), self.apply_rotation()),
                bias=self.bias,
            )
        return F.linear(x, T(self.weight), bias=self.bias)
