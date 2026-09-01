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

The ``LoRALayer`` / ``Embedding`` / ``Linear`` classes follow the API of
microsoft/LoRA (MIT License). The other backends follow the reference
implementations from the corresponding papers.
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
    """DoRA: weight-decomposed low-rank adaptation (Liu et al., 2024)."""

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
        self.m_initialized = False
        self.weight_m_wdecomp = nn.Parameter(torch.ones((out_features, 1)))
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def apply_m(self):
        if not self.m_initialized:
            self.weight_m_wdecomp.data = (
                torch.linalg.norm(self.weight, dim=1).unsqueeze(1).detach()
            )
            self.m_initialized = True

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # No-op: forward recomputes the decomposition each call, so
                # leaving self.weight.data unchanged returns to train mode.
                self.merged = False
        else:
            # Merging needs the magnitude vector from apply_m(), which is
            # initialized lazily on the first forward (so it captures the
            # pretrained weight, not the constructor placeholder). Skip the
            # merge until then; forward computes the unmerged path anyway.
            if self.merge_weights and not self.merged and self.m_initialized:
                new_weight_v = self.weight + (self.lora_B @ self.lora_A) * self.scaling
                norm_scale = (
                    self.weight_m_wdecomp.transpose(0, 1).view(-1)
                    / (torch.linalg.norm(new_weight_v, dim=1)).detach()
                )
                self.weight.data = (
                    self.weight
                    + (
                        (norm_scale - 1) * self.weight.T
                        + norm_scale
                        * (self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1))
                        * self.scaling
                    ).T
                )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        self.apply_m()
        if self.r > 0 and not self.merged:
            new_weight_v = self.weight + (self.lora_B @ self.lora_A) * self.scaling
            norm_scale = (
                self.weight_m_wdecomp.transpose(0, 1).view(-1)
                / (torch.linalg.norm(new_weight_v, dim=1)).detach()
            )
            org_result = F.linear(x, T(self.weight), bias=self.bias)
            dropout_x = self.lora_dropout(x)
            result = org_result + (
                (norm_scale - 1) * F.linear(dropout_x, T(self.weight))
            ).to(org_result.dtype)
            result += (
                norm_scale
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
        self._pissa_factorize()

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
    """SVFT: train a banded matrix in the SVD basis (Lingam et al., 2024)."""

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
        if mode:
            if self.merge_weights and self.merged:
                self.merged = False
        else:
            # Merging needs u/s_pre/v from apply_svd(), which is initialized
            # lazily on the first forward (so it factorizes the pretrained
            # weight, not the constructor placeholder). Skip the merge until
            # then; forward computes the unmerged path anyway.
            if self.merge_weights and not self.merged and bool(self.svd_initialized):
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
        self.gate.data = torch.tensor([0.0], device=s.device)
        nn.init.kaiming_uniform_(self.s[None, :])
        self.s.squeeze()
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

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                self.merged = False
        else:
            # Merging needs u/s_pre/v from apply_svd(), which is initialized
            # lazily on the first forward (so it factorizes the pretrained
            # weight, not the constructor placeholder). Skip the merge until
            # then; forward computes the unmerged path anyway.
            if self.merge_weights and not self.merged and bool(self.svd_initialized):
                sigma = self.get_sigma()
                if self.out_features >= self.in_features:
                    self.weight.data = T(
                        self.u @ torch.diag(sigma) @ self.apply_rotation()
                    )
                else:
                    self.weight.data = T(
                        (self.u @ torch.diag(sigma) @ self.apply_rotation()).T
                    )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        self.apply_svd()
        sigma = self.get_sigma()
        if not self.merged:
            if self.out_features >= self.in_features:
                return F.linear(
                    x,
                    T(self.u @ torch.diag(sigma) @ self.apply_rotation()),
                    bias=self.bias,
                )
            return F.linear(
                x,
                T((self.u @ torch.diag(sigma) @ self.apply_rotation()).T),
                bias=self.bias,
            )
        return F.linear(x, T(self.weight), bias=self.bias)
