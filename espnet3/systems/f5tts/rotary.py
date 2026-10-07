# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Self-contained rotary positional embedding for F5-TTS.

Adapted from the symbols F5 uses out of ``x_transformers``
(https://github.com/lucidrains/x-transformers, ``x_transformers/x_transformers.py``),
which is MIT-licensed; the notice above is that project's own, and it is a
different project from F5-TTS. Reproducing them here means the model no longer
depends on that package. Only the ``use_xpos=False`` path F5 exercises is
reproduced (``scale``
is always ``1.0``); ``einops`` calls are rewritten with plain torch ops but the
math/layout is identical.
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension pairwise, mapping ``(x1, x2)`` to ``(-x2, x1)``."""
    # einops: 'b ... (d r) -> b ... d r' (r=2); (-x2, x1); '... d r -> ... (d r)'
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], x.shape[-2] * 2)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, scale=1) -> torch.Tensor:
    """Apply rotary embeddings to ``t``, forced to fp32.

    Args:
        t: Queries or keys, ``[b, h, n, d]`` or ``[b, n, d]``.
        freqs: Rotation angles, ``[b, n, rot_dim]``.
        scale: xpos scale; always ``1`` on the path F5 uses.

    Returns:
        ``t`` rotated, cast back to its original dtype.
    """
    # x_transformers runs this under @autocast(enabled=False); keyed to the
    # input's device so MPS/CPU AMP get the same fp32 guarantee as CUDA.
    with torch.autocast(device_type=t.device.type, enabled=False):
        return _apply_rotary_pos_emb(t, freqs, scale)


def _apply_rotary_pos_emb(
    t: torch.Tensor, freqs: torch.Tensor, scale=1
) -> torch.Tensor:
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    if torch.is_tensor(scale):
        scale = scale[:, -seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)  # 'b n d -> b 1 n d'
        if torch.is_tensor(scale):
            scale = scale.unsqueeze(1)

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim=-1)
    return out.type(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding, the ``use_xpos=False`` path only."""

    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,
        base_rescale_factor=1.0,
    ):
        """Configure the rotation frequencies.

        Args:
            dim: Rotary dimension; must be even.
            use_xpos: Kept for signature compatibility. Only ``False`` is
                reproduced here, which is the path F5 exercises.
            scale_base: xpos scale base, unused when ``use_xpos`` is False.
            interpolation_factor: Divides positions to stretch the context.
            base: Frequency base before rescaling.
            base_rescale_factor: NTK-aware rescale; ``1.0`` leaves ``base`` as is.
        """
        super().__init__()
        # NTK-aware rescale (bloc97); factor 1.0 leaves base unchanged.
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward_from_seq_len(self, seq_len):
        """Return embeddings for positions ``0..seq_len - 1``."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    def forward(self, t, offset=0):
        """Return rotation angles for positions ``t``, computed in fp32."""
        # x_transformers runs this under @autocast(enabled=False): autocast
        # lowers einsum to bf16/fp16, which cannot represent positions > 256
        # exactly and silently corrupts phases in long utterances.
        with torch.autocast(device_type=t.device.type, enabled=False):
            return self._forward(t, offset=offset)

    def _forward(self, t, offset=0):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = t.unsqueeze(0)  # 'n -> 1 n'

        freqs = (
            torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
            / self.interpolation_factor
        )
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = freqs.reshape(*freqs.shape[:-2], freqs.shape[-2] * 2)

        if self.scale is None:
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** power.unsqueeze(-1)
        scale = torch.stack((scale, scale), dim=-1)
        scale = scale.reshape(*scale.shape[:-2], scale.shape[-2] * 2)
        return freqs, scale
