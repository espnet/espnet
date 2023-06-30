"""Multi-head Damped Exponential Moving Average (EMA) module for MEGA block.

Based/modified from https://github.com/facebookresearch/mega/blob/main/fairseq/modules/moving_average_gated_attention.py

Most variables are renamed according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/mega/modeling_mega.py.

"""  # noqa

import math
from typing import Dict, Optional, Tuple, Union

import torch


class MultiHeadDampedEMA(torch.nn.Module):
    """MultiHeadDampedEMA module definition.

    Args:
        size: Module size.
        num_heads: Number of attention heads.
        activation: Activation function type.
        truncation_length: Maximum length for truncation.

    """

    def __init__(
        self,
        size: int,
        num_heads: int = 4,
        activation: torch.nn.Module = torch.nn.ReLU(),
        truncation_length: Optional[int] = None,
    ) -> None:
        """Construct an MultiHeadDampedEMA object."""
        super().__init__()

        self.damping_factor = torch.nn.Parameter(torch.Tensor(size, num_heads, 1))
        self.decay_factor = torch.nn.Parameter(torch.Tensor(size, num_heads, 1))

        self.ema_expansion_matrix = torch.nn.Parameter(torch.Tensor(size, num_heads, 1))
        self.kernel_projection_matrix = torch.nn.Parameter(
            torch.Tensor(size, num_heads)
        )

        self.residual_weight = torch.nn.Parameter(torch.Tensor(size))

        self.scaling = math.sqrt(1.0 / num_heads)
        self.truncation_length = truncation_length

        self.activation = activation

        self._kernel = None
        self._coeffs = None

        self.num_heads = num_heads

        self.reset_parameters()

    def reset_parameters(
        self, val: float = 0.0, std1: float = 0.2, std2: float = 1.0
    ) -> None:
        """Reset module parameters.

        Args:
            val: Initialization value.
            std1: Main standard deviation.
            std2: Secondary standard deviation.

        """
        with torch.no_grad():
            torch.nn.init.normal_(self.damping_factor, mean=val, std=std1)
            torch.nn.init.normal_(self.decay_factor, mean=val, std=std1)

            ema_exp_val = torch.ones(self.num_heads, 1)

            if self.num_heads > 1:
                idx = torch.tensor(list(range(1, self.num_heads, 2)))
                ema_exp_val.index_fill_(0, idx, -1.0)

            self.ema_expansion_matrix.normal_(mean=val, std=0.02).add_(ema_exp_val)

            torch.nn.init.normal_(self.kernel_projection_matrix, mean=val, std=std2)
            torch.nn.init.normal_(self.residual_weight, mean=val, std=std2)

    def compute_ema_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute EMA coefficients.

        Args:
            None

        Returns:
            damping_factor: Damping factor / P-th order coefficient.
                              (size, num_heads, 1)
            prev_timestep_weight: Previous timestep weight / Q-th order coefficient.
                                    (size, num_heads, 1)

        """
        self._coeffs = None

        damping_factor = torch.sigmoid(self.damping_factor)
        decay_factor = torch.sigmoid(self.decay_factor)

        prev_timestep_weight = 1.0 - damping_factor * decay_factor

        return damping_factor, prev_timestep_weight

    def compute_ema_kernel(self, length: int) -> torch.Tensor:
        """Compute EMA kernel / vandermonde product.

        Args:
            length: Sequence length.

        Returns:
            : EMA kernel / Vandermonde product. (size, L)

        """
        self._kernel = None

        damping_factor, prev_timestep_weight = self.compute_ema_coefficients()

        vander = torch.arange(length).to(damping_factor).view(1, 1, length) * torch.log(
            prev_timestep_weight
        )
        kernel = (damping_factor * self.ema_expansion_matrix) * torch.exp(vander)

        return torch.einsum(
            "dnl, dn -> dl", kernel, self.kernel_projection_matrix * self.scaling
        )

    def get_ema_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get EMA coefficients.

        Args:
            None

        Returns:
            : Damping factor / P-th order coefficient. (size, num_heads, 1)
            : Previous timestep weight / Q-th order coefficient. (size, num_heads, 1)

        """
        if self._coeffs is None:
            self._coeffs = self.compute_ema_coefficients()

        return self._coeffs

    def ema_one_step(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform exponential moving average for a single step.

        Args:
            x: MultiHeadDampedEMA input sequences. (B, D, 1)
            state: MultiHeadDampedEMA state. (B, D, num_heads)

        Returns:
            out: MultiHeadDamped output sequences. (B, 1, D)
            new_state: MultiHeadDampedEMA state. (B, D, num_heads)

        """
        damping_factor, prev_timestep_weight = self.get_ema_coefficients()

        new_state = (damping_factor * self.ema_expansion_matrix).squeeze(-1) * x

        if state is not None:
            new_state = new_state + prev_timestep_weight.squeeze(-1) * state

        out = torch.einsum(
            "bdn, dn -> bd", new_state, self.kernel_projection_matrix * self.scaling
        )

        return out.unsqueeze(0), new_state

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-dimensional damped EMA.

        Args:
            x: MultiHeadDampedEMA input sequence. (L, B, D)
            mask: Sequence mask. (B, 1, L)
            state: MultiHeadDampedEMA state. (B, D, num_heads)

        Returns:
            x: MultiHeadDampedEMA output sequence. (B, L, D)
            new_state: MultiHeadDampedEMA state. (B, D, num_heads)

        """
        length = x.size(0)

        residual = x * self.residual_weight

        x = x.permute(1, 2, 0)

        if mask is not None:
            x = x.masked_fill(mask, 0.0)

        if state is not None:
            ema_output, new_state = self.ema_one_step(x, state=state["ema_state"])
            ema_output = self.activation(ema_output + residual)

            return ema_output, new_state

        kernel = self.compute_ema_kernel(
            length
            if self.truncation_length is None
            else min(self.truncation_length, length)
        )

        input_fft = torch.fft.rfft(x.float(), n=(2 * length))
        kernel_fft = torch.fft.rfft(kernel.float(), n=(2 * length))

        ema_output = torch.fft.irfft((input_fft * kernel_fft), n=(2 * length))[
            ..., :length
        ]
        ema_output = ema_output.type_as(x)

        ema_output = self.activation(ema_output.permute(2, 0, 1) + residual)

        return ema_output, None
