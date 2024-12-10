"""Multi-head Damped Exponential Moving Average (EMA) module for MEGA block.

Based/modified from https://github.com/facebookresearch/mega/blob/main/fairseq/modules/moving_average_gated_attention.py

Most variables are renamed according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/mega/modeling_mega.py.

"""  # noqa

import math
from typing import Dict, Optional, Tuple, Union

import torch


class MultiHeadDampedEMA(torch.nn.Module):
    """
    Multi-head Damped Exponential Moving Average (EMA) module for MEGA block.

    This module implements a multi-head damped EMA mechanism, which is commonly
    used in attention mechanisms for sequence processing. The design is based
    on modifications from the Fairseq library and has been adapted to follow
    the conventions set forth in the Hugging Face Transformers library.

    Attributes:
        damping_factor (torch.nn.Parameter): Parameter representing the damping
            factor for the EMA.
        decay_factor (torch.nn.Parameter): Parameter representing the decay factor
            for the EMA.
        ema_expansion_matrix (torch.nn.Parameter): Parameter for the EMA expansion
            matrix.
        kernel_projection_matrix (torch.nn.Parameter): Parameter for the kernel
            projection matrix.
        residual_weight (torch.nn.Parameter): Parameter representing the residual
            weight.
        scaling (float): Scaling factor computed as the square root of the
            inverse of the number of heads.
        truncation_length (Optional[int]): Maximum length for truncation, if
            specified.
        activation (torch.nn.Module): Activation function to apply to the output.
        num_heads (int): Number of attention heads used in the module.

    Args:
        size (int): The size of the module.
        num_heads (int, optional): The number of attention heads. Defaults to 4.
        activation (torch.nn.Module, optional): The activation function type.
            Defaults to ReLU.
        truncation_length (Optional[int], optional): The maximum length for
            truncation. Defaults to None.

    Examples:
        >>> ema = MultiHeadDampedEMA(size=128, num_heads=4)
        >>> input_tensor = torch.randn(10, 32, 128)  # (L, B, D)
        >>> output, new_state = ema(input_tensor)
        >>> print(output.shape)  # Output shape will be (B, L, D)

    Raises:
        ValueError: If the input tensor does not have the expected shape.

    Note:
        The implementation includes methods for computing EMA coefficients,
        resetting parameters, and applying the EMA in a forward pass.

    Todo:
        - Extend functionality to support additional activation functions.
        - Implement better error handling for input tensor shapes.
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
        """
        Reset module parameters.

        This method initializes the parameters of the MultiHeadDampedEMA module
        using a normal distribution. It sets the damping and decay factors,
        the EMA expansion matrix, the kernel projection matrix, and the
        residual weight.

        Args:
            val: Initialization value for the parameters.
            std1: Standard deviation for the damping and decay factors.
            std2: Standard deviation for the kernel projection matrix
                  and residual weight.

        Examples:
            >>> ema = MultiHeadDampedEMA(size=10, num_heads=4)
            >>> ema.reset_parameters(val=0.1, std1=0.3, std2=0.5)

        Note:
            The parameters are initialized in-place, and this function
            does not return any value. Use this method to reinitialize
            the model parameters, especially before training.
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
        """
        Compute EMA coefficients.

        This method computes the damping factor and the previous timestep weight,
        which are essential for the exponential moving average (EMA) calculations.
        The damping factor represents the P-th order coefficient, while the
        previous timestep weight represents the Q-th order coefficient.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - damping_factor: Damping factor / P-th order coefficient.
                                Shape: (size, num_heads, 1)
                - prev_timestep_weight: Previous timestep weight / Q-th order coefficient.
                                        Shape: (size, num_heads, 1)

        Examples:
            >>> ema = MultiHeadDampedEMA(size=10, num_heads=4)
            >>> damping, prev_weight = ema.compute_ema_coefficients()
            >>> print(damping.shape)  # Output: torch.Size([10, 4, 1])
            >>> print(prev_weight.shape)  # Output: torch.Size([10, 4, 1])

        Note:
            The damping factor is computed using a sigmoid function applied to the
            `damping_factor` parameter, and the previous timestep weight is computed
            using the formula:
            `prev_timestep_weight = 1.0 - damping_factor * decay_factor`.
        """
        self._coeffs = None

        damping_factor = torch.sigmoid(self.damping_factor)
        decay_factor = torch.sigmoid(self.decay_factor)

        prev_timestep_weight = 1.0 - damping_factor * decay_factor

        return damping_factor, prev_timestep_weight

    def compute_ema_kernel(self, length: int) -> torch.Tensor:
        """
        Compute EMA kernel / Vandermonde product.

        This method calculates the Exponential Moving Average (EMA) kernel using
        the damped factors and the EMA expansion matrix. The resulting kernel
        represents the effect of applying the EMA over a sequence of specified
        length.

        Args:
            length: The sequence length for which to compute the EMA kernel.

        Returns:
            torch.Tensor: The EMA kernel / Vandermonde product, shaped
            (size, L), where 'size' corresponds to the module size and 'L'
            corresponds to the input sequence length.

        Examples:
            >>> ema_module = MultiHeadDampedEMA(size=10, num_heads=4)
            >>> kernel = ema_module.compute_ema_kernel(length=5)
            >>> print(kernel.shape)
            torch.Size([10, 5])

        Note:
            The EMA kernel is computed based on the current damping factors and
            expansion matrix. This is crucial for the operation of the EMA
            mechanism in the model.
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
        """
        Get EMA coefficients.

        This method retrieves the damping factor and the previous timestep weight
        coefficients used in the exponential moving average (EMA) calculations.
        The coefficients are computed using the sigmoid activation function applied
        to the damping and decay factors.

        If the coefficients have not been computed yet, this method will call
        `compute_ema_coefficients()` to generate them.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Damping factor / P-th order coefficient. Shape: (size, num_heads, 1)
                - Previous timestep weight / Q-th order coefficient. Shape:
                  (size, num_heads, 1)

        Examples:
            >>> ema_module = MultiHeadDampedEMA(size=128, num_heads=4)
            >>> damping, prev_weight = ema_module.get_ema_coefficients()
            >>> print(damping.shape)  # Output: torch.Size([128, 4, 1])
            >>> print(prev_weight.shape)  # Output: torch.Size([128, 4, 1])
        """
        if self._coeffs is None:
            self._coeffs = self.compute_ema_coefficients()

        return self._coeffs

    def ema_one_step(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform exponential moving average for a single step.

        This method computes the exponential moving average (EMA) for the given
        input tensor `x` at a single time step. It utilizes the current state
        of the EMA to update the new state and generate the output sequence.

        Args:
            x: MultiHeadDampedEMA input sequences. Shape: (B, D, 1), where B is the
               batch size and D is the dimension of the input.
            state: Optional; MultiHeadDampedEMA state from the previous step.
                   Shape: (B, D, num_heads). If not provided, the EMA is computed
                   without incorporating any prior state.

        Returns:
            out: MultiHeadDamped output sequences. Shape: (B, 1, D), representing
                 the output after applying the EMA to the input sequences.
            new_state: MultiHeadDampedEMA state for the current step. Shape:
                       (B, D, num_heads), which can be used in subsequent EMA
                       computations.

        Examples:
            >>> ema_module = MultiHeadDampedEMA(size=128, num_heads=4)
            >>> input_tensor = torch.rand(32, 128, 1)  # Batch of 32
            >>> initial_state = torch.zeros(32, 128, 4)  # Initial state
            >>> output, new_state = ema_module.ema_one_step(input_tensor, initial_state)

        Note:
            The output `out` is computed by applying the EMA to the input `x`
            and adding the contribution from the previous state if provided.
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
        """
        Compute multi-dimensional damped EMA.

        This method computes the multi-dimensional damped Exponential Moving
        Average (EMA) for the input sequences using the current state and
        optionally applies a mask to the input.

        Args:
            x: MultiHeadDampedEMA input sequence. Shape (L, B, D), where:
               - L is the sequence length,
               - B is the batch size,
               - D is the feature dimension.
            mask: Optional sequence mask. Shape (B, 1, L). A mask can be used
                  to ignore certain time steps in the input sequence.
            state: Optional MultiHeadDampedEMA state. Shape (B, D, num_heads),
                   where num_heads is the number of attention heads.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - x: MultiHeadDampedEMA output sequence. Shape (B, L, D).
                - new_state: MultiHeadDampedEMA state. Shape (B, D, num_heads)
                  or None if state was not provided.

        Examples:
            >>> model = MultiHeadDampedEMA(size=128, num_heads=4)
            >>> input_tensor = torch.randn(10, 32, 128)  # (L, B, D)
            >>> mask_tensor = torch.zeros(32, 1, 10)  # (B, 1, L)
            >>> output, new_state = model(input_tensor, mask=mask_tensor)

        Note:
            If `mask` is provided, the input tensor `x` will have masked
            elements set to zero before processing. If `state` is not provided,
            the method will compute the output without using any previous
            state information.
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
