"""Attention (time mixing) modules for RWKV block.

Based/Modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py.

Some variables are renamed according to
https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py.

"""

import math
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional, Tuple

import torch

wkv_kernel = None


class WKVLinearAttention(torch.autograd.Function):
    """
    WKVLinearAttention function definition.

    This class implements a linear attention mechanism based on the RWKV model,
    which allows for efficient computation of attention scores. The forward and
    backward methods utilize custom CUDA kernels for performance.

    The implementation is based on the RWKV architecture and is inspired by
    previous works available in the RWKV-LM GitHub repository and Hugging Face
    Transformers library.

    Attributes:
        None

    Args:
        ctx: The context object used to store information for backward pass.
        time_decay: Channel-wise time decay vector. Shape: (D_att).
        time_first: Channel-wise time first vector. Shape: (D_att).
        key: Key tensor. Shape: (B, U, D_att).
        value: Value tensor. Shape: (B, U, D_att).

    Returns:
        out: Weighted Key-Value tensor. Shape: (B, U, D_att).

    Yields:
        None

    Raises:
        AssertionError: If the length of key exceeds the context size or if the
                        batch size multiplied by dimension is not a multiple of
                        the minimum of dimension and 32.

    Examples:
        >>> time_decay = torch.randn(D_att)
        >>> time_first = torch.randn(D_att)
        >>> key = torch.randn(B, U, D_att)
        >>> value = torch.randn(B, U, D_att)
        >>> output = WKVLinearAttention.apply(time_decay, time_first, key, value)
    """

    @staticmethod
    def forward(
        ctx,
        time_decay: torch.Tensor,
        time_first: torch.Tensor,
        key: torch.Tensor,
        value: torch.tensor,
    ) -> torch.Tensor:
        """
        WKVLinearAttention function forward pass.

        This method computes the forward pass for the WKV linear attention
        mechanism, which involves applying time decay and time first vectors
        to key and value tensors to produce a weighted output tensor.

        Args:
            ctx: The context object to store information for the backward pass.
            time_decay: Channel-wise time decay vector of shape (D_att).
            time_first: Channel-wise time first vector of shape (D_att).
            key: Key tensor of shape (B, U, D_att).
            value: Value tensor of shape (B, U, D_att).

        Returns:
            out: Weighted Key-Value tensor of shape (B, U, D_att).

        Raises:
            AssertionError: If the length of the key tensor exceeds the context size
            or if the product of batch size and dimension is not a multiple of
            the minimum of dimension or 32.

        Examples:
            >>> time_decay = torch.tensor([0.1, 0.2])
            >>> time_first = torch.tensor([0.3, 0.4])
            >>> key = torch.rand(2, 5, 2)  # Example with batch size 2, length 5, D_att 2
            >>> value = torch.rand(2, 5, 2)
            >>> output = WKVLinearAttention.apply(time_decay, time_first, key, value)
            >>> print(output.shape)  # Output shape will be (2, 5, 2)

        Note:
            Ensure that the WKV kernel is loaded before calling this function.
        """
        batch, length, dim = key.size()

        assert length <= wkv_kernel.context_size, (
            f"Cannot process key of length {length} while context_size "
            f"is ({wkv_kernel.context_size}). Limit should be increased."
        )

        assert batch * dim % min(dim, 32) == 0, (
            f"batch size ({batch}) by dimension ({dim}) should be a multiple of "
            f"{min(dim, 32)}"
        )

        ctx.input_dtype = key.dtype

        time_decay = -torch.exp(time_decay.contiguous())
        time_first = time_first.contiguous()

        key = key.contiguous()
        value = value.contiguous()

        out = torch.empty_like(key, memory_format=torch.contiguous_format)

        wkv_kernel.forward(time_decay, time_first, key, value, out)
        ctx.save_for_backward(time_decay, time_first, key, value, out)

        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        WKVLinearAttention function backward pass.

        This method computes the gradients of the inputs with respect to the
        output of the forward pass. It uses the saved tensors from the forward
        context to calculate the gradients for the time decay, time first,
        key, and value tensors.

        Args:
            ctx: Context object containing saved tensors from forward pass.
            grad_output: Output gradient. Shape: (B, U, D_att)

        Returns:
            grad_time_decay: Gradient for channel-wise time decay vector.
                             Shape: (D_att)
            grad_time_first: Gradient for channel-wise time first vector.
                             Shape: (D_att)
            grad_key: Gradient for key tensor. Shape: (B, U, D_att)
            grad_value: Gradient for value tensor. Shape: (B, U, D_att)

        Examples:
            >>> grad_output = torch.randn(2, 3, 4)  # Example gradient output
            >>> grad_time_decay, grad_time_first, grad_key, grad_value = (
            ...     WKVLinearAttention.backward(ctx, grad_output)
            ... )

        Note:
            Ensure that the context contains the necessary tensors saved during
            the forward pass, as they are crucial for computing the gradients.

        Raises:
            RuntimeError: If the context does not contain the expected tensors.
        """
        time_decay, time_first, key, value, output = ctx.saved_tensors
        grad_dtype = ctx.input_dtype  # noqa

        batch, _, dim = key.size()

        grad_time_decay = torch.empty(
            (batch, dim),
            memory_format=torch.contiguous_format,
            dtype=time_decay.dtype,
            device=time_decay.device,
        )

        grad_time_first = torch.empty(
            (batch, dim),
            memory_format=torch.contiguous_format,
            dtype=time_decay.dtype,
            device=time_decay.device,
        )

        grad_key = torch.empty_like(key, memory_format=torch.contiguous_format)
        grad_value = torch.empty_like(value, memory_format=torch.contiguous_format)

        wkv_kernel.backward(
            time_decay,
            time_first,
            key,
            value,
            output,
            grad_output.contiguous(),
            grad_time_decay,
            grad_time_first,
            grad_key,
            grad_value,
        )

        grad_time_decay = torch.sum(grad_time_decay, dim=0)
        grad_time_first = torch.sum(grad_time_first, dim=0)

        return (
            grad_time_decay,
            grad_time_first,
            grad_key,
            grad_value,
        )


def load_wkv_kernel(context_size: int) -> None:
    """
    Load WKV CUDA kernel.

    This function loads the WKV (Weighted Key-Value) CUDA kernel for
    efficient computation in the RWKV model. The kernel is loaded
    using the PyTorch C++ extension loader and requires CUDA
    support.

    Args:
        context_size: The context size to be used by the WKV kernel.
                      It determines the maximum length of the input
                      sequences that can be processed.

    Raises:
        ImportError: If the Ninja package is not installed or if
                     CUDA is not available.

    Note:
        Ensure that the 'ninja' package is installed in your Python
        environment to load the WKV kernel. You can install it via
        pip: `pip install ninja`.

    Examples:
        To load the WKV kernel with a specific context size, you can
        use the following code:

        ```python
        load_wkv_kernel(context_size=128)
        ```

    This will prepare the WKV kernel for usage in further
    computations related to the RWKV model.
    """
    from torch.utils.cpp_extension import load

    global wkv_kernel

    if wkv_kernel is not None and wkv_kernel.context_size == context_size:
        return

    if find_spec("ninja") is None:
        raise ImportError(
            "Ninja package was not found. WKV kernel module can't be loaded "
            "for training. Please, 'pip install ninja' in your environment."
        )

    if not torch.cuda.is_available():
        raise ImportError(
            "CUDA is currently a requirement for WKV kernel loading. "
            "Please set your devices properly and launch again."
        )

    kernel_folder = Path(__file__).resolve().parent / "cuda"
    kernel_files = [kernel_folder / f for f in ["wkv_op.cpp", "wkv_cuda.cu"]]

    kernel_cflags = [
        "-t 4",
        "-std=c++17",
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-DTmax={context_size}",
    ]

    wkv_kernel = load(
        name=f"wkv_{context_size}",
        sources=kernel_files,
        verbose=False,
        extra_cuda_cflags=kernel_cflags,
    )
    wkv_kernel.context_size = context_size


class SelfAttention(torch.nn.Module):
    """
    SelfAttention module definition.

    This module implements the SelfAttention mechanism used in RWKV architectures,
    which allows for effective time mixing in sequence processing tasks. It is based
    on the original implementation found in the RWKV-LM repository and has been
    modified to fit within the espnet2 framework.

    Attributes:
        time_shift: A zero-padding layer for temporal shifting of input sequences.
        time_decay: A learnable parameter representing the channel-wise time decay.
        time_first: A learnable parameter representing the channel-wise time first.
        time_mix_key: A learnable parameter for mixing key inputs.
        time_mix_value: A learnable parameter for mixing value inputs.
        time_mix_receptance: A learnable parameter for mixing receptance inputs.
        proj_key: A linear transformation applied to the key inputs.
        proj_value: A linear transformation applied to the value inputs.
        proj_receptance: A linear transformation applied to the receptance inputs.
        proj_output: A linear transformation applied to the final output.

    Args:
        size: Input/Output size.
        attention_size: Attention hidden size.
        context_size: Context size for WKV kernel.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.

    Examples:
        >>> self_attention = SelfAttention(size=512, attention_size=256,
        ...                                 context_size=128, block_id=0,
        ...                                 num_blocks=4)
        >>> input_tensor = torch.randn(32, 10, 512)  # (B, U, size)
        >>> output, _ = self_attention(input_tensor)
    """

    def __init__(
        self,
        size: int,
        attention_size: int,
        context_size: int,
        block_id: int,
        num_blocks: int,
    ) -> None:
        """Construct a SelfAttention object."""
        super().__init__()

        load_wkv_kernel(context_size)

        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

        self.time_decay = torch.nn.Parameter(torch.empty(attention_size))
        self.time_first = torch.nn.Parameter(torch.empty(attention_size))

        self.time_mix_key = torch.nn.Parameter(torch.empty(1, 1, size))
        self.time_mix_value = torch.nn.Parameter(torch.empty(1, 1, size))
        self.time_mix_receptance = torch.nn.Parameter(torch.empty(1, 1, size))

        self.proj_key = torch.nn.Linear(size, attention_size, bias=True)
        self.proj_value = torch.nn.Linear(size, attention_size, bias=True)
        self.proj_receptance = torch.nn.Linear(size, attention_size, bias=True)

        self.proj_output = torch.nn.Linear(attention_size, size, bias=True)

        self.block_id = block_id

        self.reset_parameters(size, attention_size, block_id, num_blocks)

    def reset_parameters(
        self, size: int, attention_size: int, block_id: int, num_blocks: int
    ) -> None:
        """
        Reset module parameters.

        This method initializes the parameters of the SelfAttention module
        based on the given size and attention configuration. It calculates
        decay speeds and initializes time-related parameters that control
        the attention mechanism within the module.

        Args:
            size: Block size, representing the input/output dimension.
            attention_size: Attention hidden size, determining the number
                            of attention heads.
            block_id: Block index, indicating the position of this block
                       in a larger architecture.
            num_blocks: Total number of blocks in the architecture.

        Examples:
            >>> attention = SelfAttention(size=128, attention_size=64,
            ...                           context_size=512, block_id=0,
            ...                           num_blocks=4)
            >>> attention.reset_parameters(size=128, attention_size=64,
            ...                             block_id=0, num_blocks=4)
        """
        ratio_0_to_1 = block_id / (num_blocks - 1)
        ratio_1_to_almost0 = 1.0 - (block_id / num_blocks)

        time_weight = torch.ones(1, 1, size)

        for i in range(size):
            time_weight[0, 0, i] = i / size

        decay_speed = [
            -5 + 8 * (h / (attention_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            for h in range(attention_size)
        ]
        decay_speed = torch.tensor(
            decay_speed, dtype=self.time_decay.dtype, device=self.time_decay.device
        )

        zigzag = (
            torch.tensor(
                [(i + 1) % 3 - 1 for i in range(attention_size)],
                dtype=self.time_first.dtype,
                device=self.time_first.device,
            )
            * 0.5
        )

        with torch.no_grad():
            self.time_decay.data = decay_speed
            self.time_first.data = torch.ones_like(
                self.time_first * math.log(0.3) + zigzag
            )

            self.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
            self.time_mix_value.data = (
                torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
            )
            self.time_mix_receptance.data = torch.pow(
                time_weight, 0.5 * ratio_1_to_almost0
            )

    @torch.no_grad()
    def wkv_linear_attention(
        self,
        time_decay: torch.Tensor,
        time_first: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Attention (time mixing) modules for RWKV block.

        Based/Modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py.

        Some variables are renamed according to
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py.

        Attributes:
            wkv_kernel: A global variable that holds the WKV CUDA kernel.

        Args:
            time_decay: Channel-wise time decay vector. (D_att)
            time_first: Channel-wise time first vector. (D_att)
            key: Key tensor. (B, U, D_att)
            value: Value tensor. (B, U, D_att)

        Returns:
            out: Weighted Key-Value tensor. (B, U, D_att)

        Raises:
            AssertionError: If the key length exceeds the context size or if the batch
                size multiplied by dimension is not a multiple of the minimum dimension.

        Examples:
            >>> time_decay = torch.tensor([0.1, 0.2, 0.3])
            >>> time_first = torch.tensor([0.5, 0.6, 0.7])
            >>> key = torch.rand(32, 10, 64)  # (B, U, D_att)
            >>> value = torch.rand(32, 10, 64)  # (B, U, D_att)
            >>> output = WKVLinearAttention.apply(time_decay, time_first, key, value)
        """
        num_state, den_state, max_state = state

        time_decay = -torch.exp(time_decay)

        max_for_output = torch.maximum(max_state, (time_first + key))

        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp((time_first + key) - max_for_output)

        numerator = e1 * num_state + e2 * value
        denominator = e1 * den_state + e2

        max_for_state = torch.maximum(key, (max_state + time_decay))

        e1 = torch.exp((max_state + time_decay) - max_for_state)
        e2 = torch.exp(key - max_for_state)

        wkv = numerator / denominator

        state = [e1 * num_state + e2 * value, e1 * den_state + e2, max_for_state]

        return wkv, state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        WKVLinearAttention function forward pass.

        This method computes the forward pass of the WKVLinearAttention function,
        which applies a weighted key-value mechanism. It uses the provided time decay
        and time first vectors along with the key and value tensors to produce an
        output tensor.

        Args:
            ctx: The context object for storing information for backward pass.
            time_decay: Channel-wise time decay vector. Shape: (D_att).
            time_first: Channel-wise time first vector. Shape: (D_att).
            key: Key tensor. Shape: (B, U, D_att), where B is batch size and U is
                sequence length.
            value: Value tensor. Shape: (B, U, D_att).

        Returns:
            out: Weighted Key-Value tensor. Shape: (B, U, D_att).

        Raises:
            AssertionError: If the length of the key exceeds the context size or
                if the product of batch size and dimension is not a multiple of
                the minimum of dimension or 32.

        Examples:
            >>> time_decay = torch.tensor([0.1, 0.2])
            >>> time_first = torch.tensor([0.5, 0.6])
            >>> key = torch.randn(4, 10, 2)  # Example with batch size 4, length 10
            >>> value = torch.randn(4, 10, 2)
            >>> output = WKVLinearAttention.apply(time_decay, time_first, key, value)
            >>> print(output.shape)
            torch.Size([4, 10, 2])
        """
        shifted_x = (
            self.time_shift(x) if state is None else state[1][..., self.block_id]
        )

        key = x * self.time_mix_key + shifted_x * (1 - self.time_mix_key)
        value = x * self.time_mix_value + shifted_x * (1 - self.time_mix_value)
        receptance = x * self.time_mix_receptance + shifted_x * (
            1 - self.time_mix_receptance
        )

        key = self.proj_key(key)
        value = self.proj_value(value)
        receptance = torch.sigmoid(self.proj_receptance(receptance))

        if state is not None:
            state[1][..., self.block_id] = x

            wkv, att_state = self.wkv_linear_attention(
                self.time_decay,
                self.time_first,
                key,
                value,
                tuple(s[..., self.block_id] for s in state[2:]),
            )

            state[2][..., self.block_id] = att_state[0]
            state[3][..., self.block_id] = att_state[1]
            state[4][..., self.block_id] = att_state[2]
        else:
            wkv = WKVLinearAttention.apply(self.time_decay, self.time_first, key, value)

        x = self.proj_output(receptance * wkv)

        return x, state
