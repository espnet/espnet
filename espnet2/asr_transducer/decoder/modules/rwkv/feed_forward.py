"""Feed-forward (channel mixing) module for RWKV block.

Based/Modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py

Some variables are renamed according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py.

"""  # noqa

from typing import List, Optional, Tuple

import torch


class FeedForward(torch.nn.Module):
    """
    Feed-forward (channel mixing) module for RWKV block.

    Based/Modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py

    Some variables are renamed according to 
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py.

    Attributes:
        time_shift (torch.nn.ZeroPad2d): Zero padding layer for time shifting.
        time_mix_key (torch.nn.Parameter): Parameter for time mixing key.
        time_mix_receptance (torch.nn.Parameter): Parameter for time mixing receptance.
        proj_key (torch.nn.Linear): Linear transformation for the key projection.
        proj_value (torch.nn.Linear): Linear transformation for the value projection.
        proj_receptance (torch.nn.Linear): Linear transformation for the receptance.

    Args:
        size (int): Input/Output size.
        hidden_size (int): Hidden size.
        block_id (int): Block index.
        num_blocks (int): Number of blocks in the architecture.

    Methods:
        reset_parameters(size: int, block_id: int, num_blocks: int) -> None:
            Reset module parameters based on block size and index.

        forward(x: torch.Tensor, state: Optional[List[torch.Tensor]] = None) 
        -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
            Compute channel mixing for the input sequences.

    Examples:
        >>> ff = FeedForward(size=256, hidden_size=128, block_id=0, num_blocks=4)
        >>> input_tensor = torch.randn(32, 10, 256)  # (Batch, Sequence, Size)
        >>> output, state = ff(input_tensor)

    Raises:
        ValueError: If the input tensor shape is incorrect.

    Note:
        This module is part of the RWKV architecture, which is designed for 
        efficient sequence modeling tasks.
    """

    def __init__(
        self, size: int, hidden_size: int, block_id: int, num_blocks: int
    ) -> None:
        """Construct a FeedForward object."""
        super().__init__()

        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

        self.time_mix_key = torch.nn.Parameter(torch.empty(1, 1, size))
        self.time_mix_receptance = torch.nn.Parameter(torch.empty(1, 1, size))

        self.proj_key = torch.nn.Linear(size, hidden_size, bias=True)
        self.proj_value = torch.nn.Linear(hidden_size, size, bias=True)
        self.proj_receptance = torch.nn.Linear(size, size, bias=True)

        self.block_id = block_id

        self.reset_parameters(size, block_id, num_blocks)

    def reset_parameters(self, size: int, block_id: int, num_blocks: int) -> None:
        """
        Reset module parameters.

        This method initializes the parameters of the FeedForward module based on the 
        provided size, block_id, and the total number of blocks in the architecture. 
        The parameters are set using a power function of a time weight tensor, which 
        helps in controlling the influence of different time steps during training.

        Args:
            size: Block size, which determines the dimensions of the input and output.
            block_id: The index of the current block in the architecture.
            num_blocks: The total number of blocks in the architecture.

        Note:
            The time mixing parameters are initialized using a ratio that scales 
            according to the block index, which helps in managing the temporal 
            dynamics of the model.

        Examples:
            >>> ff = FeedForward(size=128, hidden_size=256, block_id=0, num_blocks=4)
            >>> ff.reset_parameters(size=128, block_id=0, num_blocks=4)
        """
        ratio_1_to_almost0 = 1.0 - (block_id / num_blocks)

        time_weight = torch.ones(1, 1, size)

        for i in range(size):
            time_weight[0, 0, i] = i / size

        with torch.no_grad():
            self.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
            self.time_mix_receptance.data = torch.pow(time_weight, ratio_1_to_almost0)

    def forward(
        self, x: torch.Tensor, state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Feed-forward (channel mixing) module for RWKV block.

        This module is based on and modified from the implementation found at
        https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py. Some variables 
        have been renamed according to the Hugging Face Transformers implementation at 
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py.

        Attributes:
            time_shift (torch.nn.ZeroPad2d): A zero padding layer for time shifting.
            time_mix_key (torch.nn.Parameter): Parameter for mixing keys over time.
            time_mix_receptance (torch.nn.Parameter): Parameter for mixing receptance 
                over time.
            proj_key (torch.nn.Linear): Linear transformation for keys.
            proj_value (torch.nn.Linear): Linear transformation for values.
            proj_receptance (torch.nn.Linear): Linear transformation for receptance.
            block_id (int): The index of the current block.

        Args:
            size (int): Input/Output size.
            hidden_size (int): Hidden size.
            block_id (int): Block index.
            num_blocks (int): Total number of blocks in the architecture.

        Methods:
            reset_parameters(size: int, block_id: int, num_blocks: int) -> None:
                Resets the parameters of the FeedForward module.

            forward(x: torch.Tensor, state: Optional[List[torch.Tensor]] = None) -> 
                Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
                Computes the channel mixing operation.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]]]: 
                - x: FeedForward output sequences with shape (B, U, size).
                - state: Updated decoder hidden state, shape [5 x (B, 1, size, N)].

        Examples:
            >>> ff = FeedForward(size=256, hidden_size=512, block_id=0, num_blocks=2)
            >>> input_tensor = torch.randn(32, 10, 256)  # (B, U, size)
            >>> output, state = ff.forward(input_tensor)

        Note:
            The `state` parameter is optional. If provided, it should contain the 
            decoder hidden state for the current block.

        Raises:
            ValueError: If the input tensor dimensions do not match the expected 
            size.
        """
        shifted_x = (
            self.time_shift(x) if state is None else state[0][..., self.block_id]
        )

        key = x * self.time_mix_key + shifted_x * (1 - self.time_mix_key)
        receptance = x * self.time_mix_receptance + shifted_x * (
            1 - self.time_mix_receptance
        )

        key = torch.square(torch.relu(self.proj_key(key)))
        value = self.proj_value(key)
        receptance = torch.sigmoid(self.proj_receptance(receptance))

        if state is not None:
            state[0][..., self.block_id] = x

        x = receptance * value

        return x, state
