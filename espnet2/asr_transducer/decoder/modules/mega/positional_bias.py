"""Positional bias related modules.

Based/modified from https://github.com/facebookresearch/mega/blob/main/fairseq/modules/relative_positional_bias.py
"""  # noqa

import math
from typing import Tuple

import torch


class RelativePositionBias(torch.nn.Module):
    """
    Positional bias related modules.

    Based/modified from https://github.com/facebookresearch/mega/blob/main/fairseq/modules/relative_positional_bias.py

    This module defines classes for implementing relative position bias in neural
    networks. Relative position bias helps models leverage positional information
    when processing sequences, particularly in tasks like natural language
    processing and speech recognition.

    Classes:
        - RelativePositionBias: Computes relative position bias for sequences.
        - RotaryRelativePositionBias: Computes rotary positional embeddings for 
        sequences.

    Attributes:
        max_positions: Maximum number of relative positions.
    """

    def __init__(self, max_positions: int) -> None:
        """Construct a RelativePositionBias object."""
        super().__init__()

        self.max_positions = max_positions

        self.relative_position_bias = torch.nn.Parameter(
            torch.Tensor(2 * self.max_positions - 1)
        )

        self.reset_parameters()

    def reset_parameters(self, val: float = 0.0, std: float = 0.02) -> None:
        """
        Reset module parameters.

        This method initializes the parameters of the RelativePositionBias 
        module using a normal distribution with the specified mean and 
        standard deviation. It is typically called during the initialization 
        of the module to ensure that the parameters are set to a reasonable 
        starting point.

        Args:
            val: Initialization value (mean of the normal distribution).
            std: Standard deviation of the normal distribution.

        Examples:
            >>> rp_bias = RelativePositionBias(max_positions=10)
            >>> rp_bias.reset_parameters(val=0.5, std=0.1)
            >>> rp_bias.relative_position_bias
            tensor([...])  # Normal distribution values around 0.5 with std 0.1

        Note:
            This method can be called multiple times to reinitialize the 
            parameters if needed.
        """
        torch.nn.init.normal_(self.relative_position_bias, mean=val, std=std)

    def forward(self, length: int) -> torch.Tensor:
        """
        Compute rotary relative position bias.

        This method computes the rotary relative position bias based on the input
        sequence length. It uses the pre-computed sine and cosine embeddings to
        create a bias matrix, which can be used in attention mechanisms to
        incorporate relative position information.

        Args:
            length: Sequence length. This should not exceed the maximum number of
                relative positions specified during the initialization of the
                module.

        Returns:
            bias: Rotary relative position bias. The output shape is (L, L), where
                L is the provided sequence length.

        Raises:
            ValueError: If the provided length exceeds the maximum positions
                supported by the module.

        Examples:
            >>> rotary_bias = RotaryRelativePositionBias(size=64, max_positions=2048)
            >>> bias = rotary_bias.forward(length=10)
            >>> print(bias.shape)
            torch.Size([10, 10])

        Note:
            This method utilizes the rotary positional embeddings calculated
            using the `rotary` method, which combines the alpha and beta parameters
            with sine and cosine functions to generate the final bias.
        """
        if length > self.max_positions:
            raise ValueError(
                f"Length {length} is too long for the maximum number of "
                f"allowed positions {self.max_positions}."
            )

        bias = self.relative_position_bias[
            (self.max_positions - length) : (self.max_positions + length - 1)
        ]
        bias = torch.nn.functional.pad(bias, (0, length))

        tile = torch.tile(bias, (length,))[:-length]
        tile = tile.view(length, (3 * length - 2))

        start = (2 * length - 1) // 2
        end = tile.size(1) - start

        tile = tile[:, start:end]

        return tile


class RotaryRelativePositionBias(torch.nn.Module):
    """
    RotaryRelativePositionBias module definition.

    This module computes rotary relative position biases using sinusoidal
    positional embeddings. It is designed to enhance the performance of 
    transformer models by providing a mechanism to capture relative 
    positional information.

    Args:
        size: Module embedding size.
        max_positions: Maximum number of relative positions (default is 2048).

    Attributes:
        sine: Sine components of the sinusoidal embeddings.
        cosine: Cosine components of the sinusoidal embeddings.
        alpha: Learnable parameter representing one set of positional embeddings.
        beta: Learnable parameter representing another set of positional embeddings.
        size: The embedding size for the module.
        max_positions: The maximum number of positions for relative bias.

    Examples:
        >>> rpb = RotaryRelativePositionBias(size=128, max_positions=2048)
        >>> input_tensor = torch.randn(10, 128)  # Sequence length of 10
        >>> output_bias = rpb.forward(length=10)
        >>> print(output_bias.shape)  # Output shape will be (10, 10)

    Note:
        This module is based on research and implementations from 
        Facebook Research's MEGA project.

    Raises:
        ValueError: If the length exceeds the maximum number of positions.
    """

    def __init__(self, size: int, max_positions: int = 2048) -> None:
        """Construct a RotaryRelativePositionBias object."""
        super().__init__()

        self.sine, self.cosine = RotaryRelativePositionBias.get_sinusoid_embeddings(
            max_positions, size
        )

        self.alpha = torch.nn.Parameter(torch.Tensor(1, size))
        self.beta = torch.nn.Parameter(torch.Tensor(1, size))

        self.register_buffer("_pe", torch.FloatTensor(1))

        self.size = size
        self.max_positions = max_positions

        self.reset_parameters()

    def reset_parameters(self, val: float = 0.0, std: float = 0.02) -> None:
        """
        Reset module parameters.

        This method initializes the parameters of the RotaryRelativePositionBias 
        module (alpha and beta) using a normal distribution. The mean and standard 
        deviation of the distribution can be specified via the parameters `val` 
        and `std`.

        Args:
            val: Initialization value (mean of the normal distribution). 
                Defaults to 0.0.
            std: Standard deviation of the normal distribution. 
                Defaults to 0.02.

        Examples:
            >>> rrp_bias = RotaryRelativePositionBias(size=128, max_positions=2048)
            >>> rrp_bias.reset_parameters(val=0.1, std=0.01)
        
        Note:
            This method is typically called during the initialization of the 
            module to ensure that parameters are set to reasonable starting values.
        """
        torch.nn.init.normal_(self.alpha, mean=val, std=std)
        torch.nn.init.normal_(self.beta, mean=val, std=std)

    @staticmethod
    def get_sinusoid_embeddings(
        max_positions: int,
        size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RotaryRelativePositionBias module definition.

        This module implements rotary relative positional bias using sinusoidal
        embeddings. The embeddings are generated based on the specified maximum 
        number of positions and the size of the embeddings. This technique helps 
        to capture the relative positions of tokens in a sequence, enhancing the 
        model's understanding of spatial relationships.

        Attributes:
            size (int): The embedding size for the positional encodings.
            max_positions (int): The maximum number of relative positions allowed.
            sine (torch.Tensor): Sine components of the sinusoidal embeddings.
            cosine (torch.Tensor): Cosine components of the sinusoidal embeddings.
            alpha (torch.nn.Parameter): Learnable parameter for alpha.
            beta (torch.nn.Parameter): Learnable parameter for beta.

        Args:
            size (int): Module embedding size.
            max_positions (int, optional): Maximum number of relative positions. 
                Defaults to 2048.

        Methods:
            reset_parameters(val: float = 0.0, std: float = 0.02) -> None:
                Resets module parameters.

            get_sinusoid_embeddings(max_positions: int, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
                Computes sinusoidal positional embeddings.

            rotary(x: torch.Tensor) -> torch.Tensor:
                Computes rotary positional embeddings.

            forward(length: int) -> torch.Tensor:
                Computes rotary relative position bias.

        Examples:
            >>> rrb = RotaryRelativePositionBias(size=128, max_positions=2048)
            >>> bias = rrb.forward(length=10)
            >>> print(bias.shape)
            torch.Size([10, 10])

        Note:
            The sinusoidal embeddings are calculated using sine and cosine functions,
            which allows the model to effectively represent relative positions in a 
            continuous manner.

        Todo:
            - Implement additional functionality for dynamic position encoding.
        """
        half_size = size // 2

        emb = math.log(10000) / half_size
        emb = torch.exp(torch.arange(half_size, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)

        return torch.sin(emb), torch.cos(emb)

    def rotary(self, x: torch.Tensor) -> torch.Tensor:
        """
        RotaryRelativePositionBias module definition.

        This module computes rotary relative position bias using sinusoidal
        embeddings. It is designed to enhance the performance of attention-based
        models by providing a means to incorporate relative positional information.

        Args:
            size: Module embedding size.
            max_positions: Maximum number of relative positions.

        Attributes:
            sine: Sine components of the sinusoidal embeddings.
            cosine: Cosine components of the sinusoidal embeddings.
            alpha: Learnable parameter for rotary position bias.
            beta: Learnable parameter for rotary position bias.
            _pe: Buffer for positional embeddings.
            size: Size of the embeddings.
            max_positions: Maximum number of positions allowed.

        Methods:
            reset_parameters(val: float = 0.0, std: float = 0.02) -> None:
                Resets the module parameters.
            get_sinusoid_embeddings(max_positions: int, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
                Computes sinusoidal positional embeddings.
            rotary(x: torch.Tensor) -> torch.Tensor:
                Computes rotary positional embeddings.
            forward(length: int) -> torch.Tensor:
                Computes rotary relative position bias.

        Examples:
            >>> rrp_bias = RotaryRelativePositionBias(size=128, max_positions=2048)
            >>> input_tensor = torch.randn(10, 128)  # Example input (L, size)
            >>> bias = rrp_bias.forward(length=10)  # Compute bias for length 10

        Note:
            The rotary positional embeddings are computed based on the input
            sequence length, allowing dynamic adjustments for varying input sizes.

        Raises:
            ValueError: If the input length exceeds the maximum number of positions.
        """
        length, dim = x.size()

        x1, x2 = torch.chunk(x, 2, dim=-1)

        if self.sine is None or length > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionBias.get_sinusoid_embeddings(
                length, dim
            )

            self.max_positions = length

        self.sine = self.sine.to(self._pe)
        self.cosine = self.cosine.to(self._pe)

        sin = self.sine[:length]
        cos = self.cosine[:length]

        x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)

        return x

    def forward(self, length: int) -> torch.Tensor:
        """
        Compute rotary relative position bias.

        This method calculates the rotary relative position bias based on the input
        sequence length. It generates the bias using rotary positional embeddings
        computed from the module's parameters.

        Args:
            length: Sequence length. This should not exceed the maximum number of
                    relative positions defined during the initialization of the
                    module.

        Returns:
            bias: Rotary relative position bias. The shape of the output tensor is
                (L, L), where L is the sequence length.

        Raises:
            ValueError: If the input `length` exceeds the maximum number of allowed
                        positions defined during the initialization of the module.

        Examples:
            >>> rotary_bias = RotaryRelativePositionBias(size=128, max_positions=2048)
            >>> bias_matrix = rotary_bias.forward(length=10)
            >>> print(bias_matrix.shape)
            torch.Size([10, 10])

        Note:
            The method utilizes the `rotary` method to compute the rotary embeddings
            for the parameters `alpha` and `beta`, and then calculates the bias
            using the einsum operation.

        Todo:
            Consider optimizing the memory usage for large sequence lengths.
        """
        alpha = self.rotary(self.alpha.expand(length, self.size))
        beta = self.rotary(self.beta.expand(length, self.size))

        bias = torch.einsum("mk, nk -> mn", alpha, beta)

        return bias
