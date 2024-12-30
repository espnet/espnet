import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class SkaTdnnProjector(AbsProjector):
    """
        SkaTdnnProjector is a speaker embedding projector that utilizes a Time-Delay
    Neural Network (TDNN) architecture. This projector applies batch normalization
    and a fully connected layer to transform input feature vectors into a specified
    output size, making it suitable for tasks such as speaker recognition or
    verification.

    Attributes:
        bn (torch.nn.BatchNorm1d): Batch normalization layer for input features.
        fc (torch.nn.Linear): Fully connected layer for transforming input to
            output features.
        bn2 (torch.nn.BatchNorm1d): Batch normalization layer for output features.
        _output_size (int): The size of the output feature vector.

    Args:
        input_size (int): The size of the input feature vector.
        output_size (int): The size of the output feature vector.

    Returns:
        torch.Tensor: The transformed output tensor after applying the
            batch normalization and linear transformation.

    Examples:
        >>> projector = SkaTdnnProjector(input_size=128, output_size=64)
        >>> input_tensor = torch.randn(10, 128)  # Batch of 10 input vectors
        >>> output_tensor = projector.forward(input_tensor)
        >>> print(output_tensor.shape)  # Should print: torch.Size([10, 64])

    Note:
        The input tensor should have the shape (batch_size, input_size).
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self._output_size = output_size

        self.bn = torch.nn.BatchNorm1d(input_size)
        self.fc = torch.nn.Linear(input_size, output_size)
        self.bn2 = torch.nn.BatchNorm1d(output_size)

    def output_size(self):
        """
                Returns the output size of the projector.

        This property retrieves the output size that was set during the initialization
        of the SkaTdnnProjector instance. The output size is essential for defining the
        shape of the output tensor after the forward pass through the projector.

        Returns:
            int: The output size of the projector.

        Examples:
            projector = SkaTdnnProjector(input_size=128, output_size=64)
            assert projector.output_size() == 64

        Note:
            The output size is determined by the linear layer defined in the
            constructor of the SkaTdnnProjector class.
        """
        return self._output_size

    def forward(self, x):
        """
                Computes the forward pass of the SkaTdnnProjector.

        This method applies batch normalization to the input tensor, followed by a
        linear transformation and another batch normalization. The transformations
        are defined by the layers initialized in the constructor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size) after
            applying the transformations.

        Examples:
            >>> projector = SkaTdnnProjector(input_size=128, output_size=64)
            >>> input_tensor = torch.randn(32, 128)  # Batch of 32 samples
            >>> output_tensor = projector.forward(input_tensor)
            >>> print(output_tensor.shape)  # Should output: torch.Size([32, 64])
        """
        return self.bn2(self.fc(self.bn(x)))
