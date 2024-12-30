import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class RawNet3Projector(AbsProjector):
    """
        RawNet3Projector is a neural network projector that applies batch normalization
    followed by a linear transformation to the input data. This class is a part of
    the ESPnet2 speaker projection module and inherits from the AbsProjector class.

    Attributes:
        _output_size (int): The size of the output features after projection.
        bn (torch.nn.BatchNorm1d): Batch normalization layer for input features.
        fc (torch.nn.Linear): Linear transformation layer.

    Args:
        input_size (int): The number of input features.
        output_size (int, optional): The number of output features. Defaults to 192.

    Returns:
        torch.Tensor: The projected output features after applying batch
        normalization and linear transformation.

    Examples:
        >>> projector = RawNet3Projector(input_size=256, output_size=128)
        >>> input_tensor = torch.randn(10, 256)  # Batch size of 10
        >>> output_tensor = projector.forward(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([10, 128])

    Note:
        This projector is designed for use in speech processing tasks and is part
        of the ESPnet2 toolkit.

    Todo:
        - Add support for different activation functions in the forward method.
    """

    def __init__(self, input_size, output_size=192):
        super().__init__()
        self._output_size = output_size

        self.bn = torch.nn.BatchNorm1d(input_size)
        self.fc = torch.nn.Linear(input_size, output_size)

    def output_size(self):
        """
                Returns the output size of the projector.

        The output size is defined during the initialization of the
        RawNet3Projector instance and is typically used to determine
        the dimensions of the output tensor produced by the forward
        method.

        Attributes:
            _output_size (int): The size of the output layer, which defaults
            to 192 if not specified during initialization.

        Args:
            None

        Returns:
            int: The output size of the projector.

        Examples:
            projector = RawNet3Projector(input_size=256, output_size=128)
            print(projector.output_size())  # Output: 128

        Note:
            This method is a property that allows access to the output size
            without modifying it.
        """
        return self._output_size

    def forward(self, x):
        """
                Applies a forward pass through the RawNet3Projector model.

        This method takes an input tensor, applies batch normalization, and then
        passes the normalized output through a fully connected linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size) after
            applying batch normalization and linear transformation.

        Examples:
            >>> projector = RawNet3Projector(input_size=256, output_size=192)
            >>> input_tensor = torch.randn(10, 256)  # Batch of 10 samples
            >>> output_tensor = projector.forward(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([10, 192])

        Note:
            The input tensor must have the same size as the specified input_size
            during the initialization of the projector.
        """
        return self.fc(self.bn(x))
