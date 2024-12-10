import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class XvectorProjector(AbsProjector):
    """
        XvectorProjector is a neural network projector that transforms input vectors
    into a specified output size using fully connected layers.

    This class inherits from the AbsProjector base class and utilizes two
    fully connected layers with a ReLU activation function in between. The
    projector is primarily designed for use in speaker embedding tasks.

    Attributes:
        _output_size (int): The size of the output vectors after projection.
        fc1 (torch.nn.Linear): The first fully connected layer.
        fc2 (torch.nn.Linear): The second fully connected layer.
        act (torch.nn.ReLU): The activation function used between layers.

    Args:
        input_size (int): The size of the input vectors.
        output_size (int): The desired size of the output vectors.

    Returns:
        torch.Tensor: The transformed output vector after applying the
        fully connected layers and activation function.

    Examples:
        >>> projector = XvectorProjector(input_size=512, output_size=256)
        >>> input_vector = torch.randn(1, 512)
        >>> output_vector = projector.forward(input_vector)
        >>> output_vector.shape
        torch.Size([1, 256])

    Note:
        Ensure that the input size matches the expected input dimensions
        for the first fully connected layer.

    Todo:
        - Implement additional features for more complex projection tasks.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self._output_size = output_size

        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)
        self.act = torch.nn.ReLU()

    def output_size(self):
        """
                Returns the output size of the XvectorProjector.

        This property provides the size of the output tensor after the forward
        pass through the projector. The output size is determined during the
        initialization of the XvectorProjector instance.

        Attributes:
            output_size (int): The size of the output tensor after the forward pass.

        Returns:
            int: The output size specified during initialization.

        Examples:
            projector = XvectorProjector(input_size=128, output_size=64)
            print(projector.output_size())  # Output: 64

        Note:
            This property is read-only and is set during the initialization of
            the projector.
        """
        return self._output_size

    def forward(self, x):
        """
                Forward pass for the XvectorProjector.

        This method takes an input tensor and processes it through two fully
        connected layers with a ReLU activation in between. The first layer maps
        the input to the output size, and the second layer further transforms
        the output of the first layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
            after applying the two linear transformations and the activation.

        Examples:
            >>> projector = XvectorProjector(input_size=128, output_size=64)
            >>> input_tensor = torch.randn(32, 128)  # Batch size of 32
            >>> output_tensor = projector.forward(input_tensor)
            >>> output_tensor.shape
            torch.Size([32, 64])

        Note:
            Ensure that the input tensor has the correct shape matching the
            input_size specified during the initialization of the projector.
        """
        return self.fc2(self.act(self.fc1(x)))
