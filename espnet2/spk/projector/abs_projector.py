from abc import ABC, abstractmethod

import torch


class AbsProjector(torch.nn.Module, ABC):
    """
        Abstract base class for a projector module in a speaker processing system.

    This class serves as a template for implementing various types of projector
    modules. A projector transforms an input tensor (typically representing speaker
    embeddings) into an output tensor of a specified size. Subclasses must define
    the `output_size` method to specify the size of the output tensor and the
    `forward` method to implement the transformation logic.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If `output_size` or `forward` methods are not
        implemented in a subclass.

    Examples:
        Here is an example of how a subclass might implement this abstract class:

        ```python
        class MyProjector(AbsProjector):
            def output_size(self) -> int:
                return 128

            def forward(self, utt_embd: torch.Tensor) -> torch.Tensor:
                return torch.relu(torch.nn.Linear(utt_embd.size(1), self.output_size())(utt_embd))
        ```

    Note:
        This class inherits from `torch.nn.Module`, so any subclass must also
        comply with the requirements of PyTorch modules.

    Todo:
        Implement specific projector classes that extend this abstract class.
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Abstract property that defines the output size of the projector.

        This property must be implemented by subclasses to specify the expected
        output size when processing input tensors through the projector.

        Returns:
            int: The size of the output tensor after projection.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            class MyProjector(AbsProjector):
                def output_size(self) -> int:
                    return 128

            projector = MyProjector()
            print(projector.output_size())  # Output: 128
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, utt_embd: torch.Tensor) -> torch.Tensor:
        """
                Abstract base class for a projector in a neural network.

        This class defines the interface for projectors that take an input tensor
        (utt_embd) and produce an output tensor. The output size can be defined
        by subclasses implementing the `output_size` method.

        Attributes:
            None

        Args:
            utt_embd (torch.Tensor): The input tensor representing the utterance
            embedding.

        Returns:
            torch.Tensor: The projected output tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            class MyProjector(AbsProjector):
                def output_size(self) -> int:
                    return 128

                def forward(self, utt_embd: torch.Tensor) -> torch.Tensor:
                    # Implementation of forward pass
                    return torch.nn.functional.relu(utt_embd)

            projector = MyProjector()
            output = projector.forward(torch.randn(1, 256))  # Example input tensor

        Note:
            This class is intended to be subclassed and should not be instantiated
            directly.
        """
        raise NotImplementedError
