from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsPostEncoder(torch.nn.Module, ABC):
    """
    Abstract base class for post-encoder modules in ASR (Automatic Speech 
    Recognition) systems.

    This class defines the interface for post-encoder components that process 
    the output from encoders. Any specific implementation must inherit from 
    this class and implement the abstract methods defined herein.

    Attributes:
        None

    Methods:
        output_size() -> int:
            Returns the size of the output tensor from the post-encoder.
        
        forward(input: torch.Tensor, input_lengths: torch.Tensor) -> 
        Tuple[torch.Tensor, torch.Tensor]:
            Processes the input tensor and its corresponding lengths to produce 
            the output tensor and updated lengths.

    Raises:
        NotImplementedError: If the abstract methods are not implemented in 
        a subclass.

    Examples:
        To create a specific post-encoder, subclass this abstract class and 
        implement the required methods:

        ```python
        class MyPostEncoder(AbsPostEncoder):
            def output_size(self) -> int:
                return 256  # Example output size
            
            def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> 
            Tuple[torch.Tensor, torch.Tensor]:
                # Custom processing logic here
                return input, input_lengths  # Example return
        ```

    Note:
        This class should not be instantiated directly. It is meant to be 
        subclassed to provide concrete implementations for various post-encoder 
        functionalities.

    Todo:
        Implement specific post-encoder classes to extend this base class.
    """
    @abstractmethod
    def output_size(self) -> int:
        """
        Returns the size of the output produced by the encoder.

        This method should be implemented by subclasses to provide the 
        specific output size based on the encoding configuration. The 
        output size is typically determined by the architecture of the 
        encoder, which may vary based on the model design.

        Returns:
            int: The size of the output from the encoder.

        Raises:
            NotImplementedError: If the method is not implemented by a 
            subclass.

        Examples:
            >>> class MyEncoder(AbsPostEncoder):
            ...     def output_size(self) -> int:
            ...         return 256
            ...
            >>> encoder = MyEncoder()
            >>> encoder.output_size()
            256
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the size of the output tensor produced by the encoder.

        This method should be implemented by subclasses to provide the
        expected output size based on the specific encoding strategy.

        Returns:
            int: The size of the output tensor.
        """
        raise NotImplementedError
