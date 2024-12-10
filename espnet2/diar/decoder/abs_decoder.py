from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """
    Abstract base class for decoders in the ESPnet2 diarization module.

    This class serves as a blueprint for all decoder implementations, enforcing
    the implementation of the `forward` method and the `num_spk` property in
    derived classes. It inherits from `torch.nn.Module` and implements the
    necessary interfaces for building neural network models.

    Attributes:
        num_spk (int): The number of speakers, must be implemented in derived
            classes.

    Methods:
        forward(input: torch.Tensor, ilens: torch.Tensor) -> Tuple[torch.Tensor,
            torch.Tensor]:
            Defines the computation performed at every call.

    Args:
        input (torch.Tensor): The input tensor representing the data to be
            processed by the decoder.
        ilens (torch.Tensor): A tensor representing the lengths of the input
            sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor
        and any additional information (e.g., lengths or masks).

    Raises:
        NotImplementedError: If the `forward` method or `num_spk` property
            is not implemented in the derived class.

    Examples:
        class MyDecoder(AbsDecoder):
            @property
            def num_spk(self):
                return 2

            def forward(self, input, ilens):
                # Implementation of the forward method
                pass

        decoder = MyDecoder()
        print(decoder.num_spk)  # Output: 2

    Note:
        This class should not be instantiated directly. Instead, create
        subclasses that implement the abstract methods.
    """

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass for the AbsDecoder class, which is an abstract base
        class for decoders in a neural network model. This method must be implemented
        by any subclass of AbsDecoder to define the specific decoding mechanism.

        Args:
            input (torch.Tensor): The input tensor representing the encoded features
                that need to be decoded.
            ilens (torch.Tensor): A tensor representing the lengths of the input
                sequences. This is used to handle variable-length inputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors. The
                first tensor is the decoded output, and the second tensor is a tensor
                representing any additional information required by the decoder (e.g.,
                attention weights or hidden states).

        Raises:
            NotImplementedError: If the method is called directly on the AbsDecoder
                class, as this class is intended to be subclassed.

        Examples:
            >>> decoder = MyDecoder()  # MyDecoder is a subclass of AbsDecoder
            >>> input_tensor = torch.randn(32, 100, 256)  # Example input
            >>> input_lengths = torch.tensor([100] * 32)  # Example lengths
            >>> output, additional_info = decoder.forward(input_tensor, input_lengths)

        Note:
            This method must be overridden in any subclass of AbsDecoder to provide
            the specific decoding logic.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        """
        Abstract property that must be implemented by subclasses to return the number
        of speakers that the decoder can handle.

        This property is essential for the functioning of the decoder, as it informs
        the model how many unique speakers it should expect during the decoding
        process.

        Returns:
            int: The number of speakers that the decoder can process.

        Raises:
            NotImplementedError: If the property is not implemented in the subclass.

        Examples:
            class MyDecoder(AbsDecoder):
                @property
                def num_spk(self):
                    return 2  # This decoder can handle 2 speakers.
        """
        raise NotImplementedError
