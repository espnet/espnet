# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based Neural Codec abstrast class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import torch


class AbsGANCodec(ABC, torch.nn.Module):
    """
    GAN-based Neural Codec model abstract class.

    This abstract class serves as a base for implementing GAN-based neural codecs.
    It defines the core methods that any subclass must implement to function as a
    GAN codec, including methods for encoding and decoding waveforms, as well as
    obtaining meta information about the codec.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If any of the abstract methods are called directly
        on this class.

    Examples:
        To create a custom GAN codec, subclass `AbsGANCodec` and implement
        the required abstract methods as follows:

        ```python
        class MyGANCodec(AbsGANCodec):
            def meta_info(self) -> Dict[str, Any]:
                return {"name": "MyGANCodec", "version": "1.0"}

            def forward(self, forward_generator, *args, **kwargs) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
                # Implementation for forward pass
                pass

            def encode(self, *args, **kwargs) -> torch.Tensor:
                # Implementation for encoding
                pass

            def decode(self, *args, **kwargs) -> torch.Tensor:
                # Implementation for decoding
                pass
        ```

    Note:
        This class is designed to be subclassed, and should not be instantiated
        directly.

    Todo:
        Implement additional utility methods that can assist in training
        and evaluating GAN codecs.
    """

    @abstractmethod
    def meta_info(self) -> Dict[str, Any]:
        """
        Return meta information of the codec.

        This method is expected to return a dictionary containing meta information
        about the GAN-based codec, such as version, authorship, or any other
        relevant details that describe the codec's characteristics.

        Returns:
            Dict[str, Any]: A dictionary containing meta information about the
            codec.

        Examples:
            >>> codec = MyGANCodec()  # MyGANCodec should be a subclass of AbsGANCodec
            >>> info = codec.meta_info()
            >>> print(info)
            {'version': '1.0', 'author': 'Jiatong Shi', 'description': 'A GAN-based codec'}
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """
        Return generator or discriminator loss.

        This method processes the input through the forward generator and
        computes the loss associated with the generator or discriminator.

        Args:
            forward_generator: The generator or discriminator function that will
                process the input.
            *args: Variable length argument list that can include additional
                parameters required by the forward generator.
            **kwargs: Arbitrary keyword arguments that can be passed to the
                forward generator for customization.

        Returns:
            A dictionary containing the loss values. The dictionary can include
            tensors representing generator or discriminator losses, additional
            metrics, or an integer status code.

        Raises:
            NotImplementedError: If this method is called directly from an
                instance of AbsGANCodec without being overridden in a subclass.

        Examples:
            >>> codec = MyGANCodec()
            >>> loss = codec.forward(generator, input_data)
            >>> print(loss)

        Note:
            This method is intended to be overridden by subclasses of AbsGANCodec
            to provide specific loss computations based on their architecture.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
            Encode the input waveform into a latent representation using the codec.

        This method processes the input waveform and generates an encoded
        representation that can be used for further tasks such as decoding
        or other operations within the GAN framework.

        Args:
            *args: Variable length argument list that can include the input
                waveform and any additional parameters required for encoding.
            **kwargs: Arbitrary keyword arguments that may be needed for
                encoding, such as configuration settings.

        Returns:
            torch.Tensor: A tensor representing the encoded latent
            representation of the input waveform.

        Raises:
            NotImplementedError: If the method is not implemented in a
            subclass.

        Examples:
            # Example usage of the encode method
            codec = MyCustomGANCodec()  # Assuming MyCustomGANCodec inherits from AbsGANCodec
            waveform = torch.randn(1, 16000)  # Example waveform tensor
            encoded = codec.encode(waveform)
            print(encoded.shape)  # Output the shape of the encoded representation

        Note:
            The actual implementation of this method should define how the
            encoding is performed based on the specific codec design.
        """
        raise NotImplemented

    @abstractmethod
    def decode(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
            Return decoded waveform from codecs.

        This method is responsible for converting the encoded codecs back into
        a waveform. The exact implementation will depend on the specific
        codec derived from this abstract class. This function is expected to
        handle any necessary transformations and return the resulting
        waveform as a tensor.

        Args:
            *args: Positional arguments that may be required by the specific
                implementation.
            **kwargs: Keyword arguments that may be required by the specific
                implementation.

        Returns:
            torch.Tensor: The decoded waveform represented as a tensor.

        Raises:
            NotImplementedError: If the method is not implemented in the
                subclass.

        Examples:
            # Example of usage in a subclass implementation
            class MyCodec(AbsGANCodec):
                def decode(self, *args, **kwargs) -> torch.Tensor:
                    # Custom decoding logic here
                    return decoded_waveform_tensor

            codec = MyCodec()
            waveform = codec.decode(encoded_codecs)
        """
        raise NotImplemented
