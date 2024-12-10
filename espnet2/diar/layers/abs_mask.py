from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsMask(torch.nn.Module, ABC):
    """
    Abstract base class for defining a masking mechanism in speaker diarization.

    This class serves as a blueprint for creating different types of masks that can
    be applied to the input features in speaker diarization tasks. It inherits from
    PyTorch's `torch.nn.Module` and requires subclasses to implement the necessary
    methods and properties to specify the masking behavior.

    Attributes:
        max_num_spk (int): The maximum number of speakers that the mask can handle.
            This must be defined in subclasses.

    Methods:
        forward(input, ilens, bottleneck_feat, num_spk) -> Tuple[Tuple[torch.Tensor],
            torch.Tensor, OrderedDict]:
            Abstract method that must be implemented by subclasses to define the
            forward pass of the mask.

    Args:
        input (torch.Tensor): The input tensor representing the audio features.
        ilens (torch.Tensor): A tensor containing the lengths of the input sequences.
        bottleneck_feat (torch.Tensor): Features extracted from the bottleneck layer.
        num_spk (int): The number of speakers to consider in the masking process.

    Returns:
        Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]: A tuple containing:
            - A tuple of tensors representing the masks for each speaker.
            - A tensor representing the combined output.
            - An ordered dictionary containing any additional information.

    Raises:
        NotImplementedError: If the subclass does not implement the `max_num_spk`
            property or the `forward` method.

    Examples:
        class MyMask(AbsMask):
            @property
            def max_num_spk(self) -> int:
                return 5

            def forward(self, input, ilens, bottleneck_feat, num_spk):
                # Implement the masking logic here
                pass

        my_mask = MyMask()
        output = my_mask(input_tensor, input_lengths, bottleneck_features, num_speakers)

    Note:
        This class should not be instantiated directly. Subclasses must provide
        concrete implementations of the abstract methods.
    """
    @property
    @abstractmethod
    def max_num_spk(self) -> int:
        """
        Abstract property that defines the maximum number of speakers supported by the
        masking model. This property should be implemented in subclasses of the
        `AbsMask` class to specify the maximum number of speakers that can be handled.

        Returns:
            int: The maximum number of speakers supported by the model.

        Raises:
            NotImplementedError: If the property is accessed without being overridden
            in a subclass.

        Examples:
            class MyMask(AbsMask):
                @property
                def max_num_spk(self) -> int:
                    return 5

            my_mask = MyMask()
            print(my_mask.max_num_spk)  # Output: 5
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input,
        ilens,
        bottleneck_feat,
        num_spk,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        """
        Gets the maximum number of speakers.

        Returns:
            int: The maximum number of speakers that can be processed by
            this mask implementation.
        """
        raise NotImplementedError
