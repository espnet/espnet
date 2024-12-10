# Copyright 2021 Tomoki Hayashi
# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Singing-voice-synthesis abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsSVS(torch.nn.Module, ABC):
    """
        Singing Voice Synthesis (SVS) abstract class.

    This class serves as an abstract base for singing voice synthesis models.
    It defines the essential methods that any concrete implementation must
    override, ensuring a consistent interface for SVS functionalities.

    Attributes:
        require_raw_singing (bool): Indicates whether raw singing data is
            required for the synthesis process. Defaults to False.
        require_vocoder (bool): Indicates whether a vocoder is required for
            the synthesis process. Defaults to True.

    Methods:
        forward: Calculates outputs and returns the loss tensor.
        inference: Returns predicted output as a dictionary.

    Examples:
        To create a concrete implementation of the AbsSVS class, one must
        subclass it and implement the abstract methods:

        ```python
        class MySVS(AbsSVS):
            def forward(self, text, text_lengths, feats, feats_lengths, **kwargs):
                # Implementation of the forward method
                pass

            def inference(self, text, **kwargs):
                # Implementation of the inference method
                pass
        ```

    Note:
        This class uses PyTorch as the underlying framework and inherits from
        `torch.nn.Module` to ensure compatibility with PyTorch's neural network
        components.
    """

    @abstractmethod
    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
                Calculate outputs and return the loss tensor.

        This method is responsible for processing the input tensors and generating
        the output tensors along with the corresponding loss. The implementation
        of this method must be provided in subclasses of the `AbsSVS` class.

        Args:
            text (torch.Tensor): The input text represented as a tensor.
            text_lengths (torch.Tensor): The lengths of the input text sequences.
            feats (torch.Tensor): The feature representations of the audio.
            feats_lengths (torch.Tensor): The lengths of the feature sequences.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
            containing:
                - The loss tensor.
                - A dictionary of auxiliary outputs.
                - A tensor representing any additional output information.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            # Example of using the forward method in a subclass
            class MySVS(AbsSVS):
                def forward(self, text, text_lengths, feats, feats_lengths, **kwargs):
                    # Implementation goes here
                    pass
        """
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        text: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
                Singing-voice-synthesis abstract class.

        This class serves as a base for singing voice synthesis (SVS) models,
        defining the necessary methods and properties that must be implemented by
        any derived class.

        Attributes:
            require_raw_singing (bool): Indicates whether raw singing data is required.
            require_vocoder (bool): Indicates whether a vocoder is required for synthesis.

        Methods:
            forward(text, text_lengths, feats, feats_lengths, **kwargs):
                Calculate outputs and return the loss tensor.
            inference(text, **kwargs):
                Return predicted output as a dictionary.

        Raises:
            NotImplementedError: If the method is not implemented in the derived class.

        Examples:
            class MySVS(AbsSVS):
                def forward(self, text, text_lengths, feats, feats_lengths, **kwargs):
                    # Implement forward logic here
                    pass

                def inference(self, text, **kwargs):
                    # Implement inference logic here
                    return {"output": torch.tensor([])}

            my_svs = MySVS()
            output = my_svs.inference(torch.tensor([1, 2, 3]))
        """
        raise NotImplementedError

    @property
    def require_raw_singing(self):
        """Return whether or not raw_singing is required."""
        return False

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return True
