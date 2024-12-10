from abc import ABC, abstractmethod

import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsS2STAuxAttention(torch.nn.Module, ABC):
    """
    Base class for all S2ST auxiliary attention modules.

    This class serves as an abstract base class for implementing different
    types of auxiliary attention mechanisms used in Speech-to-Speech Translation
    (S2ST) models. Subclasses should implement the `forward` method to define
    their specific attention computation.

    For more details on the underlying principles and implementation, refer to
    the paper: https://arxiv.org/abs/2107.08661

    Attributes:
        name (str): The name of the attention module, which will be used as
        the key in the reporter. This is to be implemented in the subclasses.

    Methods:
        forward():
            Computes the attention weights and returns a tensor.

    Raises:
        NotImplementedError: If the `forward` method is not implemented in
        a subclass.

    Examples:
        class MyAuxAttention(AbsS2STAuxAttention):
            def forward(self):
                # Implement specific attention mechanism
                return torch.tensor([1.0, 2.0, 3.0])

        attention = MyAuxAttention()
        output = attention.forward()
    """

    # the name will be the key that appears in the reporter
    @property
    def name(self) -> str:
        return NotImplementedError

    @abstractmethod
    def forward(
        self,
    ) -> torch.Tensor:
        """
            Executes the forward pass of the auxiliary attention module.

        This method is intended to be implemented by subclasses of the
        AbsS2STAuxAttention class. It should define how the forward
        computation is carried out and return the resulting tensor.

        Returns:
            torch.Tensor: The output tensor from the forward computation,
            which should have the shape of (batch).

        Raises:
            NotImplementedError: If the method is not implemented in a
            subclass.

        Examples:
            # Example usage in a subclass:
            class MyAttention(AbsS2STAuxAttention):
                def forward(self) -> torch.Tensor:
                    # Implement the forward logic here
                    return torch.tensor([1.0, 2.0, 3.0])  # Example output

            attention = MyAttention()
            output = attention.forward()
            print(output)  # Output: tensor([1., 2., 3.])
        """
        # the return tensor should be shape of (batch)
        raise NotImplementedError
