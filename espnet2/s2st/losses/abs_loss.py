from abc import ABC, abstractmethod

import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsS2STLoss(torch.nn.Module, ABC):
    """
        Base class for all S2ST (Speech-to-Speech Translation) loss modules.

    This class serves as an abstract base class for implementing various
    loss functions in the context of speech-to-speech translation. It defines
    the structure and contract for all derived loss classes, ensuring they
    implement the required methods.

    Attributes:
        name (str): The name of the loss function, to be implemented by derived
            classes.

    Methods:
        forward: Computes the loss for the given input tensors. This method must
            be implemented by any subclass.

    Raises:
        NotImplementedError: If the derived class does not implement the
            `forward` method or the `name` property.

    Examples:
        To create a custom loss, subclass this class and implement the `name`
        property and `forward` method:

        ```python
        class CustomLoss(AbsS2STLoss):
            @property
            def name(self) -> str:
                return "custom_loss"

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                # Implement the loss calculation
                return torch.mean(inputs)
        ```
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
            Computes the forward pass for the S2ST loss module.

        This method must be implemented by subclasses of AbsS2STLoss. It defines
        the computation that takes place during the forward pass of the loss
        calculation. The output tensor should have the shape of (batch).

        Returns:
            torch.Tensor: A tensor representing the computed loss for the batch.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.

        Examples:
            class CustomLoss(AbsS2STLoss):
                def forward(self) -> torch.Tensor:
                    # Example implementation
                    return torch.tensor([1.0, 2.0, 3.0])

            loss = CustomLoss()
            result = loss.forward()
            print(result)  # Output: tensor([1.0, 2.0, 3.0])
        """
        # the return tensor should be shape of (batch)
        raise NotImplementedError
