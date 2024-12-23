from abc import ABC, abstractmethod

import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsUASRLoss(torch.nn.Module, ABC):
    """
    Base class for all Diarization loss modules.

    This class serves as an abstract base for creating various types of
    loss functions used in Unsupervised Audio Source Separation tasks.
    It defines the essential interface that any derived loss class must
    implement.

    Attributes:
        name (str): The name of the loss function, which will be used as a key
        in the reporter.

    Methods:
        forward: This method must be implemented by derived classes to compute
        the loss based on the model's predictions and the target values.

    Raises:
        NotImplementedError: If the `forward` method is not implemented in a
        derived class.

    Examples:
        To create a custom loss class, inherit from AbsUASRLoss and implement
        the `name` property and `forward` method as follows:

        ```python
        class CustomLoss(AbsUASRLoss):
            @property
            def name(self) -> str:
                return "custom_loss"

            def forward(self, predictions, targets) -> torch.Tensor:
                # Implement the custom loss computation
                loss = torch.mean((predictions - targets) ** 2)
                return loss
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
            Compute the forward pass of the loss function.

        This method must be implemented by any subclass of `AbsUASRLoss`. The
        return tensor should have the shape of (batch), representing the computed
        loss values for each sample in the batch.

        Returns:
            torch.Tensor: A tensor containing the computed loss values for the
            input batch.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> class MyLoss(AbsUASRLoss):
            ...     def forward(self) -> torch.Tensor:
            ...         return torch.tensor([1.0, 2.0, 3.0])
            >>> loss = MyLoss()
            >>> output = loss.forward()
            >>> print(output)
            tensor([1.0, 2.0, 3.0])
        """
        # the return tensor should be shape of (batch)
        raise NotImplementedError
