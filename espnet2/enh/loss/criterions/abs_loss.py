from abc import ABC, abstractmethod

import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsEnhLoss(torch.nn.Module, ABC):
    """
    Base class for all Enhancement loss modules.

    This class serves as an abstract base class for defining different types of 
    enhancement loss functions used in audio processing tasks. It provides a 
    structure for loss modules that can be further extended to implement specific 
    loss functions.

    Attributes:
        name (str): The name of the loss module, which will be used as a key 
            in the reporter. Must be implemented in derived classes.
        only_for_test (bool): A boolean flag indicating whether the criterion 
            will only be evaluated during the inference stage. Defaults to 
            False.

    Methods:
        forward(ref, inf) -> torch.Tensor:
            Computes the enhancement loss based on reference and inferred 
            signals. Must be implemented in derived classes.

    Raises:
        NotImplementedError: If the `forward` method is not implemented in a 
            derived class.

    Examples:
        To create a custom enhancement loss, subclass `AbsEnhLoss` and 
        implement the `forward` method:

        ```python
        class CustomLoss(AbsEnhLoss):
            @property
            def name(self) -> str:
                return "custom_loss"

            def forward(self, ref, inf) -> torch.Tensor:
                # Custom loss computation logic
                return torch.mean((ref - inf) ** 2)
        ```

    Note:
        This class is intended to be subclassed. It should not be instantiated 
        directly.
    """

    # the name will be the key that appears in the reporter
    @property
    def name(self) -> str:
        return NotImplementedError

    # This property specifies whether the criterion will only
    # be evaluated during the inference stage
    @property
    def only_for_test(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self,
        ref,
        inf,
    ) -> torch.Tensor:
        # the return tensor should be shape of (batch)
        raise NotImplementedError
