# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)
from abc import abstractmethod

import torch
import torch.nn as nn


class AbsLoss(nn.Module):
    """
        Abstract base class for loss functions in the ESPnet2 framework.

    This class serves as a template for creating custom loss functions. It inherits
    from PyTorch's `nn.Module` and requires the implementation of the `forward`
    method in derived classes.

    Attributes:
        nout (int): The number of output classes for the loss function.

    Args:
        nout (int): The number of output units for the loss function.
        **kwargs: Additional keyword arguments for further customization.

    Methods:
        forward(x: torch.Tensor, label=None) -> torch.Tensor:
            Computes the loss given the input tensor and the corresponding labels.

    Raises:
        NotImplementedError: If the `forward` method is not implemented in a
        derived class.

    Examples:
        class CustomLoss(AbsLoss):
            def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
                # Custom loss calculation here
                return loss_value

        loss = CustomLoss(nout=10)
        output = loss(torch.randn(5, 10), label=torch.randint(0, 10, (5,)))
    """

    def __init__(self, nout: int, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        """
            Computes the forward pass of the loss function.

        This method is an abstract method that must be implemented by any subclass
        of `AbsLoss`. It takes an input tensor and an optional label tensor to
        compute the loss.

        Args:
            x (torch.Tensor): The input tensor for which the loss is calculated.
            label (optional): The ground truth labels for the input tensor. Default is None.

        Returns:
            torch.Tensor: The computed loss value as a tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            # Example of a subclass implementation
            class MyLoss(AbsLoss):
                def forward(self, x, label=None):
                    return torch.mean(x)  # Replace with actual loss computation

            loss = MyLoss(nout=10)
            input_tensor = torch.randn(5, 10)
            output = loss.forward(input_tensor, label=torch.tensor([1, 0, 1, 0, 1]))
        """
        raise NotImplementedError
