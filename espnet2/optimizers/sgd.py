import torch
from typeguard import typechecked


class SGD(torch.optim.SGD):
    """
        Thin inheritance of `torch.optim.SGD` to bind the required arguments, 'lr'.

    This class is a lightweight wrapper around the PyTorch SGD optimizer, providing
    default values for its parameters, except for 'param'. It is intended to be
    used in scenarios where the optimizer is invoked with specific arguments,
    ensuring compatibility with the `AbsTask.main()` method.

    Note:
        The arguments of the optimizer invoked by `AbsTask.main()` must have a
        default value except for 'param'. The reason for `SGD.lr` lacking a default
        value is not clear.

    Args:
        params (iterable): Parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate. Default is 0.1.
        momentum (float, optional): Momentum factor. Default is 0.0.
        dampening (float, optional): Dampening for momentum. Default is 0.0.
        weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        nesterov (bool, optional): Enables Nesterov momentum. Default is False.

    Examples:
        >>> import torch
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> optimizer.step()  # Update parameters based on gradients

    Raises:
        ValueError: If `lr` is not a positive float.
    """

    @typechecked
    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
