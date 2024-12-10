import torch


def add_gradient_noise(
    model: torch.nn.Module,
    iteration: int,
    duration: float = 100,
    eta: float = 1.0,
    scale_factor: float = 0.55,
):
    """
        Adds noise from a standard normal distribution to the gradients.

    The standard deviation (`sigma`) of the noise is controlled by three hyper-parameters.
    As the number of iterations increases, `sigma` approaches zero, effectively reducing
    the noise added to the gradients.

    Args:
        model (torch.nn.Module): The model whose gradients will be modified.
        iteration (int): The current iteration number.
        duration (float, optional): The interval duration controlling the change of
            `sigma`. Default is 100. Acceptable values are 100 or 1000.
        eta (float, optional): The magnitude of `sigma`. Default is 1.0. Acceptable
            values include 0.01, 0.3, or 1.0.
        scale_factor (float, optional): The scale of `sigma`. Default is 0.55.

    Examples:
        >>> model = SomeModel()
        >>> for i in range(1000):
        >>>     add_gradient_noise(model, i)

    Note:
        The function assumes that the model's parameters have gradients computed
        prior to calling this function.
    """
    interval = (iteration // duration) + 1
    sigma = eta / interval**scale_factor
    for param in model.parameters():
        if param.grad is not None:
            _shape = param.grad.size()
            noise = sigma * torch.randn(_shape).to(param.device)
            param.grad += noise
