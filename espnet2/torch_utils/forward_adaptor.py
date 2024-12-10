import torch
from typeguard import typechecked


class ForwardAdaptor(torch.nn.Module):
    """
        Wrapped module to parallelize a specified method.

    This class allows you to wrap a PyTorch module and specify a method
    that can be parallelized using `torch.nn.DataParallel`. By default,
    `DataParallel` only parallelizes the `forward()` method, so this
    class provides a way to wrap other methods for parallel execution.

    Attributes:
        module (torch.nn.Module): The PyTorch module to be wrapped.
        name (str): The name of the method to be parallelized.

    Args:
        module (torch.nn.Module): The module containing the method to be
            parallelized.
        name (str): The name of the method to parallelize.

    Raises:
        ValueError: If the specified method does not exist in the module.

    Examples:
        >>> class A(torch.nn.Module):
        ...     def foo(self, x):
        ...         return x * 2
        >>> model = A()
        >>> model = ForwardAdaptor(model, "foo")
        >>> model = torch.nn.DataParallel(model, device_ids=[0, 1])
        >>> x = torch.randn(2, 10)
        >>> model(x)  # Calls model.foo(x) in parallel
    """

    @typechecked
    def __init__(self, module: torch.nn.Module, name: str):
        super().__init__()
        self.module = module
        self.name = name
        if not hasattr(module, name):
            raise ValueError(f"{module} doesn't have {name}")

    def forward(self, *args, **kwargs):
        """
                Wrapped module to parallelize specified method.

        This class allows you to wrap a PyTorch module and parallelize a specified
        method using `torch.nn.DataParallel`. The `DataParallel` only parallelizes
        the `forward()` method, so any other method needs to be wrapped in this
        class to allow for parallel execution.

        Attributes:
            module (torch.nn.Module): The original PyTorch module to wrap.
            name (str): The name of the method to parallelize.

        Args:
            module (torch.nn.Module): The PyTorch module that contains the method to
                be parallelized.
            name (str): The name of the method to be invoked in the wrapped module.

        Raises:
            ValueError: If the specified method name does not exist in the module.

        Examples:
            >>> class A(torch.nn.Module):
            ...     def foo(self, x):
            ...         return x * 2
            >>> model = A()
            >>> model = ForwardAdaptor(model, "foo")
            >>> model = torch.nn.DataParallel(model, device_ids=[0, 1])
            >>> x = torch.randn(2, 10)
            >>> model(x)
        """
        func = getattr(self.module, self.name)
        return func(*args, **kwargs)
