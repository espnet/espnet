import torch


def pytorch_cudnn_version() -> str:
    """
        Returns a string containing the versions of PyTorch and cuDNN, along with
    information about CUDA availability and cuDNN settings.

    The function checks the current version of PyTorch, whether CUDA is available,
    and if cuDNN is enabled. If cuDNN is enabled, it also retrieves the version of
    cuDNN and its configuration settings (benchmarking and determinism).

    Args:
        None

    Returns:
        str: A formatted string that includes the PyTorch version, CUDA availability,
        and cuDNN details if applicable.

    Examples:
        >>> print(pytorch_cudnn_version())
        pytorch.version=1.10.0, cuda.available=True,
        cudnn.version=8005, cudnn.benchmark=True, cudnn.deterministic=False

    Note:
        This function requires the PyTorch library to be installed and may only
        work in an environment where CUDA is supported.
    """
    message = (
        f"pytorch.version={torch.__version__}, "
        f"cuda.available={torch.cuda.is_available()}, "
    )

    if torch.backends.cudnn.enabled:
        message += (
            f"cudnn.version={torch.backends.cudnn.version()}, "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark}, "
            f"cudnn.deterministic={torch.backends.cudnn.deterministic}"
        )
    return message
