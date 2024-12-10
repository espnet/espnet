import torch


def is_flash_attn_supported() -> bool:
    """
        Determines if Flash Attention is supported on the current GPU.

    This function checks whether the current GPU is compatible with Flash Attention.
    It does so by verifying if a CUDA-capable GPU is available and if its name
    matches any of the supported GPU models.

    Attributes:
        None

    Args:
        None

    Returns:
        bool: True if the current GPU supports Flash Attention, False otherwise.

    Raises:
        None

    Examples:
        >>> is_flash_attn_supported()
        True  # if the current GPU is supported
        False  # if the current GPU is not supported

    Note:
        This function currently supports GPUs from the Ampere, Ada, and Hopper
        architectures. The list of supported GPUs may not be exhaustive.

    Todo:
        - Update the list of supported GPUs as new models are released.
    """

    if not torch.cuda.is_available():
        return False

    # Approximate list of supported GPUs, might be missing some
    AMPERE_GPUs = ["RTX 20", "RTX 30", "A10", "A20", "A30", "A40", "A50", "A60"]
    ADA_GPUs = ["RTX 40", "RTX 45", "RTX 50", "RTX 60", "L4"]
    HOPPER_GPUs = ["H100", "H200"]

    SUPPORTED_GPUs = AMPERE_GPUs + ADA_GPUs + HOPPER_GPUs

    gpu_name = torch.cuda.get_device_name()
    for supported in SUPPORTED_GPUs:
        if supported in gpu_name:
            return True


if __name__ == "__main__":
    print(is_flash_attn_supported())
