import humanfriendly
import numpy as np
import torch


def get_human_readable_count(number: int) -> str:
    """
        Return a human-readable count of a given integer number.

    This function abbreviates an integer number using the suffixes K, M, B, and T
    for thousands, millions, billions, and trillions, respectively.

    Examples:
        >>> get_human_readable_count(123)
        '123 '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3 B'
        >>> get_human_readable_count(4e12)  # (four trillion)
        '4 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: A positive integer number.

    Returns:
        A string formatted according to the pattern described above.

    Raises:
        AssertionError: If the input number is negative.
    """
    assert number >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    return f"{number:.2f} {labels[index]}"


def to_bytes(dtype) -> int:
    """
        Convert a PyTorch data type to the corresponding byte size.

    This function takes a PyTorch data type as input and returns the size in bytes
    by extracting the number of bits represented by the type and converting it to bytes.

    Args:
        dtype: A PyTorch data type (e.g., torch.float16, torch.float32, etc.).

    Returns:
        An integer representing the size of the data type in bytes.

    Examples:
        >>> to_bytes(torch.float16)
        2
        >>> to_bytes(torch.float32)
        4
        >>> to_bytes(torch.float64)
        8
    """
    # torch.float16 -> 16
    return int(str(dtype)[-2:]) // 8


def model_summary(model: torch.nn.Module) -> str:
    """
        Generate a summary of the given PyTorch model, including the total number of
    parameters, trainable parameters, and model structure.

    This function inspects the provided PyTorch model and returns a string
    summary containing important information about the model's architecture,
    including the class name, total number of parameters, number of trainable
    parameters, the percentage of trainable parameters, the size of the model
    in bytes, and the data type of the model parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.

    Returns:
        str: A formatted string containing the model summary.

    Examples:
        >>> import torch
        >>> class SimpleModel(torch.nn.Module):
        ...     def __init__(self):
        ...         super(SimpleModel, self).__init__()
        ...         self.fc = torch.nn.Linear(10, 5)
        ...
        ...     def forward(self, x):
        ...         return self.fc(x)
        >>> model = SimpleModel()
        >>> print(model_summary(model))
        Model structure:
        SimpleModel(
          (fc): Linear(in_features=10, out_features=5, bias=True)
        )

        Model summary:
            Class Name: SimpleModel
            Total Number of model parameters: 55
            Number of trainable parameters: 55 (100.0%)
            Size: 448 B
            Type: torch.float32
    """
    message = "Model structure:\n"
    message += str(model)
    tot_params = sum(p.numel() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_trainable = "{:.1f}".format(num_params * 100.0 / tot_params)
    tot_params = get_human_readable_count(tot_params)
    num_params = get_human_readable_count(num_params)
    message += "\n\nModel summary:\n"
    message += f"    Class Name: {model.__class__.__name__}\n"
    message += f"    Total Number of model parameters: {tot_params}\n"
    message += (
        f"    Number of trainable parameters: {num_params} ({percent_trainable}%)\n"
    )
    num_bytes = humanfriendly.format_size(
        sum(
            p.numel() * to_bytes(p.dtype) for p in model.parameters() if p.requires_grad
        )
    )
    message += f"    Size: {num_bytes}\n"
    dtype = next(iter(model.parameters())).dtype
    message += f"    Type: {dtype}"
    return message
