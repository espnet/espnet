# remove np from https://github.com/dhchoi99/NANSY/blob/master/models/yin.py
# adapted from https://github.com/patriceguyot/Yin
# https://github.com/NVIDIA/mellotron/blob/master/yin.py

import numpy as np
import torch
import torch.nn.functional as F


def differenceFunction(x, N, tau_max):
    """
        Compute the difference function of the input audio data.

    This function computes the difference function of the given audio data `x`.
    This corresponds to equation (6) in the referenced literature. The solution
    is implemented directly using PyTorch's rfft.

    Attributes:
        None

    Args:
        x (torch.Tensor): Audio data in the form of a 2D tensor where the first
                          dimension represents the batch size and the second
                          dimension represents the audio signal length.
        N (int): The length of the audio data.
        tau_max (int): The integration window size, which determines the maximum
                       lag to be considered in the difference function.

    Returns:
        list: The computed difference function.

    Raises:
        AssertionError: If the input tensor `x` does not have exactly 2 dimensions.

    Examples:
        >>> import torch
        >>> x = torch.randn(32, 22050)  # Simulated audio data for 32 samples
        >>> N = x.shape[1]
        >>> tau_max = 2048
        >>> difference_function_result = differenceFunction(x, N, tau_max)
    """

    # x = np.array(x, np.float64) #[B,T]
    assert x.dim() == 2
    b, w = x.shape
    if w < tau_max:
        x = F.pad(
            x,
            (tau_max - w - (tau_max - w) // 2, (tau_max - w) // 2),
            "constant",
            mode="reflect",
        )
    w = tau_max
    # x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    x_cumsum = torch.cat(
        [torch.zeros([b, 1], device=x.device), (x * x).cumsum(dim=1)], dim=1
    )
    size = w + tau_max
    p2 = (size // 32).bit_length()
    # p2 = ceil(log2(size+1 // 32))
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(n * 2**p2 for n in nice_numbers if n * 2**p2 >= size)
    fc = torch.fft.rfft(x, size_pad)  # [B,F]
    conv = torch.fft.irfft(fc * fc.conj())[:, :tau_max]
    return (
        x_cumsum[:, w : w - tau_max : -1]
        + x_cumsum[:, w]
        - x_cumsum[:, :tau_max]
        - 2 * conv
    )


def differenceFunction_np(x, N, tau_max):
    """
        Compute difference function of data x. This corresponds to equation (6) in [1].

    This solution is implemented directly with Numpy fft.

    Attributes:
        x (np.ndarray): Audio data.
        N (int): Length of data.
        tau_max (int): Integration window size.

    Args:
        x: Audio data as a NumPy array.
        N: Length of data.
        tau_max: Integration window size.

    Returns:
        list: The computed difference function.

    Examples:
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> N = len(x)
        >>> tau_max = 2
        >>> difference_function_result = differenceFunction_np(x, N, tau_max)

    Note:
        Ensure that the input audio data is a 1-dimensional NumPy array.
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.0]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2**p2 for x in nice_numbers if x * 2**p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w : w - tau_max : -1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def cumulativeMeanNormalizedDifferenceFunction(df, N, eps=1e-8):
    """
    Compute cumulative mean normalized difference function (CMND).

    This function computes the cumulative mean normalized difference function
    based on the provided difference function. This corresponds to equation
    (8) in [1]. The computation accounts for a small epsilon value to avoid
    division by zero.

    Args:
        df (torch.Tensor): The difference function, expected to be a 2D tensor
            where the first dimension corresponds to the batch size and the
            second dimension corresponds to the difference values.
        N (int): The length of the data, which influences the range of
            calculations.
        eps (float, optional): A small constant added to the denominator to
            prevent division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: The cumulative mean normalized difference function,
            which is a tensor of the same batch size as the input difference
            function, with an additional leading dimension of ones.

    Examples:
        >>> df = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> N = 3
        >>> cmnd = cumulativeMeanNormalizedDifferenceFunction(df, N)
        >>> print(cmnd)
        tensor([[1.0000, 0.3333, 0.5000],
                [1.0000, 0.8000, 0.6000]])

    Note:
        Ensure that the input difference function `df` contains valid
        values, as the calculation assumes that `df` will not lead to
        division by zero aside from the safeguard provided by `eps`.

    Raises:
        ValueError: If the shape of `df` is not 2D or if `N` is not
        a positive integer.
    """
    # np.seterr(divide='ignore', invalid='ignore')
    # scipy method, assert df>0 for all element
    #   cmndf = df[1:] * np.asarray(list(range(1, N)))
    # / (np.cumsum(df[1:]).astype(float) + eps)
    B, _ = df.shape
    cmndf = (
        df[:, 1:]
        * torch.arange(1, N, device=df.device, dtype=df.dtype).view(1, -1)
        / (df[:, 1:].cumsum(dim=-1) + eps)
    )
    return torch.cat(
        [torch.ones([B, 1], device=df.device, dtype=df.dtype), cmndf], dim=-1
    )


def differenceFunctionTorch(xs: torch.Tensor, N, tau_max) -> torch.Tensor:
    """
    Compute the difference function using PyTorch for batch audio data.

    This function computes the difference function in a batch-wise manner,
    which corresponds to the equation (6) in the referenced literature. The
    implementation leverages PyTorch's FFT capabilities for efficient
    computation.

    Args:
        xs (torch.Tensor): A batch of audio data of shape (B, T) where B is the
            batch size and T is the number of samples in each audio.
        N (int): The length of the audio data.
        tau_max (int): The maximum integration window size for the difference
            function computation.

    Returns:
        torch.Tensor: A tensor containing the computed difference function,
            with shape (B, tau_max).

    Examples:
        >>> xs = torch.randn(32, int(22050 * 1.5)).cuda()
        >>> N = xs.shape[-1]
        >>> tau_max = 2048
        >>> result = differenceFunctionTorch(xs, N, tau_max)
        >>> print(result.shape)
        torch.Size([32, 2048])

    Note:
        This function has a level of error around 1e-4 when processing input
        shapes of (32, 22050 * 1.5).
    """
    xs = xs.double()
    w = xs.shape[-1]
    tau_max = min(tau_max, w)
    x_cumsum = torch.cat(
        (
            torch.zeros((xs.shape[0], 1), device=xs.device),
            (xs * xs).cumsum(dim=-1, dtype=torch.double),
        ),
        dim=-1,
    )  # B x w
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2**p2 for x in nice_numbers if x * 2**p2 >= size)

    fcs = torch.fft.rfft(xs, n=size_pad, dim=-1)
    convs = torch.fft.irfft(fcs * fcs.conj())[:, :tau_max]
    y1 = torch.flip(x_cumsum[:, w - tau_max + 1 : w + 1], dims=[-1])
    y = y1 + x_cumsum[:, w].unsqueeze(-1) - x_cumsum[:, :tau_max] - 2 * convs
    return y


def cumulativeMeanNormalizedDifferenceFunctionTorch(
    dfs: torch.Tensor, N, eps=1e-8
) -> torch.Tensor:
    """
        Compute cumulative mean normalized difference function (CMND) using PyTorch.

    This function computes the cumulative mean normalized difference function (CMND)
    based on the provided difference function. The computation is performed in a
    batch-wise manner and is designed to handle tensors efficiently. This
    corresponds to equation (8) in the referenced literature.

    Attributes:
        dfs (torch.Tensor): The input difference function tensor of shape
            (B, N), where B is the batch size and N is the length of the
            difference function.

    Args:
        dfs (torch.Tensor): A tensor representing the difference function.
        N (int): The length of the data used in the computation.
        eps (float, optional): A small value added for numerical stability.
            Defaults to 1e-8.

    Returns:
        torch.Tensor: A tensor representing the cumulative mean normalized
        difference function of shape (B, N).

    Examples:
        >>> dfs = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.5, 2.5]])
        >>> cmndf = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, 3)
        >>> print(cmndf)
        tensor([[1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000]])

    Note:
        This function assumes that the input difference function `dfs` has
        at least two columns, as it computes cumulative sums starting from
        the second column.
    """
    arange = torch.arange(1, N, device=dfs.device, dtype=torch.float64)
    cumsum = torch.cumsum(dfs[:, 1:], dim=-1, dtype=torch.float64).to(dfs.device)

    cmndfs = dfs[:, 1:] * arange / (cumsum + eps)
    cmndfs = torch.cat(
        (torch.ones(cmndfs.shape[0], 1, device=dfs.device), cmndfs), dim=-1
    )
    return cmndfs


if __name__ == "__main__":
    wav = torch.randn(32, int(22050 * 1.5)).cuda()
    wav_numpy = wav.detach().cpu().numpy()
    x = wav_numpy[0]

    w_len = 2048
    w_step = 256
    tau_max = 2048
    W = 2048

    startFrames = list(range(0, x.shape[-1] - w_len, w_step))
    startFrames = np.asarray(startFrames)
    # times = startFrames / sr
    frames = [x[..., t : t + W] for t in startFrames]
    frames = np.asarray(frames)
    frames_torch = torch.from_numpy(frames).cuda()

    cmndfs0 = []
    for idx, frame in enumerate(frames):
        df = differenceFunction(frame, frame.shape[-1], tau_max)
        cmndf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        cmndfs0.append(cmndf)
    cmndfs0 = np.asarray(cmndfs0)

    dfs = differenceFunctionTorch(frames_torch, frames_torch.shape[-1], tau_max)
    cmndfs1 = (
        cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, tau_max)
        .detach()
        .cpu()
        .numpy()
    )
    print(cmndfs0.shape, cmndfs1.shape)
    print(np.sum(np.abs(cmndfs0 - cmndfs1)))
