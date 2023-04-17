# remove np from https://github.com/dhchoi99/NANSY/blob/master/models/yin.py
# adapted from https://github.com/patriceguyot/Yin
# https://github.com/NVIDIA/mellotron/blob/master/yin.py

import numpy as np
import torch
import torch.nn.functional as F


def differenceFunction(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    This solution is implemented directly with torch rfft.


    :param x: audio data (Tensor)
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
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
    Compute difference function of data x. This corresponds to equation (6) in [1]
    This solution is implemented directly with Numpy fft.


    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
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

    This corresponds to equation (8) in [1]

    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
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
    """pytorch backend batch-wise differenceFunction
    has 1e-4 level error with input shape of (32, 22050*1.5)
    Args:
        xs:
        N:
        tau_max:

    Returns:

    """
    xs = xs.double()
    w = xs.shape[-1]
    tau_max = min(tau_max, w)
    zeros = torch.zeros((xs.shape[0], 1))
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
