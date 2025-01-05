# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Pseudo QMF modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.signal import kaiser
except ImportError:
    from scipy.signal.windows import kaiser


def design_prototype_filter(
    taps: int = 62, cutoff_ratio: float = 0.142, beta: float = 9.0
) -> np.ndarray:
    """
    Design prototype filter for PQMF.

    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.

    Args:
        taps (int): The number of filter taps. Must be an even number.
        cutoff_ratio (float): Cut-off frequency ratio. Should be in the range
            (0.0, 1.0).
        beta (float): Beta coefficient for Kaiser window.

    Returns:
        ndarray: Impulse response of the prototype filter of shape (taps + 1,).

    Raises:
        AssertionError: If `taps` is not even or if `cutoff_ratio` is not in
            the range (0.0, 1.0).

    Examples:
        >>> filter_response = design_prototype_filter(taps=64, cutoff_ratio=0.1)
        >>> filter_response.shape
        (65,)

    .. _`A Kaiser window approach for the design of prototype filters of cosine
        modulated filterbanks`: https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """
        Pseudo QMF modules.

    This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

    The PQMF (Pseudo Quadrature Mirror Filter) module is designed for
    near-perfect reconstruction filter banks based on the concept of
    pseudo-QMF banks. It includes methods for both analysis and
    synthesis of signals using these filter banks.

    The implementation uses a Kaiser window approach to design the
    prototype filters that are fundamental to the PQMF operation.

    Attributes:
        analysis_filter (torch.Tensor): Analysis filter coefficients.
        synthesis_filter (torch.Tensor): Synthesis filter coefficients.
        updown_filter (torch.Tensor): Filter for downsampling and upsampling.
        subbands (int): Number of subbands in the PQMF.

    Args:
        subbands (int): The number of subbands.
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for Kaiser window.

    Methods:
        analysis(x: torch.Tensor) -> torch.Tensor:
            Applies the analysis operation on the input tensor using PQMF.

        synthesis(x: torch.Tensor) -> torch.Tensor:
            Applies the synthesis operation on the input tensor using PQMF.

    Examples:
        >>> pqmf = PQMF(subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0)
        >>> input_tensor = torch.randn(1, 1, 1024)  # Example input
        >>> analysis_output = pqmf.analysis(input_tensor)
        >>> synthesis_output = pqmf.synthesis(analysis_output)

    Notes:
        - The cutoff_ratio and beta parameters are optimized for
          `subbands = 4`. See discussion in
          https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.
        - Power will be decreased in the synthesis process, so the output
          is multiplied by the number of subbands. This approach should
          be reviewed for correctness.
        - Further understanding of the reconstruction procedure is
          needed (TODO).
    """

    def __init__(
        self,
        subbands: int = 4,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ):
        """Initilize PQMF module.

        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.

        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.

        """
        super().__init__()

        # build analysis & synthesis filter coefficients
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """
                Pseudo QMF modules.

        This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        """
                Pseudo QMF modules.

        This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.
        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(
            x, self.updown_filter * self.subbands, stride=self.subbands
        )
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
