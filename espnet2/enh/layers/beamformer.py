"""Beamformer module."""

from typing import List, Union

import torch
from packaging.version import parse as V
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import (
    cat,
    complex_norm,
    einsum,
    inverse,
    is_complex,
    is_torch_complex_tensor,
    matmul,
    reverse,
    solve,
    to_double,
)

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
EPS = torch.finfo(torch.double).eps


def prepare_beamformer_stats(
    signal,
    masks_speech,
    mask_noise,
    powers=None,
    beamformer_type="mvdr",
    bdelay=3,
    btaps=5,
    eps=1e-6,
):
    """
    Prepare necessary statistics for constructing the specified beamformer.

    This function computes the required statistics for different types of 
    beamformers based on the provided signal, masks, and optional powers. 
    The output statistics can include power spectral densities (PSD) of 
    noise and speech, which are essential for beamforming algorithms.

    Args:
        signal (torch.complex64/ComplexTensor): 
            Input signal tensor of shape (..., F, C, T), where F is the 
            number of frequency bins, C is the number of channels, and T 
            is the number of time frames.
        masks_speech (List[torch.Tensor]): 
            A list of masks for all speech sources, each of shape (..., F, C, T).
        mask_noise (torch.Tensor): 
            Noise mask tensor of shape (..., F, C, T).
        powers (List[torch.Tensor], optional): 
            List of power tensors for all speech sources, each of shape (..., F, T).
            Used for wMPDR or WPD beamformers. Defaults to None.
        beamformer_type (str, optional): 
            Specifies the type of beamformer to use. Options include "mvdr", 
            "wmpdr", "wpd", etc. Defaults to "mvdr".
        bdelay (int, optional): 
            Delay factor used for WPD beamformers. Defaults to 3.
        btaps (int, optional): 
            Number of filter taps used for WPD beamformers. Defaults to 5.
        eps (torch.Tensor, optional): 
            A small constant to prevent division by zero. Defaults to 1e-6.

    Returns:
        beamformer_stats (dict): 
            A dictionary containing necessary statistics, including:
            - "psd_n": Power spectral density of noise.
            - "psd_speech": Power spectral density of speech.
            - "psd_distortion": Power spectral density of distortion.
        
            Note:
            * When `masks_speech` is a tensor or a single-element list, all 
              returned statistics are tensors.
            * When `masks_speech` is a multi-element list, some returned 
              statistics can be a list, e.g., "psd_n" for MVDR, 
              "psd_speech" and "psd_distortion".

    Examples:
        >>> signal = torch.randn(1, 64, 2, 128, dtype=torch.complex64)
        >>> masks_speech = [torch.rand(1, 64, 2, 128) for _ in range(2)]
        >>> mask_noise = torch.rand(1, 64, 2, 128)
        >>> stats = prepare_beamformer_stats(signal, masks_speech, mask_noise)

    Raises:
        AssertionError: If the specified beamformer type is not supported.
    """
    from espnet2.enh.layers.dnn_beamformer import BEAMFORMER_TYPES

    assert beamformer_type in BEAMFORMER_TYPES, "%s is not supported yet"

    if isinstance(masks_speech, (list, tuple)):
        masks_speech = [to_double(m) for m in masks_speech]
    else:
        masks_speech = [to_double(masks_speech)]
    num_spk = len(masks_speech)

    if (
        beamformer_type.startswith("wmpdr")
        or beamformer_type.startswith("wpd")
        or beamformer_type == "wlcmp"
        or beamformer_type == "wmwf"
    ):
        if powers is None:
            power_input = signal.real**2 + signal.imag**2
            # Averaging along the channel axis: (..., C, T) -> (..., T)
            powers = [(power_input * m).mean(dim=-2) for m in masks_speech]
        else:
            assert len(powers) == num_spk, (len(powers), num_spk)
        inverse_powers = [1 / torch.clamp(p, min=eps) for p in powers]

    psd_speeches = [get_power_spectral_density_matrix(signal, m) for m in masks_speech]
    if (
        beamformer_type == "mvdr_souden"
        or beamformer_type == "sdw_mwf"
        or beamformer_type == "r1mwf"
        or beamformer_type.startswith("mvdr_tfs")
        or not beamformer_type.endswith("_souden")
    ):
        # MVDR or other RTF-based formulas
        if mask_noise is not None:
            psd_bg = get_power_spectral_density_matrix(signal, to_double(mask_noise))
        if num_spk == 1:
            assert mask_noise is not None
            psd_noise = psd_bg
        else:
            psd_noise = []
            for i in range(num_spk):
                if beamformer_type.startswith("mvdr_tfs"):
                    # NOTE: psd_noise is a list only for this beamformer
                    psd_noise_i = [psd for j, psd in enumerate(psd_speeches) if j != i]
                else:
                    psd_sum = sum(psd for j, psd in enumerate(psd_speeches) if j != i)
                    psd_noise_i = (
                        psd_bg + psd_sum if mask_noise is not None else psd_sum
                    )
                psd_noise.append(psd_noise_i)

    if beamformer_type in (
        "mvdr",
        "mvdr_souden",
        "mvdr_tfs_souden",
        "sdw_mwf",
        "r1mwf",
        "lcmv",
        "gev",
        "gev_ban",
    ):
        psd_n = psd_noise
    elif beamformer_type == "mvdr_tfs":
        psd_n = psd_noise
        psd_noise = [sum(psd_noise_i) for psd_noise_i in psd_noise]
    elif beamformer_type in ("mpdr", "mpdr_souden", "lcmp", "mwf"):
        psd_n = einsum("...ct,...et->...ce", signal, signal.conj())
    elif beamformer_type in ("wmpdr", "wmpdr_souden", "wlcmp", "wmwf"):
        psd_n = [
            einsum(
                "...ct,...et->...ce",
                signal * inv_p[..., None, :],
                signal.conj(),
            )
            for inv_p in inverse_powers
        ]
    elif beamformer_type in ("wpd", "wpd_souden"):
        psd_n = [
            get_covariances(signal, inv_p, bdelay, btaps, get_vector=False)
            for inv_p in inverse_powers
        ]

    if num_spk == 1:
        psd_speeches = psd_speeches[0]
        if isinstance(psd_n, (list, tuple)):
            psd_n = psd_n[0]

    if beamformer_type in (
        "mvdr",
        "mpdr",
        "wmpdr",
        "wpd",
        "lcmp",
        "wlcmp",
        "lcmv",
        "mvdr_tfs",
    ):
        return {"psd_n": psd_n, "psd_speech": psd_speeches, "psd_distortion": psd_noise}
    elif (
        beamformer_type.endswith("_souden")
        or beamformer_type.startswith("gev")
        or beamformer_type == "mwf"
        or beamformer_type == "wmwf"
        or beamformer_type == "sdw_mwf"
        or beamformer_type == "r1mwf"
    ):
        return {"psd_n": psd_n, "psd_speech": psd_speeches}


def get_power_spectral_density_matrix(
    xs, mask, normalization=True, reduction="mean", eps: float = 1e-15
):
    """
    Return cross-channel power spectral density (PSD) matrix.

    This function computes the cross-channel power spectral density (PSD) matrix 
    by applying the provided mask to the input signal. The PSD matrix is crucial for 
    various beamforming techniques in audio signal processing, especially in scenarios 
    involving multi-channel audio.

    Args:
        xs (torch.complex64/ComplexTensor): 
            The input signal tensor of shape (..., F, C, T), where F is the number 
            of frequency bins, C is the number of channels, and T is the number of 
            time frames.
        mask (torch.Tensor): 
            The mask tensor of shape (..., F, C, T) that is applied to the input 
            signal to compute the PSD.
        normalization (bool): 
            Whether to normalize the mask along the time axis before computing the PSD. 
            Default is True.
        reduction (str): 
            Specifies the reduction method to apply. Can be "mean" or "median". 
            Default is "mean".
        eps (float): 
            A small constant added for numerical stability to avoid division by zero. 
            Default is 1e-15.

    Returns:
        psd (torch.complex64/ComplexTensor): 
            The computed power spectral density matrix of shape (..., F, C, C), where 
            each element represents the PSD between channels.

    Raises:
        ValueError: If an unknown reduction mode is specified.

    Examples:
        >>> xs = torch.randn(2, 256, 4, 100, dtype=torch.complex64)
        >>> mask = torch.randn(2, 256, 4, 100)
        >>> psd_matrix = get_power_spectral_density_matrix(xs, mask)
        >>> print(psd_matrix.shape)
        torch.Size([2, 256, 4, 4])

    Note:
        The reduction mode determines how the mask is aggregated across the channels. 
        If "mean" is selected, the mean value is taken across the channel dimension. 
        If "median" is selected, the median value is taken instead.
    """
    if reduction == "mean":
        # Averaging mask along C: (..., C, T) -> (..., 1, T)
        mask = mask.mean(dim=-2, keepdim=True)
    elif reduction == "median":
        mask = mask.median(dim=-2, keepdim=True)
    else:
        raise ValueError("Unknown reduction mode: %s" % reduction)

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., C, C_2)
    psd = einsum("...ct,...et->...ce", xs * mask, xs.conj())

    return psd


def get_rtf(
    psd_speech,
    psd_noise,
    mode="power",
    reference_vector: Union[int, torch.Tensor] = 0,
    iterations: int = 3,
):
    """
    Calculate the relative transfer function (RTF).

    The RTF is calculated using either the power method or eigenvalue 
    decomposition. The algorithm for the power method is as follows:
        1) rtf = reference_vector
        2) for i in range(iterations):
             rtf = (psd_noise^-1 @ psd_speech) @ rtf
             rtf = rtf / ||rtf||_2  # this normalization can be skipped
        3) rtf = psd_noise @ rtf
        4) rtf = rtf / rtf[..., ref_channel, :]

    Note: Normalization at the reference channel is not performed here.

    Args:
        psd_speech (torch.complex64/ComplexTensor):
            Speech covariance matrix with shape (..., F, C, C).
        psd_noise (torch.complex64/ComplexTensor):
            Noise covariance matrix with shape (..., F, C, C).
        mode (str): One of ("power", "evd").
            - "power": Uses the power method.
            - "evd": Uses eigenvalue decomposition.
        reference_vector (torch.Tensor or int): 
            Can be either a tensor of shape (..., C) or a scalar.
        iterations (int): Number of iterations to perform in the power method.

    Returns:
        rtf (torch.complex64/ComplexTensor): 
            The calculated RTF with shape (..., F, C, 1).

    Examples:
        >>> psd_s = torch.randn(10, 8, 4, 4, dtype=torch.complex64)
        >>> psd_n = torch.randn(10, 8, 4, 4, dtype=torch.complex64)
        >>> rtf = get_rtf(psd_s, psd_n, mode="power", iterations=5)
        >>> print(rtf.shape)
        torch.Size([10, 8, 4, 1])
    """
    if mode == "power":
        phi = solve(psd_speech, psd_noise)
        rtf = (
            phi[..., reference_vector, None]
            if isinstance(reference_vector, int)
            else matmul(phi, reference_vector[..., None, :, None])
        )
        for _ in range(iterations - 2):
            rtf = matmul(phi, rtf)
            # rtf = rtf / complex_norm(rtf, dim=-1, keepdim=True)
        rtf = matmul(psd_speech, rtf)
    elif mode == "evd":
        assert (
            is_torch_1_9_plus
            and is_torch_complex_tensor(psd_speech)
            and is_torch_complex_tensor(psd_noise)
        )
        e_vec = generalized_eigenvalue_decomposition(psd_speech, psd_noise)[1]
        rtf = matmul(psd_noise, e_vec[..., -1, None])
    else:
        raise ValueError("Unknown mode: %s" % mode)
    return rtf


def get_mvdr_vector(
    psd_s,
    psd_n,
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """
    Return the MVDR (Minimum Variance Distortionless Response) vector.

    The MVDR vector is computed using the formula:
        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (torch.complex64/ComplexTensor): 
            Speech covariance matrix (..., F, C, C)
        psd_n (torch.complex64/ComplexTensor): 
            Observation/noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor): 
            Reference vector of shape (..., C)
        diagonal_loading (bool): 
            Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float): 
            Regularization term added to the diagonal
        eps (float): 
            Small constant to avoid division by zero

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): 
            MVDR beamforming vector of shape (..., F, C)

    Examples:
        >>> psd_s = torch.rand(10, 8, 4, 4, dtype=torch.complex64)
        >>> psd_n = torch.rand(10, 8, 4, 4, dtype=torch.complex64)
        >>> ref_vec = torch.rand(10, 4)
        >>> mvdr_vector = get_mvdr_vector(psd_s, psd_n, ref_vec)
        >>> print(mvdr_vector.shape)
        torch.Size([10, 8, 4])
    """
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    numerator = solve(psd_s, psd_n)
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_mvdr_vector_with_rtf(
    psd_n: Union[torch.Tensor, ComplexTensor],
    psd_speech: Union[torch.Tensor, ComplexTensor],
    psd_noise: Union[torch.Tensor, ComplexTensor],
    iterations: int = 3,
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Return the MVDR (Minimum Variance Distortionless Response) vector
    calculated with RTF:

        h = (Npsd^-1 @ rtf) / (rtf^H @ Npsd^-1 @ rtf)

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_n (torch.complex64/ComplexTensor):
            observation/noise covariance matrix (..., F, C, C)
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        iterations (int): number of iterations in power method
        reference_vector (torch.Tensor or int): (..., C) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float): regularization factor for diagonal loading
        eps (float): small constant to avoid division by zero

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)

    Examples:
        >>> psd_n = torch.randn(2, 5, 3, 3, dtype=torch.complex64)
        >>> psd_speech = torch.randn(2, 5, 3, 3, dtype=torch.complex64)
        >>> psd_noise = torch.randn(2, 5, 3, 3, dtype=torch.complex64)
        >>> vector = get_mvdr_vector_with_rtf(psd_n, psd_speech, psd_noise)
        >>> print(vector.shape)  # Should print a shape like (2, 5, 3)

    Note:
        - The `reference_vector` can be an index or a tensor used for scaling the output.
        - The function supports both real and complex tensor types.
    """
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    # (B, F, C, 1)
    rtf = get_rtf(
        psd_speech,
        psd_noise,
        mode="power",
        reference_vector=reference_vector,
        iterations=iterations,
    )

    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    numerator = solve(rtf, psd_n).squeeze(-1)
    denominator = einsum("...d,...d->...", rtf.squeeze(-1).conj(), numerator)
    if reference_vector is not None:
        if isinstance(reference_vector, int):
            scale = rtf.squeeze(-1)[..., reference_vector, None].conj()
        else:
            scale = (rtf.squeeze(-1).conj() * reference_vector[..., None, :]).sum(
                dim=-1, keepdim=True
            )
        beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    else:
        beamforming_vector = numerator / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector


def apply_beamforming_vector(
    beamform_vector: Union[torch.Tensor, ComplexTensor],
    mix: Union[torch.Tensor, ComplexTensor],
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Apply the beamforming vector to the mixed signal.

    This function computes the output signal by applying the provided 
    beamforming vector to the mixed input signal. The operation 
    essentially performs a weighted sum of the input channels using the 
    beamforming vector.

    Args:
        beamform_vector (Union[torch.Tensor, ComplexTensor]): 
            The beamforming vector of shape (..., C), where C is the 
            number of channels.
        mix (Union[torch.Tensor, ComplexTensor]): 
            The mixed input signal of shape (..., C, T), where T is the 
            number of time steps.

    Returns:
        Union[torch.Tensor, ComplexTensor]: 
            The output signal of shape (..., T), which is the result of 
            applying the beamforming vector to the mixed signal.

    Examples:
        >>> import torch
        >>> beamform_vector = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
        >>> mix = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.complex64)
        >>> output = apply_beamforming_vector(beamform_vector, mix)
        >>> print(output)
        tensor([[ 1.+0.j,  2.+0.j],
                [ 5.+0.j,  6.+0.j]])
    """
    # (..., C) x (..., C, T) -> (..., T)
    es = einsum("...c,...ct->...t", beamform_vector.conj(), mix)
    return es


def get_mwf_vector(
    psd_s,
    psd_n,
    reference_vector: Union[torch.Tensor, int],
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """
    Return the MWF (Minimum Multi-channel Wiener Filter) vector.

    The MWF vector is calculated using the formula:

        h = (Npsd^-1 @ Spsd) @ u

    where:
        - Npsd is the noise covariance matrix (psd_n).
        - Spsd is the speech covariance matrix (psd_s).
        - u is the reference vector.

    Args:
        psd_s (torch.complex64/ComplexTensor): 
            Speech covariance matrix with shape (..., F, C, C).
        psd_n (torch.complex64/ComplexTensor): 
            Power-normalized observation covariance matrix with shape (..., F, C, C).
        reference_vector (torch.Tensor or int): 
            Reference vector with shape (..., C) or a scalar index.
        diagonal_loading (bool): 
            Whether to add a tiny term to the diagonal of psd_n to avoid singularities.
        diag_eps (float): 
            Regularization term added to the diagonal if diagonal_loading is True.
        eps (float): 
            Small constant to prevent division by zero.

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): 
            The calculated MWF vector with shape (..., F, C).

    Examples:
        >>> psd_s = torch.rand(2, 4, 3, 3, dtype=torch.complex64)
        >>> psd_n = torch.rand(2, 4, 3, 3, dtype=torch.complex64)
        >>> reference_vector = torch.tensor([1.0, 0.0, 0.0])
        >>> mwf_vector = get_mwf_vector(psd_s, psd_n, reference_vector)

    Note:
        - The function assumes that psd_s and psd_n are complex tensors.
        - The reference_vector can be provided either as a tensor or an index.
    """
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    ws = solve(psd_s, psd_n)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_sdw_mwf_vector(
    psd_speech,
    psd_noise,
    reference_vector: Union[torch.Tensor, int],
    denoising_weight: float = 1.0,
    approx_low_rank_psd_speech: bool = False,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """
    Return the SDW-MWF (Speech Distortion Weighted Multi-channel Wiener Filter) vector.

    The formula for the SDW-MWF is given by:
        h = (Spsd + mu * Npsd)^-1 @ Spsd @ u

    This filter emphasizes the preservation of speech while reducing noise.

    References:
        [1] Spatially pre-processed speech distortion weighted multi-channel
        Wiener filtering for noise reduction; A. Spriet et al., 2004
        https://dl.acm.org/doi/abs/10.1016/j.sigpro.2004.07.028
        [2] Rank-1 constrained multichannel Wiener filter for speech recognition in
        noisy environments; Z. Wang et al., 2018
        https://hal.inria.fr/hal-01634449/document
        [3] Low-rank approximation based multichannel Wiener filter algorithms for
        noise reduction with application in cochlear implants; R. Serizel, 2014
        https://ieeexplore.ieee.org/document/6730918

    Args:
        psd_speech (torch.complex64/ComplexTensor): 
            Speech covariance matrix with shape (..., F, C, C).
        psd_noise (torch.complex64/ComplexTensor): 
            Noise covariance matrix with shape (..., F, C, C).
        reference_vector (torch.Tensor or int): 
            Reference vector with shape (..., C) or scalar.
        denoising_weight (float): 
            Trade-off parameter between noise reduction and speech distortion.
            A larger value leads to more noise reduction at the expense of
            more speech distortion. The plain MWF is obtained with
            `denoising_weight = 1` (default).
        approx_low_rank_psd_speech (bool): 
            Whether to replace original input psd_speech with its low-rank
            approximation as in [2].
        iterations (int): 
            Number of iterations in power method, only used when
            `approx_low_rank_psd_speech = True`.
        diagonal_loading (bool): 
            Whether to add a tiny term to the diagonal of psd_n.
        diag_eps (float): 
            Regularization factor for diagonal loading.
        eps (float): 
            Small constant to prevent division by zero.

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): 
            The computed beamforming vector with shape (..., F, C).

    Examples:
        >>> psd_s = torch.randn(1, 8, 2, 2, dtype=torch.complex64)
        >>> psd_n = torch.randn(1, 8, 2, 2, dtype=torch.complex64)
        >>> ref_vector = torch.tensor([1, 0], dtype=torch.complex64)
        >>> vector = get_sdw_mwf_vector(psd_s, psd_n, ref_vector)
        >>> print(vector.shape)
        torch.Size([1, 8, 2])
    """
    if approx_low_rank_psd_speech:
        if diagonal_loading:
            psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

        # (B, F, C, 1)
        recon_vec = get_rtf(
            psd_speech,
            psd_noise,
            mode="power",
            iterations=iterations,
            reference_vector=reference_vector,
        )
        # Eq. (25) in Ref[2]
        psd_speech_r1 = matmul(recon_vec, recon_vec.conj().transpose(-1, -2))
        sigma_speech = FC.trace(psd_speech) / (FC.trace(psd_speech_r1) + eps)
        psd_speech_r1 = psd_speech_r1 * sigma_speech[..., None, None]
        # c.f. Eq. (62) in Ref[3]
        psd_speech = psd_speech_r1

    psd_n = psd_speech + denoising_weight * psd_noise
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    ws = solve(psd_speech, psd_n)

    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_rank1_mwf_vector(
    psd_speech,
    psd_noise,
    reference_vector: Union[torch.Tensor, int],
    denoising_weight: float = 1.0,
    approx_low_rank_psd_speech: bool = False,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """
    Return the R1-MWF (Rank-1 Multi-channel Wiener Filter) vector.

    The R1-MWF is calculated using the formula:

        h = (Npsd^-1 @ Spsd) / (mu + Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        [1] Rank-1 constrained multichannel Wiener filter for speech recognition in
        noisy environments; Z. Wang et al, 2018
        https://hal.inria.fr/hal-01634449/document
        [2] Low-rank approximation based multichannel Wiener filter algorithms for
        noise reduction with application in cochlear implants; R. Serizel, 2014
        https://ieeexplore.ieee.org/document/6730918

    Args:
        psd_speech (torch.complex64/ComplexTensor):
            Speech covariance matrix (..., F, C, C).
        psd_noise (torch.complex64/ComplexTensor):
            Noise covariance matrix (..., F, C, C).
        reference_vector (torch.Tensor or int): 
            Reference vector, either (..., C) or a scalar index.
        denoising_weight (float): 
            Trade-off parameter between noise reduction and speech distortion.
            A larger value leads to more noise reduction at the expense of more 
            speech distortion. When `denoising_weight = 0`, it corresponds to 
            the MVDR beamformer.
        approx_low_rank_psd_speech (bool): 
            Whether to replace original input `psd_speech` with its low-rank 
            approximation as in [1].
        iterations (int): 
            Number of iterations in power method, only used when 
            `approx_low_rank_psd_speech = True`.
        diagonal_loading (bool): 
            Whether to add a tiny term to the diagonal of `psd_n`.
        diag_eps (float): 
            Regularization factor for diagonal loading.
        eps (float): 
            Small constant to avoid division by zero.

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): 
            Beamforming vector of shape (..., F, C).

    Examples:
        >>> psd_speech = torch.rand(10, 8, 4, 4, dtype=torch.complex64)
        >>> psd_noise = torch.rand(10, 8, 4, 4, dtype=torch.complex64)
        >>> ref_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        >>> result = get_rank1_mwf_vector(psd_speech, psd_noise, ref_vec)
        >>> print(result.shape)
        torch.Size([10, 8, 4])

    Note:
        - The function utilizes Tikhonov regularization to stabilize the 
          inversion of the noise covariance matrix.
    """
    if approx_low_rank_psd_speech:
        if diagonal_loading:
            psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

        # (B, F, C, 1)
        recon_vec = get_rtf(
            psd_speech,
            psd_noise,
            mode="power",
            iterations=iterations,
            reference_vector=reference_vector,
        )
        # Eq. (25) in Ref[1]
        psd_speech_r1 = matmul(recon_vec, recon_vec.conj().transpose(-1, -2))
        sigma_speech = FC.trace(psd_speech) / (FC.trace(psd_speech_r1) + eps)
        psd_speech_r1 = psd_speech_r1 * sigma_speech[..., None, None]
        # c.f. Eq. (62) in Ref[2]
        psd_speech = psd_speech_r1
    elif diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    numerator = solve(psd_speech, psd_noise)

    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (denoising_weight + FC.trace(numerator)[..., None, None] + eps)

    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_rtf_matrix(
    psd_speeches,
    psd_noises,
    diagonal_loading: bool = True,
    ref_channel: int = 0,
    rtf_iterations: int = 3,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """
    Calculate the RTF matrix with each column being the relative transfer 
    function of the corresponding source.

    This function computes the relative transfer function (RTF) matrix for a 
    set of speech covariance matrices and noise covariance matrices. Each 
    column of the resulting matrix corresponds to the RTF of a specific 
    speech source relative to the specified reference channel.

    Args:
        psd_speeches (list): A list of speech covariance matrices, where 
            each matrix has the shape (..., F, C, C).
        psd_noises (list): A list of noise covariance matrices, where each 
            matrix has the shape (..., F, C, C).
        diagonal_loading (bool): If True, adds a small regularization term 
            to the diagonal of the noise covariance matrices.
        ref_channel (int): The index of the reference channel for RTF 
            normalization.
        rtf_iterations (int): The number of iterations for the RTF 
            computation.
        diag_eps (float): The regularization factor to be added to the 
            diagonal when diagonal loading is enabled.
        eps (float): A small constant to avoid division by zero.

    Returns:
        torch.complex64/ComplexTensor: The RTF matrix with shape 
            (..., F, C, num_spk), where num_spk is the number of speech 
            sources.

    Note:
        - The function assumes that `psd_speeches` and `psd_noises` are 
          both lists of the same length, corresponding to the number of 
          sources.
        - The output RTF matrix is normalized at the reference channel.

    Examples:
        >>> psd_speeches = [torch.rand(10, 4, 4), torch.rand(10, 4, 4)]
        >>> psd_noises = [torch.rand(10, 4, 4), torch.rand(10, 4, 4)]
        >>> rtf_matrix = get_rtf_matrix(psd_speeches, psd_noises)
    """
    assert isinstance(psd_speeches, list) and isinstance(psd_noises, list)
    rtf_mat = cat(
        [
            get_rtf(
                psd_speeches[spk],
                tik_reg(psd_n, reg=diag_eps, eps=eps) if diagonal_loading else psd_n,
                mode="power",
                reference_vector=ref_channel,
                iterations=rtf_iterations,
            )
            for spk, psd_n in enumerate(psd_noises)
        ],
        dim=-1,
    )
    # normalize at the reference channel
    return rtf_mat / rtf_mat[..., ref_channel, None, :]


def get_lcmv_vector_with_rtf(
    psd_n: Union[torch.Tensor, ComplexTensor],
    rtf_mat: Union[torch.Tensor, ComplexTensor],
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Return the LCMV (Linearly Constrained Minimum Variance) vector
        calculated with RTF:

        h = (Npsd^-1 @ rtf_mat) @ (rtf_mat^H @ Npsd^-1 @ rtf_mat)^-1 @ p

    Reference:
        H. L. Van Trees, “Optimum array processing: Part IV of detection, estimation,
        and modulation theory,” John Wiley & Sons, 2004. (Chapter 6.7)

    Args:
        psd_n (torch.complex64/ComplexTensor):
            observation/noise covariance matrix (..., F, C, C)
        rtf_mat (torch.complex64/ComplexTensor):
            RTF matrix (..., F, C, num_spk)
        reference_vector (torch.Tensor or int): (..., num_spk) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float): Regularization factor for diagonal loading
        eps (float): Small constant to avoid division by zero

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    
    Examples:
        >>> psd_n = torch.randn(2, 4, 3, 3, dtype=torch.complex64)
        >>> rtf_mat = torch.randn(2, 4, 3, 5, dtype=torch.complex64)
        >>> reference_vector = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex64)
        >>> lcmv_vector = get_lcmv_vector_with_rtf(psd_n, rtf_mat, reference_vector)
    """
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, num_spk) -> (..., C_1, num_spk)
    numerator = solve(rtf_mat, psd_n)
    denominator = matmul(rtf_mat.conj().transpose(-1, -2), numerator)
    if isinstance(reference_vector, int):
        ws = inverse(denominator)[..., reference_vector, None]
    else:
        ws = solve(reference_vector, denominator)
    beamforming_vector = matmul(numerator, ws).squeeze(-1)
    return beamforming_vector


def generalized_eigenvalue_decomposition(a: torch.Tensor, b: torch.Tensor, eps=1e-6):
    """
    Solves the generalized eigenvalue decomposition through Cholesky decomposition.

    This function computes the generalized eigenvalue decomposition of matrices `a` 
    and `b`, such that:

        a @ e_vec = e_val * b @ e_vec

    The method utilizes Cholesky decomposition on matrix `b` to transform the problem 
    into a standard eigenvalue problem.

    Steps involved:
    1. Perform Cholesky decomposition on `b`: 
       b = L @ L^H, where `L` is a lower triangular matrix.
    2. Define C = L^-1 @ a @ L^-H, which is Hermitian.
    3. Solve the eigenvalue problem C @ y = lambda * y.
    4. Obtain the eigenvectors e_vec = L^-H @ y.

    Reference: https://www.netlib.org/lapack/lug/node54.html

    Args:
        a (torch.Tensor): A complex Hermitian or real symmetric matrix whose 
            eigenvalues and eigenvectors will be computed. Shape: (..., C, C).
        b (torch.Tensor): A complex Hermitian or real symmetric definite positive 
            matrix. Shape: (..., C, C).
        eps (float, optional): A small value for numerical stability during 
            Cholesky decomposition. Default is 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - e_val (torch.Tensor): Generalized eigenvalues (ascending order).
            - e_vec (torch.Tensor): Generalized eigenvectors.

    Examples:
        >>> a = torch.tensor([[1, 0], [0, 2]], dtype=torch.complex64)
        >>> b = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
        >>> e_val, e_vec = generalized_eigenvalue_decomposition(a, b)
        >>> print(e_val)
        >>> print(e_vec)

    Note:
        The input matrices `a` and `b` must be Hermitian or symmetric as per the 
        requirements of eigenvalue decomposition.
    """
    try:
        cholesky = torch.linalg.cholesky(b)
    except RuntimeError:
        b = tik_reg(b, reg=eps, eps=eps)
        cholesky = torch.linalg.cholesky(b)
    inv_cholesky = cholesky.inverse()
    # Compute C matrix L⁻1 a L^-H
    cmat = inv_cholesky @ a @ inv_cholesky.conj().transpose(-1, -2)
    # Performing the eigenvalue decomposition
    e_val, e_vec = torch.linalg.eigh(cmat)
    # Collecting the eigenvectors
    e_vec = torch.matmul(inv_cholesky.conj().transpose(-1, -2), e_vec)
    return e_val, e_vec


def gev_phase_correction(vector):
    """
    Phase correction to reduce distortions due to phase inconsistencies.

    This function applies phase correction to a beamforming vector to
    minimize distortions that arise from phase inconsistencies across 
    different frequency channels. It is particularly useful in scenarios 
    involving generalized eigenvalue (GEV) beamformers.

    Args:
        vector (torch.complex64/ComplexTensor): 
            Beamforming vector with shape (..., F, C), where F is the number
            of frequency bins and C is the number of channels.

    Returns:
        torch.complex64/ComplexTensor: 
            Phase corrected beamforming vectors with the same shape as the input.

    Examples:
        >>> import torch
        >>> vector = torch.rand(2, 5, 3, dtype=torch.complex64)  # (B, F, C)
        >>> corrected_vector = gev_phase_correction(vector)
        >>> print(corrected_vector.shape)
        torch.Size([2, 5, 3])

    Note:
        The function computes a correction factor for each frequency 
        channel based on the phase angle of the beamforming vector. The 
        correction is applied in the complex domain, adjusting the phase 
        while keeping the magnitude intact.
    """
    B, F, C = vector.shape
    correction = torch.empty_like(vector.real)
    for f in range(F):
        correction[:, f, :] = torch.exp(
            (vector[:, f, :] * vector[:, f - 1, :].conj())
            .sum(dim=-1, keepdim=True)
            .angle()
        )
    if isinstance(vector, ComplexTensor):
        correction = ComplexTensor(torch.cos(correction), -torch.sin(correction))
    else:
        correction = torch.exp(-1j * correction)
    return vector * correction


def blind_analytic_normalization(ws, psd_noise, eps=1e-8):
    """
    Blind analytic normalization (BAN) for post-filtering.

    This function normalizes the beamformer vector using blind analytic
    normalization, which is useful in the context of speech enhancement
    and noise reduction.

    Args:
        ws (torch.complex64/ComplexTensor): 
            Beamformer vector of shape (..., F, C).
        psd_noise (torch.complex64/ComplexTensor): 
            Noise Power Spectral Density (PSD) matrix of shape (..., F, C, C).
        eps (float, optional): 
            A small constant to avoid division by zero. Default is 1e-8.

    Returns:
        ws_ban (torch.complex64/ComplexTensor): 
            Normalized beamformer vector of shape (..., F).

    Examples:
        >>> ws = torch.randn(2, 3, 4, dtype=torch.complex64)
        >>> psd_noise = torch.randn(2, 3, 4, 4, dtype=torch.complex64)
        >>> normalized_ws = blind_analytic_normalization(ws, psd_noise)

    Note:
        The normalization helps in enhancing the robustness of the
        beamformer against noise and improves the overall performance
        in applications like speech recognition and enhancement.
    """
    C2 = psd_noise.size(-1) ** 2
    denominator = einsum("...c,...ce,...e->...", ws.conj(), psd_noise, ws)
    numerator = einsum(
        "...c,...ce,...eo,...o->...", ws.conj(), psd_noise, psd_noise, ws
    )
    gain = (numerator + eps).sqrt() / (denominator * C2 + eps)
    return gain


def get_gev_vector(
    psd_noise: Union[torch.Tensor, ComplexTensor],
    psd_speech: Union[torch.Tensor, ComplexTensor],
    mode="power",
    reference_vector: Union[int, torch.Tensor] = 0,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Return the generalized eigenvalue (GEV) beamformer vector.

    The GEV beamformer is defined by the equation:
        psd_speech @ h = lambda * psd_noise @ h

    Reference:
        Blind acoustic beamforming based on generalized eigenvalue 
        decomposition; E. Warsitz and R. Haeb-Umbach, 2007.

    Args:
        psd_noise (torch.complex64/ComplexTensor): 
            Noise covariance matrix of shape (..., F, C, C).
        psd_speech (torch.complex64/ComplexTensor): 
            Speech covariance matrix of shape (..., F, C, C).
        mode (str): One of ("power", "evd"). 
            - "power": Power method for calculating the eigenvector.
            - "evd": Eigenvalue decomposition (only for torch builtin 
              complex tensors).
        reference_vector (torch.Tensor or int): 
            Reference vector of shape (..., C) or a scalar.
        iterations (int): Number of iterations for the power method.
        diagonal_loading (bool): Whether to add a small term to the 
            diagonal of psd_noise to prevent singularity.
        diag_eps (float): Regularization parameter for diagonal loading.
        eps (float): Small constant to avoid division by zero.

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): 
            GEV beamformer vector of shape (..., F, C).

    Examples:
        >>> psd_n = torch.rand(10, 8, 4, 4, dtype=torch.complex64)
        >>> psd_s = torch.rand(10, 8, 4, 4, dtype=torch.complex64)
        >>> vector = get_gev_vector(psd_n, psd_s, mode="power")
        >>> print(vector.shape)
        torch.Size([10, 8, 4])
    """
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    if mode == "power":
        phi = solve(psd_speech, psd_noise)
        e_vec = (
            phi[..., reference_vector, None]
            if isinstance(reference_vector, int)
            else matmul(phi, reference_vector[..., None, :, None])
        )
        for _ in range(iterations - 1):
            e_vec = matmul(phi, e_vec)
            # e_vec = e_vec / complex_norm(e_vec, dim=-1, keepdim=True)
        e_vec = e_vec.squeeze(-1)
    elif mode == "evd":
        assert (
            is_torch_1_9_plus
            and is_torch_complex_tensor(psd_speech)
            and is_torch_complex_tensor(psd_noise)
        )
        # e_vec = generalized_eigenvalue_decomposition(psd_speech, psd_noise)[1][...,-1]
        e_vec = psd_noise.new_zeros(psd_noise.shape[:-1])
        for f in range(psd_noise.shape[-3]):
            try:
                e_vec[..., f, :] = generalized_eigenvalue_decomposition(
                    psd_speech[..., f, :, :], psd_noise[..., f, :, :]
                )[1][..., -1]
            except RuntimeError:
                # port from github.com/fgnt/nn-gev/blob/master/fgnt/beamforming.py#L106
                print(
                    "GEV beamformer: LinAlg error for frequency {}".format(f),
                    flush=True,
                )
                C = psd_noise.size(-1)
                e_vec[..., f, :] = (
                    psd_noise.new_ones(e_vec[..., f, :].shape)
                    / FC.trace(psd_noise[..., f, :, :])
                    * C
                )
    else:
        raise ValueError("Unknown mode: %s" % mode)

    beamforming_vector = e_vec / complex_norm(e_vec, dim=-1, keepdim=True)
    beamforming_vector = gev_phase_correction(beamforming_vector)
    return beamforming_vector


def signal_framing(
    signal: Union[torch.Tensor, ComplexTensor],
    frame_length: int,
    frame_step: int,
    bdelay: int,
    do_padding: bool = False,
    pad_value: int = 0,
    indices: List = None,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Expand `signal` into several frames, with each frame of length 
    `frame_length`.

    This function divides a given signal into overlapping frames of a specified 
    length and step size. It is particularly useful in speech processing and 
    signal analysis where segmenting the signal into smaller parts is required 
    for further processing. If padding is enabled, the signal can be padded at 
    the beginning to accommodate the specified delay.

    Args:
        signal (Union[torch.Tensor, ComplexTensor]): 
            The input signal to be framed with shape (..., T), where T is the 
            length of the signal.
        frame_length (int): 
            The length of each segment (frame) to be extracted from the signal.
        frame_step (int): 
            The step size for moving the frame across the signal.
        bdelay (int): 
            Delay for WPD (Weighted Power Distortionless response).
        do_padding (bool, optional): 
            Whether or not to pad the input signal at the beginning of the 
            time dimension. Default is False.
        pad_value (int, optional): 
            The value to fill in the padding if `do_padding` is True. 
            Default is 0.
        indices (List, optional): 
            Pre-computed indices for extracting frames. If None, indices will 
            be computed based on `frame_length` and `frame_step`.

    Returns:
        Union[torch.Tensor, ComplexTensor]: 
            If `do_padding` is True, returns a tensor of shape (..., T, 
            frame_length) where T is the length of the padded signal. If 
            `do_padding` is False, returns a tensor of shape (..., T - 
            bdelay - frame_length + 2, frame_length), which represents the 
            framed signal segments.

    Examples:
        >>> signal = torch.randn(10)  # Example signal of length 10
        >>> frames = signal_framing(signal, frame_length=4, frame_step=2, 
        ...                          bdelay=1)
        >>> print(frames.shape)  # Output: (4, 4) for non-padding case

        >>> padded_frames = signal_framing(signal, frame_length=4, 
        ...                                frame_step=2, bdelay=1, 
        ...                                do_padding=True)
        >>> print(padded_frames.shape)  # Output: (5, 4) for padding case
    """
    if isinstance(signal, ComplexTensor):
        complex_wrapper = ComplexTensor
        pad_func = FC.pad
    elif is_torch_complex_tensor(signal):
        complex_wrapper = torch.complex
        pad_func = torch.nn.functional.pad
    else:
        pad_func = torch.nn.functional.pad

    frame_length2 = frame_length - 1
    # pad to the right at the last dimension of `signal` (time dimension)
    if do_padding:
        # (..., T) --> (..., T + bdelay + frame_length - 2)
        signal = pad_func(
            signal, (bdelay + frame_length2 - 1, 0), "constant", pad_value
        )
        do_padding = False

    if indices is None:
        # [[ 0, 1, ..., frame_length2 - 1,              frame_length2 - 1 + bdelay ],
        #  [ 1, 2, ..., frame_length2,                  frame_length2 + bdelay     ],
        #  [ 2, 3, ..., frame_length2 + 1,              frame_length2 + 1 + bdelay ],
        #  ...
        #  [ T-bdelay-frame_length2, ..., T-1-bdelay,   T-1 ]]
        indices = [
            [*range(i, i + frame_length2), i + frame_length2 + bdelay - 1]
            for i in range(0, signal.shape[-1] - frame_length2 - bdelay + 1, frame_step)
        ]

    if is_complex(signal):
        real = signal_framing(
            signal.real,
            frame_length,
            frame_step,
            bdelay,
            do_padding,
            pad_value,
            indices,
        )
        imag = signal_framing(
            signal.imag,
            frame_length,
            frame_step,
            bdelay,
            do_padding,
            pad_value,
            indices,
        )
        return complex_wrapper(real, imag)
    else:
        # (..., T - bdelay - frame_length + 2, frame_length)
        signal = signal[..., indices]
        return signal


def get_covariances(
    Y: Union[torch.Tensor, ComplexTensor],
    inverse_power: torch.Tensor,
    bdelay: int,
    btaps: int,
    get_vector: bool = False,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Calculates the power normalized spatio-temporal covariance matrix 
    of the framed signal.

    Args:
        Y : Complex STFT signal with shape (B, F, C, T)
        inverse_power : Weighting factor with shape (B, F, T)
        bdelay (int): Delay for WPD.
        btaps (int): Number of taps.
        get_vector (bool): If True, returns both the covariance matrix 
            and the covariance vector.

    Returns:
        If `get_vector` is True, returns:
            - Correlation matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
            - Correlation vector: (B, F, btaps + 1, C, C)
        Otherwise, returns:
            - Correlation matrix: (B, F, (btaps + 1) * C, (btaps + 1) * C)

    Raises:
        AssertionError: If `inverse_power` does not have 3 dimensions or 
            its size does not match with `Y`.

    Examples:
        >>> Y = torch.randn(2, 4, 3, 10, dtype=torch.complex64)  # (B, F, C, T)
        >>> inverse_power = torch.rand(2, 4, 10)  # (B, F, T)
        >>> cov_matrix = get_covariances(Y, inverse_power, bdelay=2, btaps=3)
        >>> cov_matrix.shape
        torch.Size([2, 4, 16, 16])  # (B, F, (btaps + 1) * C, (btaps + 1) * C)

        >>> cov_matrix, cov_vector = get_covariances(Y, inverse_power, 
        ...     bdelay=2, btaps=3, get_vector=True)
        >>> cov_vector.shape
        torch.Size([2, 4, 4, 3, 3])  # (B, F, btaps + 1, C, C)

    Note:
        - The `bdelay` and `btaps` parameters control the delay and 
          the number of taps in the covariance calculation.
    """
    assert inverse_power.dim() == 3, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), (inverse_power.size(0), Y.size(0))

    Bs, Fdim, C, T = Y.shape

    # (B, F, C, T - bdelay - btaps + 1, btaps + 1)
    Psi = signal_framing(Y, btaps + 1, 1, bdelay, do_padding=False)[
        ..., : T - bdelay - btaps + 1, :
    ]
    # Reverse along btaps-axis:
    # [tau, tau-bdelay, tau-bdelay-1, ..., tau-bdelay-frame_length+1]
    Psi = reverse(Psi, dim=-1)
    Psi_norm = Psi * inverse_power[..., None, bdelay + btaps - 1 :, None]

    # let T' = T - bdelay - btaps + 1
    # (B, F, C, T', btaps + 1) x (B, F, C, T', btaps + 1)
    #  -> (B, F, btaps + 1, C, btaps + 1, C)
    covariance_matrix = einsum("bfdtk,bfetl->bfkdle", Psi, Psi_norm.conj())

    # (B, F, btaps + 1, C, btaps + 1, C)
    #   -> (B, F, (btaps + 1) * C, (btaps + 1) * C)
    covariance_matrix = covariance_matrix.view(
        Bs, Fdim, (btaps + 1) * C, (btaps + 1) * C
    )

    if get_vector:
        # (B, F, C, T', btaps + 1) x (B, F, C, T')
        #    --> (B, F, btaps +1, C, C)
        covariance_vector = einsum(
            "bfdtk,bfet->bfked", Psi_norm, Y[..., bdelay + btaps - 1 :].conj()
        )
        return covariance_matrix, covariance_vector
    else:
        return covariance_matrix


def get_WPD_filter(
    Phi: Union[torch.Tensor, ComplexTensor],
    Rf: Union[torch.Tensor, ComplexTensor],
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Return the WPD (Weighted Power Distortionless response) vector.

    The WPD vector is computed as follows:

        h = (Rf^-1 @ Phi_{xx}) / tr[(Rf^-1) @ Phi_{xx}] @ u

    This implementation follows the method described in the reference below.

    Reference:
        T. Nakatani and K. Kinoshita, "A Unified Convolutional Beamformer
        for Simultaneous Denoising and Dereverberation," in IEEE Signal
        Processing Letters, vol. 26, no. 6, pp. 903-907, June 2019, doi:
        10.1109/LSP.2019.2911179.
        https://ieeexplore.ieee.org/document/8691481

    Args:
        Phi (torch.complex64/ComplexTensor): 
            (B, F, (btaps+1) * C, (btaps+1) * C) 
            is the PSD of zero-padded speech [x^T(t,f) 0 ... 0]^T.
        Rf (torch.complex64/ComplexTensor): 
            (B, F, (btaps+1) * C, (btaps+1) * C) 
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): 
            (B, (btaps+1) * C) 
            is the reference vector.
        diagonal_loading (bool): 
            Whether to add a tiny term to the diagonal of Rf.
        diag_eps (float): 
            Small value for regularization.
        eps (float): 
            Small value to prevent division by zero.

    Returns:
        filter_matrix (torch.complex64/ComplexTensor): 
            (B, F, (btaps + 1) * C) 
            the computed WPD filter matrix.
    """
    if diagonal_loading:
        Rf = tik_reg(Rf, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = solve(Phi, Rf)
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    # (B, F, (btaps + 1) * C)
    return beamform_vector


def get_WPD_filter_v2(
    Phi: Union[torch.Tensor, ComplexTensor],
    Rf: Union[torch.Tensor, ComplexTensor],
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Return the WPD vector (v2).

    This implementation is more efficient than `get_WPD_filter` as it skips
    unnecessary computation with zeros.

    WPD stands for Weighted Power minimization Distortionless response 
    convolutional beamformer, which is defined as:

        h = (Rf^-1 @ Phi_{xx}) / tr[(Rf^-1) @ Phi_{xx}] @ u

    Reference:
        T. Nakatani and K. Kinoshita, "A Unified Convolutional Beamformer
        for Simultaneous Denoising and Dereverberation," in IEEE Signal
        Processing Letters, vol. 26, no. 6, pp. 903-907, June 2019, doi:
        10.1109/LSP.2019.2911179.
        https://ieeexplore.ieee.org/document/8691481

    Args:
        Phi (torch.complex64/ComplexTensor): (B, F, C, C)
            is speech PSD.
        Rf (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, C)
            is the reference_vector.
        diagonal_loading (bool): 
            Whether to add a tiny term to the diagonal of psd_n.
        diag_eps (float): 
            Small constant for regularization.
        eps (float): 
            Small constant to prevent division by zero.

    Returns:
        filter_matrix (torch.complex64/ComplexTensor): (B, F, (btaps + 1) * C)
            The computed WPD filter matrix.
    """
    C = reference_vector.shape[-1]
    if diagonal_loading:
        Rf = tik_reg(Rf, reg=diag_eps, eps=eps)
    inv_Rf = inverse(Rf)
    # (B, F, (btaps+1) * C, C)
    inv_Rf_pruned = inv_Rf[..., :C]
    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = matmul(inv_Rf_pruned, Phi)
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., (btaps+1) * C, C) / (...,) -> (..., (btaps+1) * C, C)
    ws = numerator / (FC.trace(numerator[..., :C, :])[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    # (B, F, (btaps+1) * C)
    return beamform_vector


def get_WPD_filter_with_rtf(
    psd_observed_bar: Union[torch.Tensor, ComplexTensor],
    psd_speech: Union[torch.Tensor, ComplexTensor],
    psd_noise: Union[torch.Tensor, ComplexTensor],
    iterations: int = 3,
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-15,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Return the WPD vector calculated with RTF.

    WPD is the Weighted Power minimization Distortionless response
    convolutional beamformer. As follows:

        h = (Rf^-1 @ vbar) / (vbar^H @ R^-1 @ vbar)

    Reference:
        T. Nakatani and K. Kinoshita, "A Unified Convolutional Beamformer
        for Simultaneous Denoising and Dereverberation," in IEEE Signal
        Processing Letters, vol. 26, no. 6, pp. 903-907, June 2019, doi:
        10.1109/LSP.2019.2911179.
        https://ieeexplore.ieee.org/document/8691481

    Args:
        psd_observed_bar (torch.complex64/ComplexTensor):
            Stacked observation covariance matrix.
        psd_speech (torch.complex64/ComplexTensor):
            Speech covariance matrix (..., F, C, C).
        psd_noise (torch.complex64/ComplexTensor):
            Noise covariance matrix (..., F, C, C).
        iterations (int): Number of iterations in power method.
        reference_vector (torch.Tensor or int): (..., C) or scalar.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n.
        diag_eps (float): Regularization factor for diagonal loading.
        eps (float): Small constant to prevent division by zero.

    Returns:
        beamform_vector (torch.complex64/ComplexTensor): The resulting WPD 
        filter vector (..., F, C).

    Examples:
        >>> psd_observed_bar = torch.randn(4, 8, 10, 10, dtype=torch.complex64)
        >>> psd_speech = torch.randn(4, 8, 10, 10, dtype=torch.complex64)
        >>> psd_noise = torch.randn(4, 8, 10, 10, dtype=torch.complex64)
        >>> filter_vector = get_WPD_filter_with_rtf(psd_observed_bar, psd_speech, psd_noise)

    Note:
        Ensure that the input tensors are of appropriate shapes and types.
    """
    if isinstance(psd_speech, ComplexTensor):
        pad_func = FC.pad
    elif is_torch_complex_tensor(psd_speech):
        pad_func = torch.nn.functional.pad
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )

    C = psd_noise.size(-1)
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    # (B, F, C, 1)
    rtf = get_rtf(
        psd_speech,
        psd_noise,
        mode="power",
        reference_vector=reference_vector,
        iterations=iterations,
    )

    # (B, F, (K+1)*C, 1)
    rtf = pad_func(rtf, (0, 0, 0, psd_observed_bar.shape[-1] - C), "constant", 0)
    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    numerator = solve(rtf, psd_observed_bar).squeeze(-1)
    denominator = einsum("...d,...d->...", rtf.squeeze(-1).conj(), numerator)
    if reference_vector is not None:
        if isinstance(reference_vector, int):
            scale = rtf.squeeze(-1)[..., reference_vector, None].conj()
        else:
            scale = (
                rtf.squeeze(-1)[:, :, :C].conj() * reference_vector[..., None, :]
            ).sum(dim=-1, keepdim=True)
        beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    else:
        beamforming_vector = numerator / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector


def perform_WPD_filtering(
    filter_matrix: Union[torch.Tensor, ComplexTensor],
    Y: Union[torch.Tensor, ComplexTensor],
    bdelay: int,
    btaps: int,
) -> Union[torch.Tensor, ComplexTensor]:
    """
    Perform WPD filtering.

    This function applies the Weighted Power Minimization (WPD) filtering
    to a complex Short-Time Fourier Transform (STFT) signal using the
    provided filter matrix. The WPD filter is designed to enhance the
    target signal while minimizing distortion.

    Args:
        filter_matrix: A filter matrix of shape (B, F, (btaps + 1) * C) 
            where B is the batch size, F is the number of frequency bins,
            C is the number of channels, and btaps is the number of filter
            taps.
        Y: A complex STFT signal with shape (B, F, C, T) where T is the
            number of time frames.

    Returns:
        enhanced (torch.complex64/ComplexTensor): A tensor of shape (B, F, T)
            representing the enhanced signal after applying the WPD filter.

    Examples:
        >>> filter_matrix = torch.randn(2, 256, 6)  # Example filter
        >>> Y = torch.randn(2, 256, 2, 100)          # Example STFT signal
        >>> enhanced_signal = perform_WPD_filtering(filter_matrix, Y, 3, 5)
        >>> print(enhanced_signal.shape)
        torch.Size([2, 256, 100])

    Note:
        - The input STFT signal should be properly shaped and the filter
          matrix should correspond to the dimensions of the STFT signal.
    """
    # (B, F, C, T) --> (B, F, C, T, btaps + 1)
    Ytilde = signal_framing(Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Ytilde = reverse(Ytilde, dim=-1)

    Bs, Fdim, C, T = Y.shape
    # --> (B, F, T, btaps + 1, C) --> (B, F, T, (btaps + 1) * C)
    Ytilde = Ytilde.permute(0, 1, 3, 4, 2).contiguous().view(Bs, Fdim, T, -1)
    # (B, F, T, 1)
    enhanced = einsum("...tc,...c->...t", Ytilde, filter_matrix.conj())
    return enhanced


def tik_reg(mat, reg: float = 1e-8, eps: float = 1e-8):
    """
    Perform Tikhonov regularization on a complex matrix by modifying its real part.

Tikhonov regularization, also known as ridge regression, is a technique used to
stabilize the solution of ill-posed problems by adding a regularization term. 
This function specifically targets the real part of the input complex matrix 
and adds a scaled identity matrix to it.

Args:
    mat (torch.complex64/ComplexTensor): Input matrix of shape (..., C, C),
        where C is the number of channels.
    reg (float): Regularization factor that determines the strength of the 
        regularization. A higher value applies more regularization.
    eps (float): A small constant added to prevent division by zero or 
        ensure numerical stability.

Returns:
    ret (torch.complex64/ComplexTensor): Regularized matrix of shape (..., C, C).

Note:
    The regularization is applied only to the real part of the matrix. The 
    imaginary part remains unchanged. This is particularly useful in scenarios 
    involving covariance matrices in beamforming applications, where the real 
    part may become ill-conditioned.

Examples:
    >>> import torch
    >>> mat = torch.tensor([[1.0 + 2.0j, 0.0 + 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]])
    >>> regularized_mat = tik_reg(mat, reg=0.1, eps=1e-8)
    >>> print(regularized_mat)
    """
    # Add eps
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    shape = [1 for _ in range(mat.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(*mat.shape[:-2], 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(mat).real[..., None, None] * reg
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps
    mat = mat + epsilon * eye
    return mat
