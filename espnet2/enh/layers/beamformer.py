"""Beamformer module."""
from typing import List, Optional, Union

import torch
from packaging.version import parse as V
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import (cat, complex_norm, einsum,
                                              inverse, is_complex,
                                              is_torch_complex_tensor, matmul,
                                              reverse, solve, to_double)

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
    """Prepare necessary statistics for constructing the specified beamformer.

    Args:
        signal (torch.complex64/ComplexTensor): (..., F, C, T)
        masks_speech (List[torch.Tensor]): (..., F, C, T) masks for all speech sources
        mask_noise (torch.Tensor): (..., F, C, T) noise mask
        powers (List[torch.Tensor]): powers for all speech sources (..., F, T)
                                     used for wMPDR or WPD beamformers
        beamformer_type (str): one of the pre-defined beamformer types
        bdelay (int): delay factor, used for WPD beamformser
        btaps (int): number of filter taps, used for WPD beamformser
        eps (torch.Tensor): tiny constant
    Returns:
        beamformer_stats (dict): a dictionary containing all necessary statistics
            e.g. "psd_n", "psd_speech", "psd_distortion"
            Note:
            * When `masks_speech` is a tensor or a single-element list, all returned
              statistics are tensors;
            * When `masks_speech` is a multi-element list, some returned statistics
              can be a list, e.g., "psd_n" for MVDR, "psd_speech" and "psd_distortion".

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
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (torch.complex64/ComplexTensor): (..., F, C, T)
        reduction (str): "mean" or "median"
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (torch.complex64/ComplexTensor): (..., F, C, C)

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
    use_torch_solver: bool = True,
):
    """Calculate the relative transfer function (RTF)

    Algorithm of power method:
        1) rtf = reference_vector
        2) for i in range(iterations):
             rtf = (psd_noise^-1 @ psd_speech) @ rtf
             rtf = rtf / ||rtf||_2  # this normalization can be skipped
        3) rtf = psd_noise @ rtf
        4) rtf = rtf / rtf[..., ref_channel, :]
    Note: 4) Normalization at the reference channel is not performed here.

    Args:
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        mode (str): one of ("power", "evd")
            "power": power method
            "evd": eigenvalue decomposition
        reference_vector (torch.Tensor or int): (..., C) or scalar
        iterations (int): number of iterations in power method
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
    Returns:
        rtf (torch.complex64/ComplexTensor): (..., F, C, 1)
    """
    if mode == "power":
        if use_torch_solver:
            phi = solve(psd_speech, psd_noise)
        else:
            phi = matmul(inverse(psd_noise), psd_speech)
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
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """Return the MVDR (Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_n (torch.complex64/ComplexTensor):
            observation/noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    if use_torch_solver:
        numerator = solve(psd_s, psd_n)
    else:
        numerator = matmul(inverse(psd_n), psd_s)
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
    normalize_ref_channel: Optional[int] = None,
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """Return the MVDR (Minimum Variance Distortionless Response) vector
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
        normalize_ref_channel (int): reference channel for normalizing the RTF
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    # (B, F, C, 1)
    rtf = get_rtf(
        psd_speech,
        psd_noise,
        reference_vector=reference_vector,
        iterations=iterations,
        use_torch_solver=use_torch_solver,
    )

    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    if use_torch_solver:
        numerator = solve(rtf, psd_n).squeeze(-1)
    else:
        numerator = matmul(inverse(psd_n), rtf).squeeze(-1)
    denominator = einsum("...d,...d->...", rtf.squeeze(-1).conj(), numerator)
    if normalize_ref_channel is not None:
        scale = rtf.squeeze(-1)[..., normalize_ref_channel, None].conj()
        beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    else:
        beamforming_vector = numerator / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector


def apply_beamforming_vector(
    beamform_vector: Union[torch.Tensor, ComplexTensor],
    mix: Union[torch.Tensor, ComplexTensor],
) -> Union[torch.Tensor, ComplexTensor]:
    # (..., C) x (..., C, T) -> (..., T)
    es = einsum("...c,...ct->...t", beamform_vector.conj(), mix)
    return es


def get_mwf_vector(
    psd_s,
    psd_n,
    reference_vector: Union[torch.Tensor, int],
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """Return the MWF (Minimum Multi-channel Wiener Filter) vector:

        h = (Npsd^-1 @ Spsd) @ u

    Args:
        psd_s (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_n (torch.complex64/ComplexTensor):
            power-normalized observation covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    if use_torch_solver:
        ws = solve(psd_s, psd_n)
    else:
        ws = matmul(inverse(psd_n), psd_s)
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
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """Return the SDW-MWF (Speech Distortion Weighted Multi-channel Wiener Filter) vector

        h = (Spsd + mu * Npsd)^-1 @ Spsd @ u

    Reference:
        [1] Spatially pre-processed speech distortion weighted multi-channel Wiener
        filtering for noise reduction; A. Spriet et al, 2004
        https://dl.acm.org/doi/abs/10.1016/j.sigpro.2004.07.028
        [2] Rank-1 constrained multichannel Wiener filter for speech recognition in
        noisy environments; Z. Wang et al, 2018
        https://hal.inria.fr/hal-01634449/document
        [3] Low-rank approximation based multichannel Wiener filter algorithms for
        noise reduction with application in cochlear implants; R. Serizel, 2014
        https://ieeexplore.ieee.org/document/6730918

    Args:
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        denoising_weight (float): a trade-off parameter between noise reduction and
            speech distortion.
            A larger value leads to more noise reduction at the expense of more speech
            distortion.
            The plain MWF is obtained with `denoising_weight = 1` (by default).
        approx_low_rank_psd_speech (bool): whether to replace original input psd_speech
            with its low-rank approximation as in [2]
        iterations (int): number of iterations in power method, only used when
            `approx_low_rank_psd_speech = True`
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
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
            use_torch_solver=use_torch_solver,
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

    if use_torch_solver:
        ws = solve(psd_speech, psd_n)
    else:
        ws = matmul(inverse(psd_n), psd_speech)

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
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """Return the R1-MWF (Rank-1 Multi-channel Wiener Filter) vector

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
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        denoising_weight (float): a trade-off parameter between noise reduction and
            speech distortion.
            A larger value leads to more noise reduction at the expense of more speech
            distortion.
            When `denoising_weight = 0`, it corresponds to MVDR beamformer.
        approx_low_rank_psd_speech (bool): whether to replace original input psd_speech
            with its low-rank approximation as in [1]
        iterations (int): number of iterations in power method, only used when
            `approx_low_rank_psd_speech = True`
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
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
            use_torch_solver=use_torch_solver,
        )
        # Eq. (25) in Ref[1]
        psd_speech_r1 = matmul(recon_vec, recon_vec.conj().transpose(-1, -2))
        sigma_speech = FC.trace(psd_speech) / (FC.trace(psd_speech_r1) + eps)
        psd_speech_r1 = psd_speech_r1 * sigma_speech[..., None, None]
        # c.f. Eq. (62) in Ref[2]
        psd_speech = psd_speech_r1
    elif diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    if use_torch_solver:
        numerator = solve(psd_speech, psd_noise)
    else:
        numerator = matmul(inverse(psd_noise), psd_speech)

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
    use_torch_solver: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """Calculate the RTF matrix with each column the relative transfer function
    of the corresponding source.
    """  # noqa: H405
    assert isinstance(psd_speeches, list) and isinstance(psd_noises, list)
    rtf_mat = cat(
        [
            get_rtf(
                psd_speeches[spk],
                tik_reg(psd_n, reg=diag_eps, eps=eps) if diagonal_loading else psd_n,
                reference_vector=ref_channel,
                iterations=rtf_iterations,
                use_torch_solver=use_torch_solver,
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
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """Return the LCMV (Linearly Constrained Minimum Variance) vector
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
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, num_spk) -> (..., C_1, num_spk)
    if use_torch_solver:
        numerator = solve(rtf_mat, psd_n)
    else:
        numerator = matmul(inverse(psd_n), rtf_mat)
    denominator = matmul(rtf_mat.conj().transpose(-1, -2), numerator)
    if isinstance(reference_vector, int):
        ws = inverse(denominator)[..., reference_vector, None]
    else:
        ws = solve(reference_vector, denominator)
    beamforming_vector = matmul(numerator, ws).squeeze(-1)
    return beamforming_vector


def generalized_eigenvalue_decomposition(a: torch.Tensor, b: torch.Tensor, eps=1e-6):
    """Solves the generalized eigenvalue decomposition through Cholesky decomposition.

    ported from https://github.com/asteroid-team/asteroid/blob/master/asteroid/dsp/beamforming.py#L464

    a @ e_vec = e_val * b @ e_vec
    |
    |   Cholesky decomposition on `b`:
    |       b = L @ L^H, where `L` is a lower triangular matrix
    |
    |   Let C = L^-1 @ a @ L^-H, it is Hermitian.
    |
    => C @ y = lambda * y
    => e_vec = L^-H @ y

    Reference: https://www.netlib.org/lapack/lug/node54.html

    Args:
        a: A complex Hermitian or real symmetric matrix whose eigenvalues and
            eigenvectors will be computed. (..., C, C)
        b: A complex Hermitian or real symmetric definite positive matrix. (..., C, C)
    Returns:
        e_val: generalized eigenvalues (ascending order)
        e_vec: generalized eigenvectors
    """  # noqa: H405, E501
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
    """Phase correction to reduce distortions due to phase inconsistencies.

    ported from https://github.com/fgnt/nn-gev/blob/master/fgnt/beamforming.py#L169

    Args:
        vector: Beamforming vector with shape (..., F, C)
    Returns:
        w: Phase corrected beamforming vectors
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
    """Blind analytic normalization (BAN) for post-filtering

    Args:
        ws (torch.complex64/ComplexTensor): beamformer vector (..., F, C)
        psd_noise (torch.complex64/ComplexTensor): noise PSD matrix (..., F, C, C)
        eps (float)
    Returns:
        ws_ban (torch.complex64/ComplexTensor): normalized beamformer vector (..., F)
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
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """Return the generalized eigenvalue (GEV) beamformer vector:

        psd_speech @ h = lambda * psd_noise @ h

    Reference:
        Blind acoustic beamforming based on generalized eigenvalue decomposition;
        E. Warsitz and R. Haeb-Umbach, 2007.

    Args:
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        mode (str): one of ("power", "evd")
            "power": power method
            "evd": eigenvalue decomposition (only for torch builtin complex tensors)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        iterations (int): number of iterations in power method
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    if mode == "power":
        if use_torch_solver:
            phi = solve(psd_speech, psd_noise)
        else:
            phi = matmul(inverse(psd_noise), psd_speech)
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
    """Expand `signal` into several frames, with each frame of length `frame_length`.

    Args:
        signal : (..., T)
        frame_length:   length of each segment
        frame_step:     step for selecting frames
        bdelay:         delay for WPD
        do_padding:     whether or not to pad the input signal at the beginning
                          of the time dimension
        pad_value:      value to fill in the padding

    Returns:
        torch.Tensor:
            if do_padding: (..., T, frame_length)
            else:          (..., T - bdelay - frame_length + 2, frame_length)
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
    """Calculates the power normalized spatio-temporal covariance
        matrix of the framed signal.

    Args:
        Y : Complex STFT signal with shape (B, F, C, T)
        inverse_power : Weighting factor with shape (B, F, T)

    Returns:
        Correlation matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
        Correlation vector: (B, F, btaps + 1, C, C)
    """  # noqa: H405, D205, D400, D401
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
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Union[torch.Tensor, ComplexTensor]:
    """Return the WPD vector.

        WPD is the Weighted Power minimization Distortionless response
        convolutional beamformer. As follows:

        h = (Rf^-1 @ Phi_{xx}) / tr[(Rf^-1) @ Phi_{xx}] @ u

    Reference:
        T. Nakatani and K. Kinoshita, "A Unified Convolutional Beamformer
        for Simultaneous Denoising and Dereverberation," in IEEE Signal
        Processing Letters, vol. 26, no. 6, pp. 903-907, June 2019, doi:
        10.1109/LSP.2019.2911179.
        https://ieeexplore.ieee.org/document/8691481

    Args:
        Phi (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the PSD of zero-padded speech [x^T(t,f) 0 ... 0]^T.
        Rf (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, (btaps+1) * C)
            is the reference_vector.
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):

    Returns:
        filter_matrix (torch.complex64/ComplexTensor): (B, F, (btaps + 1) * C)
    """
    if diagonal_loading:
        Rf = tik_reg(Rf, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    if use_torch_solver:
        numerator = solve(Phi, Rf)
    else:
        numerator = matmul(inverse(Rf), Phi)
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
    """Return the WPD vector (v2).

       This implementation is more efficient than `get_WPD_filter` as
        it skips unnecessary computation with zeros.

    Args:
        Phi (torch.complex64/ComplexTensor): (B, F, C, C)
            is speech PSD.
        Rf (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, C)
            is the reference_vector.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):

    Returns:
        filter_matrix (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C)
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
    normalize_ref_channel: Optional[int] = None,
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-15,
) -> Union[torch.Tensor, ComplexTensor]:
    """Return the WPD vector calculated with RTF.

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
            stacked observation covariance matrix
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        iterations (int): number of iterations in power method
        reference_vector (torch.Tensor or int): (..., C) or scalar
        normalize_ref_channel (int):
            reference channel for normalizing the RTF
        use_torch_solver (bool):
            Whether to use `solve` instead of `inverse`
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor)r: (..., F, C)
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
        reference_vector=reference_vector,
        iterations=iterations,
        use_torch_solver=use_torch_solver,
    )

    # (B, F, (K+1)*C, 1)
    rtf = pad_func(rtf, (0, 0, 0, psd_observed_bar.shape[-1] - C), "constant", 0)
    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    if use_torch_solver:
        numerator = solve(rtf, psd_observed_bar).squeeze(-1)
    else:
        numerator = matmul(inverse(psd_observed_bar), rtf).squeeze(-1)
    denominator = einsum("...d,...d->...", rtf.squeeze(-1).conj(), numerator)
    if normalize_ref_channel is not None:
        scale = rtf.squeeze(-1)[..., normalize_ref_channel, None].conj()
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
    """Perform WPD filtering.

    Args:
        filter_matrix: Filter matrix (B, F, (btaps + 1) * C)
        Y : Complex STFT signal with shape (B, F, C, T)

    Returns:
        enhanced (torch.complex64/ComplexTensor): (B, F, T)
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
    """Perform Tikhonov regularization (only modifying real part).

    Args:
        mat (torch.complex64/ComplexTensor): input matrix (..., C, C)
        reg (float): regularization factor
        eps (float)
    Returns:
        ret (torch.complex64/ComplexTensor): regularized matrix (..., C, C)
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
