"""Beamformer module."""

from typing import List, Union

import torch
import torch_complex.functional as FC
import torchaudio


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

    This function computes the necessary statistics needed for various types
    of beamformers. It takes the input signal, speech masks, noise masks, and
    additional parameters to calculate power spectral densities (PSD) for
    speech and noise, which are essential for beamforming algorithms.

    Args:
        signal (torch.complex64):
            Input signal of shape (..., F, C, T) where F is the number of
            frequency bins, C is the number of channels, and T is the number
            of time frames.
        masks_speech (List[torch.Tensor]):
            A list of masks for all speech sources, each of shape (..., F, C, T).
        mask_noise (torch.Tensor):
            Noise mask of shape (..., F, C, T).
        powers (List[torch.Tensor], optional):
            Powers for all speech sources of shape (..., F, T) used for wMPDR
            or WPD beamformers. Defaults to None.
        beamformer_type (str, optional):
            Type of beamformer to be used. It can be one of the pre-defined
            types. Defaults to "mvdr".
        bdelay (int, optional):
            Delay factor used for WPD beamformer. Defaults to 3.
        btaps (int, optional):
            Number of filter taps used for WPD beamformer. Defaults to 5.
        eps (torch.Tensor, optional):
            A small constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        beamformer_stats (dict):
            A dictionary containing all necessary statistics such as "psd_n",
            "psd_speech", and "psd_distortion". The exact contents depend on
            the beamformer type.

    Note:
        * When `masks_speech` is a tensor or a single-element list, all
          returned statistics are tensors.
        * When `masks_speech` is a multi-element list, some returned
          statistics can be a list, e.g., "psd_n" for MVDR, "psd_speech"
          and "psd_distortion".

    Examples:
        >>> signal = torch.randn(2, 64, 4, 100, dtype=torch.complex64)  # (B, F, C, T)
        >>> masks_speech = [torch.rand(2, 64, 4, 100) for _ in range(2)]
        >>> mask_noise = torch.rand(2, 64, 4, 100)
        >>> stats = prepare_beamformer_stats(signal, masks_speech, mask_noise)
        >>> print(stats.keys())
        dict_keys(['psd_n', 'psd_speech', 'psd_distortion'])
    """
    from espnet2.enh.layers.dnn_beamformer import BEAMFORMER_TYPES

    assert beamformer_type in BEAMFORMER_TYPES, "%s is not supported yet"

    if isinstance(masks_speech, (list, tuple)):
        masks_speech = [m.cdouble() for m in masks_speech]
    else:
        masks_speech = [masks_speech.cdouble()]
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
            powers = [(power_input * m.abs()).mean(dim=-2) for m in masks_speech]
        else:
            assert len(powers) == num_spk, (len(powers), num_spk)
        inverse_powers = [1 / torch.clamp(p, min=eps) for p in powers]

    psd_transform = torchaudio.transforms.PSD(multi_mask=True, normalize=True)
    psd_speeches = [
        psd_transform(signal.transpose(-2, -3), m.transpose(-2, -3))
        for m in masks_speech
    ]
    if (
        beamformer_type == "mvdr_souden"
        or beamformer_type == "sdw_mwf"
        or beamformer_type == "r1mwf"
        or beamformer_type.startswith("mvdr_tfs")
        or not beamformer_type.endswith("_souden")
    ):
        # MVDR or other RTF-based formulas
        if mask_noise is not None:
            psd_bg = psd_transform(
                signal.transpose(-2, -3), mask_noise.cdouble().transpose(-2, -3)
            )
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
        psd_n = torch.einsum("...ct,...et->...ce", signal, signal.conj())
    elif beamformer_type in ("wmpdr", "wmpdr_souden", "wlcmp", "wmwf"):
        psd_n = [
            torch.einsum(
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


def get_rtf(
    psd_speech,
    psd_noise,
    mode="power",
    reference_vector: Union[int, torch.Tensor] = 0,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
):
    """
    Calculate the relative transfer function (RTF).

    This function computes the relative transfer function using either the power
    method or eigenvalue decomposition based on the provided mode. The RTF is used
    in various beamforming applications to enhance speech signals by estimating
    the relationship between the speech and noise covariance matrices.

    Args:
        psd_speech (torch.complex64):
            Speech covariance matrix with shape (..., F, C, C).
        psd_noise (torch.complex64):
            Noise covariance matrix with shape (..., F, C, C).
        mode (str): One of ("power", "evd").
            - "power": Uses the power method for RTF computation.
            - "evd": Uses eigenvalue decomposition for RTF computation.
        reference_vector (torch.Tensor or int):
            A tensor of shape (..., C) or a scalar representing the reference
            channel for the RTF computation.
        iterations (int): Number of iterations for the power method.
            This parameter is ignored if mode is "evd".
        diagonal_loading (bool): Whether to add a small constant to the diagonal
            of the noise covariance matrix to improve numerical stability.
        diag_eps (float): Small constant added for diagonal loading.

    Returns:
        rtf (torch.complex64): The computed relative transfer function with shape
        (..., F, C).

    Raises:
        ValueError: If an unknown mode is specified.

    Examples:
        >>> psd_speech = torch.randn(10, 256, 4, 4, dtype=torch.complex64)
        >>> psd_noise = torch.randn(10, 256, 4, 4, dtype=torch.complex64)
        >>> rtf = get_rtf(psd_speech, psd_noise, mode="power", iterations=3)
    """
    if mode == "power":
        rtf = torchaudio.functional.rtf_power(
            psd_speech,
            psd_noise,
            reference_vector,
            n_iter=iterations,
            diagonal_loading=diagonal_loading,
            diag_eps=diag_eps,
        )
    elif mode == "evd":
        rtf = torchaudio.functional.rtf_evd(psd_speech)
    else:
        raise ValueError("Unknown mode: %s" % mode)
    return rtf


def get_mvdr_vector(
    psd_s,
    psd_n,
    reference_vector: Union[torch.Tensor, int],
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
):
    """
    Return the MVDR (Minimum Variance Distortionless Response) vector.

    The MVDR vector is computed using the following formula:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (torch.complex64):
            Speech covariance matrix with shape (..., F, C, C).
        psd_n (torch.complex64):
            Observation/noise covariance matrix with shape (..., F, C, C).
        reference_vector (torch.Tensor or int):
            Reference vector of shape (..., C) or a scalar index.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n for stability.
        diag_eps (float):
            Regularization factor added to the diagonal of psd_n if
            diagonal_loading is True.
        eps (float):
            A small constant to prevent division by zero.

    Returns:
        beamform_vector (torch.complex64):
            The computed MVDR beamforming vector with shape (..., F, C).

    Examples:
        >>> psd_s = torch.randn(10, 4, 2, 2, dtype=torch.complex64)
        >>> psd_n = torch.randn(10, 4, 2, 2, dtype=torch.complex64)
        >>> reference_vector = torch.tensor([1, 0], dtype=torch.complex64)
        >>> mvdr_vector = get_mvdr_vector(psd_s, psd_n, reference_vector)

    Note:
        Ensure that the dimensions of psd_s and psd_n match and are properly
        formatted as complex tensors.
    """
    return torchaudio.functional.mvdr_weights_souden(
        psd_s,
        psd_n,
        reference_vector,
        diagonal_loading=diagonal_loading,
        diag_eps=diag_eps,
        eps=eps,
    )


def get_mvdr_vector_with_rtf(
    psd_n: torch.Tensor,
    psd_speech: torch.Tensor,
    psd_noise: torch.Tensor,
    iterations: int = 3,
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return the MVDR (Minimum Variance Distortionless Response) vector
    calculated with RTF:

        h = (Npsd^-1 @ rtf) / (rtf^H @ Npsd^-1 @ rtf)

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_n (torch.complex64):
            observation/noise covariance matrix (..., F, C, C)
        psd_speech (torch.complex64):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64):
            noise covariance matrix (..., F, C, C)
        iterations (int): number of iterations in power method
        reference_vector (torch.Tensor or int): (..., C) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float): Small constant for diagonal loading
        eps (float): Small constant to prevent division by zero

    Returns:
        beamform_vector (torch.complex64): (..., F, C)

    Examples:
        >>> psd_n = torch.rand(2, 4, 3, 3, dtype=torch.complex64)
        >>> psd_speech = torch.rand(2, 4, 3, 3, dtype=torch.complex64)
        >>> psd_noise = torch.rand(2, 4, 3, 3, dtype=torch.complex64)
        >>> vector = get_mvdr_vector_with_rtf(psd_n, psd_speech, psd_noise)
    """
    # (B, F, C)
    rtf = get_rtf(
        psd_speech,
        psd_noise,
        reference_vector=reference_vector,
        iterations=iterations,
        diagonal_loading=diagonal_loading,
        diag_eps=diag_eps,
    )
    return torchaudio.functional.mvdr_weights_rtf(
        rtf,
        psd_n,
        reference_vector,
        diagonal_loading=diagonal_loading,
        diag_eps=diag_eps,
        eps=eps,
    )


def apply_beamforming_vector(
    beamform_vector: torch.Tensor, mix: torch.Tensor
) -> torch.Tensor:
    """
    Apply the beamforming vector to the mixed signal.

    This function computes the output of a beamformer by applying the
    provided beamforming vector to the mixed signal. The operation is
    performed using the Einstein summation convention.

    Args:
        beamform_vector (torch.Tensor): The beamforming vector with shape
            (..., C), where C is the number of channels.
        mix (torch.Tensor): The mixed signal with shape (..., C, T),
            where T is the number of time frames.

    Returns:
        torch.Tensor: The output signal after applying the beamforming
            vector, with shape (..., T).

    Examples:
        >>> beamform_vector = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
        >>> mix = torch.tensor([[[0.5, 0.5], [0.1, 0.1]]], dtype=torch.complex64)
        >>> output = apply_beamforming_vector(beamform_vector, mix)
        >>> print(output.shape)
        torch.Size([1, 2])

    Note:
        The operation is performed as follows:
        output = sum(conjugate(beamform_vector) * mix, axis=-2)
    """
    # (..., C) x (..., C, T) -> (..., T)
    es = torch.einsum("...c,...ct->...t", beamform_vector.conj(), mix)
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
    Return the MWF (Minimum Multi-channel Wiener Filter) vector:

    h = (Npsd^-1 @ Spsd) @ u

    Args:
        psd_s (torch.complex64):
            speech covariance matrix (..., F, C, C)
        psd_n (torch.complex64):
            power-normalized observation covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float): Regularization term for diagonal loading
        eps (float): Small constant to avoid division by zero

    Returns:
        beamform_vector (torch.complex64): (..., F, C)

    Examples:
        >>> psd_s = torch.rand(10, 8, 2, 2, dtype=torch.complex64)
        >>> psd_n = torch.rand(10, 8, 2, 2, dtype=torch.complex64)
        >>> ref_vec = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        >>> result = get_mwf_vector(psd_s, psd_n, ref_vec)
        >>> print(result.shape)
        torch.Size([10, 8, 2])
    """
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    ws = torch.linalg.solve(psd_n, psd_s)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = torch.einsum(
            "...fec,...c->...fe", ws, reference_vector.to(dtype=ws.dtype)
        )
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

    The SDW-MWF is defined as:
        h = (Spsd + mu * Npsd)^-1 @ Spsd @ u

    References:
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
        psd_speech (torch.complex64):
            Speech covariance matrix (..., F, C, C).
        psd_noise (torch.complex64):
            Noise covariance matrix (..., F, C, C).
        reference_vector (torch.Tensor or int):
            Reference vector of shape (..., C) or scalar.
        denoising_weight (float):
            A trade-off parameter between noise reduction and speech distortion.
            A larger value leads to more noise reduction at the expense of more
            speech distortion. The plain MWF is obtained with `denoising_weight = 1`.
        approx_low_rank_psd_speech (bool):
            Whether to replace original input psd_speech with its low-rank
            approximation as in [2].
        iterations (int):
            Number of iterations in power method, only used when
            `approx_low_rank_psd_speech = True`.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n.
        diag_eps (float):
            Regularization parameter for diagonal loading.
        eps (float):
            Small constant to prevent division by zero.

    Returns:
        beamform_vector (torch.complex64):
            The computed beamformer vector of shape (..., F, C).
    """
    if approx_low_rank_psd_speech:
        if diagonal_loading:
            psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

        # (B, F, C)
        recon_vec = get_rtf(
            psd_speech,
            psd_noise,
            mode="power",
            iterations=iterations,
            reference_vector=reference_vector,
            diagonal_loading=False,
        )
        # Eq. (25) in Ref[2]
        psd_speech_r1 = torch.einsum("...c,...e->...ce", recon_vec, recon_vec.conj())
        sigma_speech = FC.trace(psd_speech) / (FC.trace(psd_speech_r1) + eps)
        psd_speech_r1 = psd_speech_r1 * sigma_speech[..., None, None]
        # c.f. Eq. (62) in Ref[3]
        psd_speech = psd_speech_r1

    psd_n = psd_speech + denoising_weight * psd_noise
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    ws = torch.linalg.solve(psd_n, psd_speech)

    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = torch.einsum(
            "...fec,...c->...fe", ws, reference_vector.to(dtype=ws.dtype)
        )
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

    The Rank-1 Multi-channel Wiener Filter is computed as follows:

        h = (Npsd^-1 @ Spsd) / (mu + Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        [1] Rank-1 constrained multichannel Wiener filter for speech recognition in
        noisy environments; Z. Wang et al, 2018
        https://hal.inria.fr/hal-01634449/document
        [2] Low-rank approximation based multichannel Wiener filter algorithms for
        noise reduction with application in cochlear implants; R. Serizel, 2014
        https://ieeexplore.ieee.org/document/6730918

    Args:
        psd_speech (torch.complex64):
            Speech covariance matrix (..., F, C, C).
        psd_noise (torch.complex64):
            Noise covariance matrix (..., F, C, C).
        reference_vector (torch.Tensor or int):
            Reference vector (..., C) or scalar.
        denoising_weight (float):
            A trade-off parameter between noise reduction and speech distortion.
            A larger value leads to more noise reduction at the expense of more
            speech distortion. When `denoising_weight = 0`, it corresponds to
            the MVDR beamformer.
        approx_low_rank_psd_speech (bool):
            Whether to replace original input psd_speech with its low-rank
            approximation as in [1].
        iterations (int):
            Number of iterations in power method, only used when
            `approx_low_rank_psd_speech = True`.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n.
        diag_eps (float):
            Regularization parameter for diagonal loading.
        eps (float):
            Small constant to prevent division by zero.

    Returns:
        beamform_vector (torch.complex64):
            The computed beamforming vector (..., F, C).
    """
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)
    if approx_low_rank_psd_speech:
        # (B, F, C)
        recon_vec = get_rtf(
            psd_speech,
            psd_noise,
            mode="power",
            iterations=iterations,
            reference_vector=reference_vector,
            diagonal_loading=False,
        )
        # Eq. (25) in Ref[1]
        psd_speech_r1 = torch.einsum("...c,...e->...ce", recon_vec, recon_vec.conj())
        sigma_speech = FC.trace(psd_speech) / (FC.trace(psd_speech_r1) + eps)
        psd_speech_r1 = psd_speech_r1 * sigma_speech[..., None, None]
        # c.f. Eq. (62) in Ref[2]
        psd_speech = psd_speech_r1

    numerator = torch.linalg.solve(psd_noise, psd_speech)

    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (denoising_weight + FC.trace(numerator)[..., None, None] + eps)

    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = torch.einsum(
            "...fec,...c->...fe", ws, reference_vector.to(dtype=ws.dtype)
        )
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

    This function computes the relative transfer function (RTF) for each
    speech source given their respective power spectral density (PSD)
    matrices. The resulting RTF matrix normalizes the transfer functions
    with respect to a specified reference channel.

    Args:
        psd_speeches (list): A list of speech covariance matrices, where
            each matrix has shape (..., F, C, C).
        psd_noises (list): A list of noise covariance matrices, where each
            matrix has shape (..., F, C, C).
        diagonal_loading (bool): Whether to add a small constant to the
            diagonal of the noise covariance matrices to improve numerical
            stability. Defaults to True.
        ref_channel (int): The index of the reference channel used for
            normalization. Defaults to 0.
        rtf_iterations (int): Number of iterations for computing the RTF
            using the power method. Defaults to 3.
        diag_eps (float): A small value added for diagonal loading to avoid
            singular matrices. Defaults to 1e-7.
        eps (float): A small constant to avoid division by zero in
            calculations. Defaults to 1e-8.

    Returns:
        torch.Tensor: The RTF matrix with shape (..., F, C) where each
            column corresponds to the relative transfer function of a
            different speech source.

    Examples:
        >>> psd_speeches = [torch.randn(1, 8, 2, 2) for _ in range(3)]
        >>> psd_noises = [torch.randn(1, 8, 2, 2) for _ in range(3)]
        >>> rtf_matrix = get_rtf_matrix(psd_speeches, psd_noises)
        >>> rtf_matrix.shape
        torch.Size([1, 8, 2, 3])  # (Batch, Frequency, Channels, Sources)

    Raises:
        AssertionError: If either `psd_speeches` or `psd_noises` is not a
        list or if their lengths do not match.
    """
    assert isinstance(psd_speeches, list) and isinstance(psd_noises, list)
    rtf_mat = torch.stack(
        [
            get_rtf(
                psd_speeches[spk],
                psd_n,
                reference_vector=ref_channel,
                iterations=rtf_iterations,
                diagonal_loading=diagonal_loading,
                diag_eps=diag_eps,
            )
            for spk, psd_n in enumerate(psd_noises)
        ],
        dim=-1,
    )
    # normalize at the reference channel
    return rtf_mat / rtf_mat[..., ref_channel, None, :]


def get_lcmv_vector_with_rtf(
    psd_n: torch.Tensor,
    rtf_mat: torch.Tensor,
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return the LCMV (Linearly Constrained Minimum Variance) vector
        calculated with RTF:

        h = (Npsd^-1 @ rtf_mat) @ (rtf_mat^H @ Npsd^-1 @ rtf_mat)^-1 @ p

    Reference:
        H. L. Van Trees, “Optimum array processing: Part IV of detection, estimation,
        and modulation theory,” John Wiley & Sons, 2004. (Chapter 6.7)

    Args:
        psd_n (torch.complex64):
            observation/noise covariance matrix (..., F, C, C)
        rtf_mat (torch.complex64):
            RTF matrix (..., F, C, num_spk)
        reference_vector (torch.Tensor or int): (..., num_spk) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float): Regularization term to be added to the diagonal
        eps (float): Small constant to avoid division by zero

    Returns:
        beamform_vector (torch.complex64): (..., F, C)

    Examples:
        >>> psd_n = torch.randn(2, 5, 3, 3, dtype=torch.complex64)
        >>> rtf_mat = torch.randn(2, 5, 3, 4, dtype=torch.complex64)
        >>> reference_vector = torch.tensor([1.0, 0.0, 0.0])
        >>> beamform_vector = get_lcmv_vector_with_rtf(
        ...     psd_n, rtf_mat, reference_vector=reference_vector)
    """
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, num_spk) -> (..., C_1, num_spk)
    numerator = torch.linalg.solve(psd_n, rtf_mat)
    denominator = torch.matmul(rtf_mat.conj().transpose(-1, -2), numerator)
    if isinstance(reference_vector, int):
        ws = denominator.inverse()[..., reference_vector, None]
    else:
        ws = torch.linalg.solve(denominator, reference_vector)
    beamforming_vector = torch.matmul(numerator, ws).squeeze(-1)
    return beamforming_vector


def generalized_eigenvalue_decomposition(a: torch.Tensor, b: torch.Tensor, eps=1e-6):
    """
    Solves the generalized eigenvalue decomposition through Cholesky decomposition.

    The function computes the generalized eigenvalue decomposition defined by:

        a @ e_vec = e_val * b @ e_vec

    It performs Cholesky decomposition on matrix `b`:

        b = L @ L^H, where `L` is a lower triangular matrix.

    Then, it constructs the matrix C:

        C = L^-1 @ a @ L^-H

    Finally, the eigenvalue problem is solved for C:

        C @ y = lambda * y
        e_vec = L^-H @ y

    References:
        - [1] https://www.netlib.org/lapack/lug/node54.html
        - [2] Ported from https://github.com/asteroid-team/asteroid/blob/master/asteroid/dsp/beamforming.py#L464

    Args:
        a: A complex Hermitian or real symmetric matrix whose eigenvalues and
           eigenvectors will be computed. Shape: (..., C, C).
        b: A complex Hermitian or real symmetric definite positive matrix.
           Shape: (..., C, C).
        eps: A small constant added for numerical stability (default: 1e-6).

    Returns:
        e_val: Generalized eigenvalues in ascending order.
        e_vec: Generalized eigenvectors corresponding to the eigenvalues.

    Examples:
        >>> a = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.complex64)
        >>> b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
        >>> eigenvalues, eigenvectors = generalized_eigenvalue_decomposition(a, b)
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

    This function applies a phase correction to a beamforming vector to reduce
    distortions caused by inconsistencies in the phase across different frequency
    bins. The correction is achieved by computing the phase of the product of the
    current and the previous frequency bin's beamforming vector.

    This function is particularly useful in beamforming applications where
    maintaining consistent phase information is critical for performance.

    Args:
        vector (torch.Tensor): Beamforming vector with shape (..., F, C), where
            F is the number of frequency bins and C is the number of channels.

    Returns:
        torch.Tensor: Phase corrected beamforming vectors with the same shape
        as the input vector.

    Examples:
        >>> vector = torch.randn(2, 5, 3, dtype=torch.complex64)  # (B, F, C)
        >>> corrected_vector = gev_phase_correction(vector)
        >>> print(corrected_vector.shape)
        torch.Size([2, 5, 3])
    """
    B, F, C = vector.shape
    correction = torch.empty_like(vector.real)
    for f in range(F):
        correction[:, f, :] = torch.exp(
            (vector[:, f, :] * vector[:, f - 1, :].conj())
            .sum(dim=-1, keepdim=True)
            .angle()
        )
    correction = torch.exp(-1j * correction)
    return vector * correction


def blind_analytic_normalization(ws, psd_noise, eps=1e-8):
    """
    Blind analytic normalization (BAN) for post-filtering.

    This function normalizes the beamformer vector using the Blind Analytic
    Normalization technique, which helps in post-filtering by reducing noise
    and enhancing the signal quality.

    Args:
        ws (torch.complex64):
            Beamformer vector of shape (..., F, C).
        psd_noise (torch.complex64):
            Noise power spectral density (PSD) matrix of shape (..., F, C, C).
        eps (float):
            A small constant to avoid division by zero. Default is 1e-8.

    Returns:
        ws_ban (torch.complex64):
            Normalized beamformer vector of shape (..., F).

    Examples:
        >>> ws = torch.randn(2, 3, 4, dtype=torch.complex64)  # Shape: (B, F, C)
        >>> psd_noise = torch.randn(2, 3, 4, 4, dtype=torch.complex64)  # Shape: (B, F, C, C)
        >>> normalized_ws = blind_analytic_normalization(ws, psd_noise)
        >>> print(normalized_ws.shape)  # Output: (B, F)
    """
    C2 = psd_noise.size(-1) ** 2
    denominator = torch.einsum("...c,...ce,...e->...", ws.conj(), psd_noise, ws)
    numerator = torch.einsum(
        "...c,...ce,...eo,...o->...", ws.conj(), psd_noise, psd_noise, ws
    )
    gain = (numerator + eps).sqrt() / (denominator * C2 + eps)
    return gain


def get_gev_vector(
    psd_noise: torch.Tensor,
    psd_speech: torch.Tensor,
    mode="power",
    reference_vector: Union[int, torch.Tensor] = 0,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return the generalized eigenvalue (GEV) beamformer vector.

    The GEV beamformer vector is defined by the equation:
        psd_speech @ h = lambda * psd_noise @ h

    Reference:
        Blind acoustic beamforming based on generalized eigenvalue decomposition;
        E. Warsitz and R. Haeb-Umbach, 2007.

    Args:
        psd_noise (torch.complex64):
            Noise covariance matrix of shape (..., F, C, C).
        psd_speech (torch.complex64):
            Speech covariance matrix of shape (..., F, C, C).
        mode (str):
            Method to compute the eigenvalue decomposition, either "power"
            or "evd". Default is "power".
        reference_vector (torch.Tensor or int):
            A reference vector of shape (..., C) or a scalar index.
        iterations (int):
            Number of iterations for the power method. Default is 3.
        diagonal_loading (bool):
            If True, adds a tiny term to the diagonal of psd_noise.
        diag_eps (float):
            Regularization parameter for diagonal loading.
        eps (float):
            Small constant to avoid division by zero.

    Returns:
        beamform_vector (torch.complex64):
            The resulting GEV beamformer vector of shape (..., F, C).

    Raises:
        ValueError: If an unknown mode is provided.

    Examples:
        >>> psd_n = torch.rand(4, 8, 2, 2, dtype=torch.complex64)
        >>> psd_s = torch.rand(4, 8, 2, 2, dtype=torch.complex64)
        >>> beamformer_vector = get_gev_vector(psd_n, psd_s)
        >>> print(beamformer_vector.shape)
        torch.Size([4, 8, 2])
    """
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    if mode == "power":
        phi = torch.linalg.solve(psd_noise, psd_speech)
        e_vec = (
            phi[..., reference_vector, None]
            if isinstance(reference_vector, int)
            else torch.matmul(phi, reference_vector[..., None, :, None])
        )
        for _ in range(iterations - 1):
            e_vec = torch.matmul(phi, e_vec)
            # e_vec = e_vec / complex_norm(e_vec, dim=-1, keepdim=True)
        e_vec = e_vec.squeeze(-1)
    elif mode == "evd":
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

    beamforming_vector = e_vec / torch.norm(e_vec, dim=-1, keepdim=True)
    beamforming_vector = gev_phase_correction(beamforming_vector)
    return beamforming_vector


def signal_framing(
    signal: torch.Tensor,
    frame_length: int,
    frame_step: int,
    bdelay: int,
    do_padding: bool = False,
    pad_value: int = 0,
    indices: List = None,
) -> torch.Tensor:
    """
    Expand `signal` into several frames, with each frame of length
    `frame_length`.

    This function takes a 1D or multi-dimensional signal and divides it into
    overlapping frames. The frames can be created with optional padding, and
    the resulting tensor can be indexed based on provided indices.

    Args:
        signal (torch.Tensor): The input signal of shape (..., T).
        frame_length (int): Length of each frame.
        frame_step (int): Step size for selecting frames.
        bdelay (int): Delay for WPD (Weighted Power Distortionless Response).
        do_padding (bool): Whether to pad the input signal at the beginning
                           of the time dimension.
        pad_value (int): Value to fill in the padding.
        indices (List, optional): Specific indices to select frames. If None,
                                   indices will be automatically calculated.

    Returns:
        torch.Tensor:
            If `do_padding` is True, returns a tensor of shape (..., T,
            frame_length); otherwise, returns a tensor of shape (...,
            T - bdelay - frame_length + 2, frame_length).

    Examples:
        >>> signal = torch.tensor([1, 2, 3, 4, 5])
        >>> framed_signal = signal_framing(signal, frame_length=3,
        ... frame_step=1, bdelay=1, do_padding=True)
        >>> print(framed_signal)
        tensor([[0, 0, 1],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]])

    Note:
        The input signal can be complex. In such cases, the function
        will separately handle the real and imaginary parts before
        reconstructing the complex output.
    """
    frame_length2 = frame_length - 1
    # pad to the right at the last dimension of `signal` (time dimension)
    if do_padding:
        # (..., T) --> (..., T + bdelay + frame_length - 2)
        signal = torch.nn.functional.pad(
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

    if torch.is_complex(signal):
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
        return torch.complex(real, imag)
    else:
        # (..., T - bdelay - frame_length + 2, frame_length)
        signal = signal[..., indices]
        return signal


def get_covariances(
    Y: torch.Tensor,
    inverse_power: torch.Tensor,
    bdelay: int,
    btaps: int,
    get_vector: bool = False,
) -> torch.Tensor:
    """
    Calculates the power normalized spatio-temporal covariance matrix
    of the framed signal.

    This function computes the covariance matrix for a given complex
    short-time Fourier transform (STFT) signal. The covariance is
    normalized by a weighting factor that is provided as an inverse
    power input. It can return either the full correlation matrix or
    a correlation vector, depending on the `get_vector` flag.

    Args:
        Y (torch.Tensor):
            Complex STFT signal with shape (B, F, C, T), where B is the
            batch size, F is the number of frequency bins, C is the
            number of channels, and T is the number of time frames.
        inverse_power (torch.Tensor):
            Weighting factor with shape (B, F, T) used for normalizing
            the covariance matrix.
        bdelay (int):
            Delay for WPD (Weighted Power Distortionless response).
        btaps (int):
            Number of filter taps used in the computation.
        get_vector (bool, optional):
            If True, returns the correlation vector in addition to the
            covariance matrix. Defaults to False.

    Returns:
        torch.Tensor:
            If `get_vector` is False, returns the correlation matrix with
            shape (B, F, (btaps + 1) * C, (btaps + 1) * C). If
            `get_vector` is True, returns a tuple containing the
            correlation matrix and the correlation vector with shape
            (B, F, btaps + 1, C, C).

    Raises:
        AssertionError: If the dimensions of `inverse_power` do not match
        those of `Y`.

    Examples:
        >>> Y = torch.randn(2, 4, 3, 5, dtype=torch.complex64)
        >>> inverse_power = torch.randn(2, 4, 5)
        >>> cov_matrix = get_covariances(Y, inverse_power, bdelay=1, btaps=2)
        >>> cov_matrix.shape
        torch.Size([2, 4, 12, 12])  # (B, F, (btaps + 1) * C, (btaps + 1) * C)

        >>> cov_matrix, cov_vector = get_covariances(Y, inverse_power,
        ...     bdelay=1, btaps=2, get_vector=True)
        >>> cov_vector.shape
        torch.Size([2, 4, 3, 3])  # (B, F, btaps + 1, C, C)
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
    Psi = torch.flip(Psi, dims=(-1,))
    Psi_norm = Psi * inverse_power[..., None, bdelay + btaps - 1 :, None]

    # let T' = T - bdelay - btaps + 1
    # (B, F, C, T', btaps + 1) x (B, F, C, T', btaps + 1)
    #  -> (B, F, btaps + 1, C, btaps + 1, C)
    covariance_matrix = torch.einsum("bfdtk,bfetl->bfkdle", Psi, Psi_norm.conj())

    # (B, F, btaps + 1, C, btaps + 1, C)
    #   -> (B, F, (btaps + 1) * C, (btaps + 1) * C)
    covariance_matrix = covariance_matrix.view(
        Bs, Fdim, (btaps + 1) * C, (btaps + 1) * C
    )

    if get_vector:
        # (B, F, C, T', btaps + 1) x (B, F, C, T')
        #    --> (B, F, btaps +1, C, C)
        covariance_vector = torch.einsum(
            "bfdtk,bfet->bfked", Psi_norm, Y[..., bdelay + btaps - 1 :].conj()
        )
        return covariance_matrix, covariance_vector
    else:
        return covariance_matrix


def get_WPD_filter(
    Phi: torch.Tensor,
    Rf: torch.Tensor,
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return the WPD vector.

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
        Phi (torch.complex64): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the PSD of zero-padded speech [x^T(t,f) 0 ... 0]^T.
        Rf (torch.complex64): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, (btaps+1) * C)
            is the reference vector.
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n.
        diag_eps (float): Regularization parameter for diagonal loading.
        eps (float): Small constant to avoid division by zero.

    Returns:
        filter_matrix (torch.complex64): (B, F, (btaps + 1) * C)
            The computed WPD filter matrix.
    """
    if diagonal_loading:
        Rf = tik_reg(Rf, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = torch.linalg.solve(Rf, Phi)
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = torch.einsum(
        "...fec,...c->...fe", ws, reference_vector.to(dtype=ws.dtype)
    )
    # (B, F, (btaps + 1) * C)
    return beamform_vector


def get_WPD_filter_v2(
    Phi: torch.Tensor,
    Rf: torch.Tensor,
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return the WPD vector (v2).

       This implementation is more efficient than `get_WPD_filter` as
       it skips unnecessary computation with zeros.

    Args:
        Phi (torch.complex64): (B, F, C, C)
            is speech PSD.
        Rf (torch.complex64): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, C)
            is the reference_vector.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
            Regularization term for diagonal loading.
        eps (float):
            Small constant to prevent division by zero.

    Returns:
        filter_matrix (torch.complex64): (B, F, (btaps+1) * C)

    Examples:
        >>> Phi = torch.randn(2, 5, 4, 4, dtype=torch.complex64)
        >>> Rf = torch.randn(2, 5, 6, 6, dtype=torch.complex64)
        >>> reference_vector = torch.randn(2, 4, dtype=torch.complex64)
        >>> filter_matrix = get_WPD_filter_v2(Phi, Rf, reference_vector)

    Note:
        This function assumes that the input tensors have the correct
        shapes as specified in the arguments.
    """
    C = reference_vector.shape[-1]
    if diagonal_loading:
        Rf = tik_reg(Rf, reg=diag_eps, eps=eps)
    inv_Rf = Rf.inverse()
    # (B, F, (btaps+1) * C, C)
    inv_Rf_pruned = inv_Rf[..., :C]
    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = torch.matmul(inv_Rf_pruned, Phi)
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., (btaps+1) * C, C) / (...,) -> (..., (btaps+1) * C, C)
    ws = numerator / (FC.trace(numerator[..., :C, :])[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = torch.einsum(
        "...fec,...c->...fe", ws, reference_vector.to(dtype=ws.dtype)
    )
    # (B, F, (btaps+1) * C)
    return beamform_vector


def get_WPD_filter_with_rtf(
    psd_observed_bar: torch.Tensor,
    psd_speech: torch.Tensor,
    psd_noise: torch.Tensor,
    iterations: int = 3,
    reference_vector: Union[int, torch.Tensor] = 0,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-15,
) -> torch.Tensor:
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
        psd_observed_bar (torch.complex64):
            Stacked observation covariance matrix.
        psd_speech (torch.complex64):
            Speech covariance matrix (..., F, C, C).
        psd_noise (torch.complex64):
            Noise covariance matrix (..., F, C, C).
        iterations (int):
            Number of iterations in the power method.
        reference_vector (torch.Tensor or int):
            Reference vector shape (..., C) or scalar.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n.
        diag_eps (float):
            Regularization factor for diagonal loading.
        eps (float):
            Small constant to prevent division by zero.

    Returns:
        beamform_vector (torch.complex64):
            Beamforming vector shape (..., F, C).

    Examples:
        >>> psd_observed_bar = torch.randn(2, 256, 10, 10, dtype=torch.complex64)
        >>> psd_speech = torch.randn(2, 256, 10, 10, dtype=torch.complex64)
        >>> psd_noise = torch.randn(2, 256, 10, 10, dtype=torch.complex64)
        >>> beamform_vector = get_WPD_filter_with_rtf(psd_observed_bar,
        ...                                             psd_speech,
        ...                                             psd_noise)
        >>> beamform_vector.shape
        torch.Size([2, 256, 10])
    """
    C = psd_noise.size(-1)

    # (B, F, C)
    rtf = get_rtf(
        psd_speech,
        psd_noise,
        mode="power",
        reference_vector=reference_vector,
        iterations=iterations,
        diagonal_loading=diagonal_loading,
        diag_eps=diag_eps,
    )

    # (B, F, (K+1)*C)
    rtf = torch.nn.functional.pad(
        rtf, (0, psd_observed_bar.shape[-1] - C), "constant", 0
    )
    # numerator: (..., C_1, C_2) x (..., C_2) -> (..., C_1)
    numerator = torch.linalg.solve(psd_observed_bar, rtf)
    denominator = torch.einsum("...d,...d->...", rtf.conj(), numerator)

    if isinstance(reference_vector, int):
        scale = rtf[..., reference_vector, None].conj()
    else:
        scale = torch.einsum(
            "...c,...c->...",
            [rtf[:, :, :C].conj(), reference_vector[..., None, :].to(dtype=rtf.dtype)],
        ).unsqueeze(-1)
    beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector


def perform_WPD_filtering(
    filter_matrix: torch.Tensor,
    Y: torch.Tensor,
    bdelay: int,
    btaps: int,
) -> torch.Tensor:
    """
    Perform WPD filtering.

    This function applies the Weighted Power minimization Distortionless
    response (WPD) filtering to a complex Short-Time Fourier Transform (STFT)
    signal. The WPD filter is designed to enhance the signal by reducing noise
    while preserving the desired speech signal.

    Args:
        filter_matrix (torch.Tensor):
            Filter matrix with shape (B, F, (btaps + 1) * C), where B is the
            batch size, F is the number of frequency bins, and C is the number
            of channels.
        Y (torch.Tensor):
            Complex STFT signal with shape (B, F, C, T), where T is the number
            of time frames.

    Returns:
        enhanced (torch.complex64):
            Enhanced signal with shape (B, F, T).

    Examples:
        >>> filter_matrix = torch.rand(2, 256, 10)  # Example filter matrix
        >>> Y = torch.rand(2, 256, 10, 100, dtype=torch.complex64)  # STFT signal
        >>> bdelay = 3  # Example delay
        >>> btaps = 5   # Example number of taps
        >>> enhanced_signal = perform_WPD_filtering(filter_matrix, Y, bdelay, btaps)
        >>> print(enhanced_signal.shape)
        torch.Size([2, 256, 100])  # Output shape
    """
    # (B, F, C, T) --> (B, F, C, T, btaps + 1)
    Ytilde = signal_framing(Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Ytilde = torch.flip(Ytilde, dims=(-1,))

    Bs, Fdim, C, T = Y.shape
    # --> (B, F, T, btaps + 1, C) --> (B, F, T, (btaps + 1) * C)
    Ytilde = Ytilde.permute(0, 1, 3, 4, 2).contiguous().view(Bs, Fdim, T, -1)
    # (B, F, T, 1)
    enhanced = torch.einsum("...tc,...c->...t", Ytilde, filter_matrix.conj())
    return enhanced


def tik_reg(mat, reg: float = 1e-8, eps: float = 1e-8):
    """
        Perform Tikhonov regularization (only modifying real part).

    This function adds a regularization term to the input matrix to stabilize
    inverse calculations, particularly in situations where the matrix may be
    ill-conditioned. It works specifically for complex matrices, modifying
    only the real part while keeping the imaginary part unchanged.

    Args:
        mat (torch.complex64):
            Input matrix of shape (..., C, C) to be regularized.
        reg (float):
            Regularization factor that determines the strength of the
            regularization applied.
        eps (float):
            Small constant added to avoid division by zero.

    Returns:
        ret (torch.complex64):
            Regularized matrix of shape (..., C, C), where the real part
            has been modified by the regularization term.

    Examples:
        >>> import torch
        >>> mat = torch.randn(2, 3, 3, dtype=torch.complex64)  # Example input
        >>> reg_mat = tik_reg(mat, reg=0.1, eps=1e-8)  # Applying Tikhonov regularization
        >>> print(reg_mat.shape)  # Output shape will be (2, 3, 3)
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
