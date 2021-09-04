"""Beamformer module."""
from distutils.version import LooseVersion
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import einsum
from espnet2.enh.layers.complex_utils import inverse
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.complex_utils import is_torch_complex_tensor
from espnet2.enh.layers.complex_utils import matmul
from espnet2.enh.layers.complex_utils import reverse
from espnet2.enh.layers.complex_utils import solve


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")
EPS = torch.finfo(torch.double).eps


def get_power_spectral_density_matrix(
    xs, mask: torch.Tensor, normalization=True, eps: float = 1e-15
):
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (torch.complex64/ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (torch.complex64/ComplexTensor): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = einsum("...ct,...et->...tce", xs, xs.conj())

    # Averaging mask along C: (..., C, T) -> (..., T)
    mask = mask.mean(dim=-2)

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    # psd: (..., T, C, C)
    psd = psd_Y * mask[..., None, None]
    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(dim=-3)

    return psd


def get_rtf(
    psd_speech,
    psd_noise,
    reference_vector: Union[int, torch.Tensor, None] = None,
    iterations: int = 3,
    use_torch_solver: bool = True,
):
    """Calculate the relative transfer function (RTF) using the power method.

    Algorithm:
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
        reference_vector (torch.Tensor or int): (..., C) or scalar
        iterations (int): number of iterations in power method
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
    Returns:
        rtf (torch.complex64/ComplexTensor): (..., F, C, 1)
    """
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
        # rtf = rtf / complex_norm(rtf)
    rtf = matmul(psd_speech, rtf)
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
        reference_vector,
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
        reference_vector,
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


##############################################
# Below are for Multi-Frame MVDR beamforming #
##############################################
# modified from https://gitlab.uni-oldenburg.de/hura4843/deep-mfmvdr/-/blob/master/deep_mfmvdr (# noqa: E501)
def get_adjacent(spec, filter_length: int = 5):
    """Zero-pad and unfold stft, i.e.,

    add zeros to the beginning so that, using the multi-frame signal model,
    there will be as many output frames as input frames.

    Args:
        spec (torch.complex64/ComplexTensor): input spectrum (B, F, T)
        filter_length (int): length for frame extension
    Returns:
        ret (torch.complex64/ComplexTensor): output spectrum (B, F, T, filter_length)
    """  # noqa: D400
    if isinstance(spec, ComplexTensor):
        pad_func = FC.pad
    elif is_torch_complex_tensor(spec):
        pad_func = torch.nn.functional.pad
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )
    return (
        pad_func(spec, pad=[filter_length - 1, 0])
        .unfold(dim=-1, size=filter_length, step=1)
        .contiguous()
    )


def get_adjacent_th(spec: torch.Tensor, filter_length: int = 5) -> torch.Tensor:
    """Zero-pad and unfold stft, i.e.,

    add zeros to the beginning so that, using the multi-frame signal model,
    there will be as many output frames as input frames.

    Args:
        spec (torch.Tensor): input spectrum (B, F, T, 2)
        filter_length (int): length for frame extension
    Returns:
        ret (torch.Tensor): output spectrum (B, F, T, filter_length, 2)
    """  # noqa: D400
    return (
        torch.nn.functional.pad(spec, pad=[0, 0, filter_length - 1, 0])
        .unfold(dimension=-2, size=filter_length, step=1)
        .transpose(-2, -1)
        .contiguous()
    )


def vector_to_Hermitian(vec, use_builtin_complex=False):
    """Construct a Hermitian matrix from a vector of N**2 independent
    real-valued elements.

    Args:
        vec (torch.Tensor): (..., N ** 2)
        use_builtin_complex (bool): Whether to use builtin complex support
    Returns:
        mat (torch.complex64/ComplexTensor): (..., N, N)
    """  # noqa: H405, D205, D400
    N = int(np.sqrt(vec.shape[-1]))
    mat = torch.zeros(size=vec.shape[:-1] + (N, N, 2), device=vec.device)

    # real component
    triu = np.triu_indices(N, 0)
    triu2 = np.triu_indices(N, 1)  # above main diagonal
    tril = (triu2[1], triu2[0])  # below main diagonal; for symmetry
    mat[(...,) + triu + (np.zeros(triu[0].shape[0]),)] = vec[..., : triu[0].shape[0]]
    start = triu[0].shape[0]
    mat[(...,) + tril + (np.zeros(tril[0].shape[0]),)] = mat[
        (...,) + triu2 + (np.zeros(triu2[0].shape[0]),)
    ]

    # imaginary component
    mat[(...,) + triu2 + (np.ones(triu2[0].shape[0]),)] = vec[
        ..., start : start + triu2[0].shape[0]
    ]
    mat[(...,) + tril + (np.ones(tril[0].shape[0]),)] = -mat[
        (...,) + triu2 + (np.ones(triu2[0].shape[0]),)
    ]

    if is_torch_1_9_plus and use_builtin_complex:
        return torch.complex(mat[..., 0], mat[..., 1])
    else:
        return ComplexTensor(mat[..., 0], mat[..., 1])


def get_mfmvdr_vector(gammax, Phi, use_torch_solver: bool = True, eps: float = EPS):
    """Compute conventional MFMPDR/MFMVDR filter.

    Args:
        gammax (torch.complex64/ComplexTensor): (..., L, N)
        Phi (torch.complex64/ComplexTensor): (..., L, N, N)
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        eps (float)
    Returns:
        beamforming_vector (torch.complex64/ComplexTensor): (..., L, N)
    """
    # (..., L, N)
    if use_torch_solver:
        numerator = solve(gammax.unsqueeze(-1), Phi).squeeze(-1)
    else:
        numerator = matmul(inverse(Phi), gammax.unsqueeze(-1)).squeeze(-1)
    denominator = einsum("...d,...d->...", gammax.conj(), numerator)
    return numerator / (denominator.real.unsqueeze(-1) + eps)


def filter_minimum_gain_like(
    G_min, w, y, alpha=None, k: float = 10.0, eps: float = EPS
):
    """Approximate a minimum gain operation.

    speech_estimate = alpha w^H y + (1 - alpha) G_min Y,
    where alpha = 1 / (1 + exp(-2 k x)), x = w^H y - G_min Y

    Args:
        G_min (float): minimum gain
        w (torch.complex64/ComplexTensor): filter coefficients (..., L, N)
        y (torch.complex64/ComplexTensor): buffered and stacked input (..., L, N)
        alpha: mixing factor
        k (float): scaling in tanh-like function
        esp (float)
    Returns:
        output (torch.complex64/ComplexTensor): minimum gain-filtered output
        alpha (float): optional
    """
    # (..., L)
    filtered_input = einsum("...d,...d->...", [w.conj(), y])
    # (..., L)
    Y = y[..., -1]
    return minimum_gain_like(G_min, Y, filtered_input, alpha, k, eps)


def minimum_gain_like(
    G_min, Y, filtered_input, alpha=None, k: float = 10.0, eps: float = EPS
):
    if alpha is None:
        diff = (filtered_input + eps).abs() - (G_min * Y + eps).abs()
        alpha = 1.0 / (1.0 + torch.exp(-2 * k * diff))
        return_alpha = True
    else:
        return_alpha = False
    output = alpha * filtered_input + (1 - alpha) * G_min * Y
    if return_alpha:
        return output, alpha
    else:
        return output
