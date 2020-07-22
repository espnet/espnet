"""
This script is used to construct convolutional beamformers.
Copyright 2020  Wangyou Zhang
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import List
from typing import Union

import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends import beamformer


def inv(z):
    # z.shape: ..., N, N
    real, imag = z.real, z.imag
    dim_N = z.size(-1)
    C = torch.cat(
        (torch.cat((real, -imag), dim=-1), torch.cat((imag, real), dim=-1)), dim=-2
    )
    invC = torch.inverse(C)[..., :dim_N]
    return ComplexTensor(invC[..., :dim_N, :], invC[..., dim_N:, :])


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
    if indices is None:
        frame_length2 = frame_length - 1
        # pad to the right at the last dimension of `signal` (time dimension)
        if do_padding:
            # (..., T) --> (..., T + bdelay + frame_length - 2)
            signal = FC.pad(
                signal, (bdelay + frame_length2 - 1, 0), "constant", pad_value
            )

        # indices:
        # [[ 0, 1, ..., frame_length2 - 1,              frame_length2 - 1 + bdelay ],
        #  [ 1, 2, ..., frame_length2,                  frame_length2 + bdelay     ],
        #  [ 2, 3, ..., frame_length2 + 1,              frame_length2 + 1 + bdelay ],
        #  ...
        #  [ T-bdelay-frame_length2, ..., T-1-bdelay,   T-1 ]
        indices = [
            [*range(i, i + frame_length2), i + frame_length2 + bdelay - 1]
            for i in range(0, signal.shape[-1] - frame_length2 - bdelay + 1, frame_step)
        ]

    if isinstance(signal, ComplexTensor):
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
        return ComplexTensor(real, imag)
    else:
        # (..., T - bdelay - frame_length + 2, frame_length)
        signal = signal[..., indices]
        # signal[..., :-1] = -signal[..., :-1]
        return signal


def get_covariances(
    Y: ComplexTensor,
    inverse_power: torch.Tensor,
    bdelay: int,
    btaps: int,
    get_vector: bool = False,
) -> ComplexTensor:
    """Calculates the power normalized spatio-temporal

     covariance matrix of the framed signal.

    Args:
        Y : Complext STFT signal with shape (B, F, C, T)
        inverse_power : Weighting factor with shape (B, F, T)

    Returns:
        Correlation matrix of shape (B, F, (btaps+1) * C, (btaps+1) * C)
        Correlation vector of shape (B, F, btaps + 1, C, C)
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
    Psi = FC.reverse(Psi, dim=-1)
    Psi_norm = Psi * inverse_power[..., None, bdelay + btaps - 1 :, None]

    # let T' = T - bdelay - btaps + 1
    # (B, F, C, T', btaps + 1) x (B, F, C, T', btaps + 1)
    #  -> (B, F, btaps + 1, C, btaps + 1, C)
    covariance_matrix = FC.einsum("bfdtk,bfetl->bfkdle", (Psi, Psi_norm.conj()))

    # (B, F, btaps + 1, C, btaps + 1, C)
    #   -> (B, F, (btaps + 1) * C, (btaps + 1) * C)
    covariance_matrix = covariance_matrix.view(
        Bs, Fdim, (btaps + 1) * C, (btaps + 1) * C
    )

    if get_vector:
        # (B, F, C, T', btaps + 1) x (B, F, C, T')
        #    --> (B, F, btaps +1, C, C)
        covariance_vector = FC.einsum(
            "bfdtk,bfet->bfked", (Psi_norm, Y[..., bdelay + btaps - 1 :].conj())
        )
        return covariance_matrix, covariance_vector
    else:
        return covariance_matrix


def get_WPD_filter(
    Phi: ComplexTensor,
    Rf: ComplexTensor,
    reference_vector: torch.Tensor,
    eps: float = 1e-15,
) -> ComplexTensor:
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
        Phi (ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the PSD of zero-padded speech [x^T(t,f) 0 ... 0]^T.
        Rf (ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, (btaps+1) * C)
            is the reference_vector.
        eps (float):

    Returns:
        filter_matrix (ComplexTensor): (B, F, (btaps + 1) * C)
    """
    try:
        inv_Rf = inv(Rf)
    except Exception:
        try:
            reg_coeff_tensor = (
                ComplexTensor(torch.rand_like(Rf.real), torch.rand_like(Rf.real)) * 1e-4
            )
            Rf = Rf / 10e4
            Phi = Phi / 10e4
            Rf += reg_coeff_tensor
            inv_Rf = inv(Rf)
        except Exception:
            reg_coeff_tensor = (
                ComplexTensor(torch.rand_like(Rf.real), torch.rand_like(Rf.real)) * 1e-1
            )
            Rf = Rf / 10e10
            Phi = Phi / 10e10
            Rf += reg_coeff_tensor
            inv_Rf = inv(Rf)

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum("...ec,...cd->...ed", [inv_Rf, Phi])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum("...fec,...c->...fe", [ws, reference_vector])
    # (B, F, (btaps + 1) * C)
    return beamform_vector


def get_WPD_filter_v2(
    Phi: ComplexTensor,
    Rf: ComplexTensor,
    reference_vector: torch.Tensor,
    eps: float = 1e-15,
) -> ComplexTensor:
    """Return the WPD vector with filter v2.

        WPD is the Weighted Power minimization Distortionless response
        convolutional beamformer. As follows:

        h = (Rf^-1 @ Phi_{xx}) @ u / tr[(Rf^-1) @ Phi_{xx}]

       This implementaion is more efficient than `get_WPD_filter` as
        it skips unnecessary computation with zeros.

    Reference:
        T. Nakatani and K. Kinoshita, "A Unified Convolutional Beamformer
        for Simultaneous Denoising and Dereverberation," in IEEE Signal
        Processing Letters, vol. 26, no. 6, pp. 903-907, June 2019, doi:
        10.1109/LSP.2019.2911179.
        https://ieeexplore.ieee.org/document/8691481

    Args:
        Phi (ComplexTensor): (B, F, C, C)
            is speech PSD.
        Rf (ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, C)
            is the reference_vector.
        eps (float):

    Returns:
        filter_matrix (ComplexTensor): (B, F, (btaps+1) * C)
    """
    C = reference_vector.shape[-1]
    try:
        inv_Rf = inv(Rf)
    except Exception:
        try:
            reg_coeff_tensor = (
                ComplexTensor(torch.rand_like(Rf.real), torch.rand_like(Rf.real)) * 1e-4
            )
            Rf = Rf / 10e4
            Phi = Phi / 10e4
            Rf += reg_coeff_tensor
            inv_Rf = inv(Rf)
        except Exception:
            reg_coeff_tensor = (
                ComplexTensor(torch.rand_like(Rf.real), torch.rand_like(Rf.real)) * 1e-1
            )
            Rf = Rf / 10e10
            Phi = Phi / 10e10
            Rf += reg_coeff_tensor
            inv_Rf = inv(Rf)
    # (B, F, (btaps+1) * C, (btaps+1) * C) --> (B, F, (btaps+1) * C, C)
    inv_Rf_pruned = inv_Rf[..., :C]
    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum("...ec,...cd->...ed", [inv_Rf_pruned, Phi])
    # ws: (..., (btaps+1) * C, C) / (...,) -> (..., (btaps+1) * C, C)
    ws = numerator / (FC.trace(numerator[..., :C, :])[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum("...fec,...c->...fe", [ws, reference_vector])
    # (B, F, (btaps+1) * C)
    return beamform_vector


def perform_WPD_filtering(
    filter_matrix: ComplexTensor, Y: ComplexTensor, bdelay: int, btaps: int
) -> ComplexTensor:
    """perform_filter_operation

    Args:
        filter_matrix: Filter matrix (B, F, (btaps + 1) * C)
        Y : Complex STFT signal with shape (B, F, C, T)

    Returns:
        enhanced (ComplexTensor): (B, F, T)
    """
    # (B, F, C, T) --> (B, F, C, T, btaps + 1)
    Ytilde = signal_framing(Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Ytilde = FC.reverse(Ytilde, dim=-1)

    Bs, Fdim, C, T = Y.shape
    # --> (B, F, T, btaps + 1, C) --> (B, F, T, (btaps + 1) * C)
    Ytilde = Ytilde.permute(0, 1, 3, 4, 2).contiguous().view(Bs, Fdim, T, -1)
    # (B, F, T, 1)
    enhanced = FC.einsum("...tc,...c->...t", [Ytilde, filter_matrix.conj()])
    return enhanced


if __name__ == "__main__":
    ############################################
    #                  Example                 #
    ############################################
    eps = 1e-10
    btaps = 5
    bdelay = 3
    # pretend to be some STFT: (B, F, C, T)
    Z = ComplexTensor(torch.rand(4, 256, 2, 518), torch.rand(4, 256, 2, 518))

    # Calculate power: (B, F, C, T)
    power = Z.real ** 2 + Z.imag ** 2
    # pretend to be some mask
    mask_speech = torch.ones_like(Z.real)
    # (..., C, T) * (..., C, T) -> (..., C, T)
    power = power * mask_speech
    # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
    power = power.mean(dim=-2)
    # (B, F, T) --> (B * F, T)
    power = power.view(-1, power.shape[-1])
    inverse_power = 1 / torch.clamp(power, min=eps)

    B, Fdim, C, T = Z.shape

    # covariance matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
    covariance_matrix = get_covariances(
        Z, inverse_power, bdelay, btaps, get_vector=False
    )

    # speech signal PSD: (B, F, C, C)
    psd_speech = beamformer.get_power_spectral_density_matrix(
        Z, mask_speech, btaps, normalization=True
    )

    # reference vector: (B, C)
    ref_channel = 0
    u = torch.zeros(*(Z.size()[:-3] + (Z.size(-2),)), device=Z.device)
    u[..., ref_channel].fill_(1)

    # (B, F, (btaps + 1) * C)
    WPD_filter = get_WPD_filter_v2(psd_speech, covariance_matrix, u)

    # (B, F, T)
    enhanced = perform_WPD_filtering(WPD_filter, Z, bdelay, btaps)
