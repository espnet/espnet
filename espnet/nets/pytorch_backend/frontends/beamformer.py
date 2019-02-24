import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


# Ported from https://github.com/fgnt/nn-gev


def get_power_spectral_density_matrix(xs: ComplexTensor, mask: torch.Tensor,
                                      normalization=True) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
    Returns
        psd (ComplexTensor): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])

    # Averaging mask along C: (..., C, T) -> (..., T)
    mask = mask.mean(dim=-2)

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / mask.sum(dim=-1)[..., None]

    # psd: (..., T, C, C)
    psd = psd_Y * mask[..., None, None]
    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(dim=-3)

    return psd


def get_mvdr_vector(psd_s: ComplexTensor,
                    psd_n: ComplexTensor,
                    reference_vector: torch.Tensor) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Citation:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum('...ec,...cd->...ed', [psd_n.inverse(), psd_s])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / FC.trace(numerator)[..., None, None]
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum('...fec,...c->...fe', [ws, reference_vector])
    return beamform_vector


# TODO(kamo): Implement forward-backward function for symeig
def get_mvdr_vector2(psd_s: ComplexTensor,
                     psd_n: ComplexTensor) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ A) / (A^H @ Npsd^-1 @ A)

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
    Returns:
        beamform_vector (ComplexTensor): (..., F, C)
    """
    eigenvals, eigenvecs = FC.symeig(psd_s)
    # eigenvals: (..., C) -> (...)
    index = torch.argmax(eigenvals, dim=-1)
    # pca_vector: (..., C, C) -> (..., C)
    pca_vector = torch.gather(
        eigenvecs, dim=-1, index=index[..., None]).unsqueeze(-1)

    # numerator: (..., C_1, C_2) x (..., C_2) -> (..., C_1)
    numerator = FC.einsum('...ec,...c->...e', [FC.inv(psd_n), pca_vector])
    # denominator: (..., C) x (..., F, C) -> (..., F)
    denominator = FC.einsum('...c,...dc->...d', [pca_vector.conj(), numerator])
    # h: (..., F, C) / (..., F) -> (..., F, C)
    beamform_vector = numerator / denominator[..., None]
    return beamform_vector


# TODO(kamo): Implement forward-backward function for symeig
def get_gev_vector(psd_s: ComplexTensor, psd_n: ComplexTensor)\
        -> ComplexTensor:
    """Returns the GEV(Generalized Eigen Value) beamforming vector.

        Spsd @ h =  ev x Npsd @ h

    Citation:
        NEURAL NETWORK BASED SPECTRAL MASK ESTIMATION FOR ACOUSTIC BEAMFORMING;
        Jahn Heymann et al.., 2016;
        https://ieeexplore.ieee.org/abstract/document/7471664

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
    Returns:
        beamform_vector (ComplexTensor): (..., F, C)

    """
    eigenvals, eigenvecs = FC.symeig(psd_s, psd_n)
    # eigenvals: (..., C) -> (...)
    index = torch.argmax(eigenvals, dim=-1)
    # beamform_vector: (..., C, C) -> (..., C)
    beamform_vector = torch.gather(
        eigenvecs, dim=-1, index=index[..., None]).unsqueeze(-1)
    return beamform_vector


def blind_analytic_normalization(beamform_vector: ComplexTensor,
                                 psd_n: ComplexTensor,
                                 eps=1e-10) -> ComplexTensor:
    """Reduces distortions in beamformed output.

        h = sqrt(|h* @ Npsd @ Npsd @ h|) / |h* @ Npsd @ h| x h

    Args:
        beamform_vector (ComplexTensor): (..., C)
        psd_n (ComplexTensor): (..., C, C)
    Returns:
        beamform_vector (ComplexTensor): (..., C)

    """
    # (..., C) x (..., C, C) x (..., C, C) x (..., C) -> (...)
    numerator = FC.einsum(
        '...a,...ab,...bc,...c->...',
        [beamform_vector.conj(), psd_n, psd_n, beamform_vector])
    numerator = numerator.abs().sqrt()

    # (..., C) x (..., C, C) x (..., C) -> (...)
    denominator = FC.einsum(
        '...a,...ab,...b->...',
        [beamform_vector.conj(), psd_n, beamform_vector])
    denominator = denominator.abs()

    # normalization: (...) / (...) -> (...)
    normalization = numerator / (denominator + eps)
    # beamform_vector: (..., C) * (...) -> (..., C)
    return beamform_vector * normalization[..., None]


def apply_beamforming_vector(beamform_vector: ComplexTensor,
                             mix: ComplexTensor) -> ComplexTensor:
    # (..., C) x (..., C, T) -> (..., T)
    es = FC.einsum('...c,...ct->...t', [beamform_vector.conj(), mix])
    return es
