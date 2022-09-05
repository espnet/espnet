from typing import List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import (
    is_complex,
    is_torch_complex_tensor,
    stack,
    new_complex_like,
    einsum,
    cat,
    reverse,
)


def mag_sq(x: torch.Tensor):
    if is_complex(x):
        return x.real.pow(2) + x.imag.pow(2)
    else:
        return x.pow(2)


class LaplaceModel(torch.nn.Module):
    """
    This model implements the computation of the auxiliary variable
    in IVA with a spherical Laplacian source model
    """

    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        self._eps = eps if eps is not None else 1e-5

        # just so that training works
        self.fake = torch.nn.Parameter(torch.zeros(1))

    def cost(self, Y):
        return torch.linalg.norm(Y, dim=-2).mean(dim=-1).sum(dim=-1)

    def forward(self, X: torch.Tensor):
        """LaplaceModel forward function.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            X (torch.complex64/ComplexTensor): (B, C, F, T)
        Returns:
            masks: (torch.float32): (B, C, F, T)
        """
        # sum power over frequencies
        X_mag_sq = mag_sq(X)
        denom = 2.0 * torch.sqrt(X_mag_sq.sum(dim=-2).clamp(self._eps))
        _, r = torch.broadcast_tensors(X_mag_sq, denom[..., None, :])

        r_inv = 1.0 / torch.clamp(r, min=self._eps)
        return r_inv


class GaussModel(torch.nn.Module):
    """
    This model implements the computation of the auxiliary variable
    in IVA with a time-varying Gauss model
    """

    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        self._eps = eps if eps is not None else 1e-5

        # just so that training works
        self.fake = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor):
        """GaussModel forward function.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            X (torch.complex64/ComplexTensor): (B, C, F, T)
        Returns:
            masks: (torch.float32): (B, C, F, T)
        """
        # sum power over frequencies
        X_mag_sq = mag_sq(X)
        denom = X_mag_sq.mean(dim=-2)
        _, r = torch.broadcast_tensors(X_mag_sq, denom[..., None, :])

        r_inv = 1.0 / torch.clamp(r, min=self._eps)
        return r_inv


def hermite(A: torch.Tensor, dim1: Optional[int] = -2, dim2: Optional[int] = -1):
    if is_complex(A):
        return A.transpose(dim1, dim2).conj()
    else:
        return A.transpose(dim1, dim2)


def divide(num, denom, eps=1e-7):
    if is_torch_complex_tensor(num):
        return torch.view_as_complex(
            torch.view_as_real(num) / torch.clamp(denom[..., None], min=eps)
        )
    else:
        return num / torch.clamp(denom[..., None], min=eps)


def inv_2x2(W: Union[torch.Tensor, ComplexTensor], eps: Optional[float] = 1e-6):

    if W.shape[-1] != W.shape[-2] or W.shape[-1] != 2:
        raise ValueError("This function is specialized for 2x2 matrices")

    W11 = W[..., 0, 0]
    W21 = W[..., 1, 0]
    W12 = W[..., 0, 1]
    W22 = W[..., 1, 1]

    det = W11 * W22 - W12 * W21

    # complex clamp
    det.masked_fill_(abs(det) < eps, eps)

    adjoint = stack((stack((W22, -W21), dim=-1), stack((-W12, W11), dim=-1)), dim=-1)

    W_inv = adjoint / det[..., None, None]

    return W_inv


def eigh_2x2(
    A: Union[torch.Tensor, ComplexTensor],
    B: Union[torch.Tensor, ComplexTensor, None] = None,
    eps: Optional[float] = 0.0,
) -> Tuple[Union[torch.Tensor, ComplexTensor]]:
    """
    Specialized routine for batched 2x2 EVD and GEVD for complex hermitian matrices
    """
    assert 2 == A.shape[-1]
    assert A.shape[-2] == A.shape[-1]

    if B is not None:
        assert B.shape[-1] == 2
        assert B.shape[-1] == B.shape[-2]

        # broadcast
        if is_torch_complex_tensor(A):
            A, B = torch.broadcast_tensors(A, B)
        else:
            A_real, B_real = torch.broadcast_tensors(A.real, B.real)
            A_imag, B_imag = torch.broadcast_tensors(A.imag, B.imag)
            A = new_complex_like(A, (A_real, A_imag))
            B = new_complex_like(B, (B_real, B_imag))

        # notation
        a11 = A[..., 0, 0]
        a12 = A[..., 0, 1]
        a22 = A[..., 1, 1]
        b11 = B[..., 0, 0]
        b12 = B[..., 0, 1]
        b22 = B[..., 1, 1]

        # coefficient of secular equation: x - b * x + c
        a11b22 = a11.real * b22.real
        a22b11 = a22.real * b11.real
        re_a12b12c = a12.real * b12.real + a12.imag * b12.imag
        b = a11b22 + a22b11 - 2.0 * re_a12b12c

        det_A = a11.real * a22.real - mag_sq(a12)
        det_B = b11.real * b22.real - mag_sq(b12)
        c = det_A * det_B

        # discrimant of secular equation
        delta = b.pow(2) - 4 * c

        # we clamp to zero to avoid numerical inaccuracies
        # we know the minimum is zero because A and B should
        # be symmetric or hermitian symmetric
        delta = torch.clamp(delta, min=eps)

        # fill the eigenvectors in ascending order
        eigenvalues = A.new_full(A.shape[:-1], 0)
        eigenvalues[..., 0] = 0.5 * (b - torch.sqrt(delta))  # small eigenvalue
        eigenvalues[..., 1] = 0.5 * (b + torch.sqrt(delta))  # large eigenvalue

        # normalize the eigenvalues
        eigenvalues = eigenvalues / torch.clamp(det_B[..., None], min=eps)

        # notation
        ev1 = eigenvalues[..., 0]
        ev2 = eigenvalues[..., 1]

        # now fill the eigenvectors
        eigenvectors = A.new_full(A.shape, 0)
        # vector corresponding to small eigenvalue
        eigenvectors[..., 0, 0] = ev1 * b12 - a12
        eigenvectors[..., 1, 0] = a11 - ev1 * b11
        # vector corresponding to large eigenvalue
        eigenvectors[..., 0, 1] = ev2 * b12 - a12
        eigenvectors[..., 1, 1] = a11 - ev2 * b11

    else:
        # Do the EVD

        # secular equation: a * lambda^2 - b * lambda + c
        # where lambda is the eigenvalue
        b = A[..., 0, 0].real + A[..., 1, 1].real
        c = A[..., 0, 0].real * A[..., 1, 1].real - mag_sq(A[..., 0, 1])

        # discrimant of secular equation
        delta = b.pow(2) - 4 * c

        # we clamp to zero to avoid numerical inaccuracies
        # we know the minimum is zero because A and B should
        # be symmetric or hermitian symmetric
        delta = torch.clamp(delta, min=eps)

        # fill the eigenvectors in ascending order
        eigenvectors = A.new_full(A.shape, 0)
        eigenvalues[..., 0] = 0.5 * (delta - torch.sqrt(delta))  # small eigenvalue
        eigenvalues[..., 1] = 0.5 * (delta + torch.sqrt(delta))  # large eigenvalue

        # now fill the eigenvectors
        eigenvectors = A.new_zeros(A.shape)
        # vector corresponding to small eigenvalue
        eigenvectors[..., 0, 0] = A[..., 0, 1]
        eigenvectors[..., 1, 0] = eigenvalues[..., 0] - A[..., 1, 1]
        # vector corresponding to large eigenvalue
        eigenvectors[..., 0, 1] = eigenvalues[..., 1] - A[..., 0, 0]
        eigenvectors[..., 1, 1] = eigenvectors[..., 0, 0].conj()

    norm = torch.sqrt(torch.sum(mag_sq(eigenvectors), dim=-2, keepdim=True))
    eigenvectors = divide(eigenvectors, norm, eps=eps)

    return eigenvalues, eigenvectors


def select_most_energetic(
    x: Union[torch.Tensor, ComplexTensor],
    num: int,
    dim: Optional[int] = -2,
    dim_reduc: Optional[int] = -1,
):
    """Selects the `num` indices with most power.
    Args:
        x (torch.Tensor/ComplexTensor): The input tensor of shape (n_batch, n_channels, n_samples).
        num (int): The number of signals to select.
        dim: The axis where the selection should occur.
        dim_reduc: The axis where to perform the reduction.
    """

    power = x.abs().pow(2).mean(axis=dim_reduc, keepdim=True)

    index = torch.argsort(power.transpose(dim, -1), axis=-1, descending=True)
    index = index[..., :num]

    # need to broadcast to target size
    x_tgt = x.transpose(dim, -1)[..., :num]
    _, index = torch.broadcast_tensors(x_tgt, index)

    # reorder index axis
    index = index.transpose(dim, -1)

    ret = x.gather(dim, index)
    return ret


def spatial_model_update_iss(
    X: Union[torch.Tensor, ComplexTensor],
    weights: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    A: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-3,
):
    """Apply the spatial model update via the iterative source steering rules.
    Args:
        X (torch.Tensor/ComplexTensor): shape (..., n_channels, n_frequencies, n_frames)
            The input signal.
        weights (torch.Tensor): shape (..., n_channels, n_frequencies, n_frames)
            The weights obtained from the source model to compute the weighted statistics.
        W (torch.Tensor): shape (..., n_frequencies, n_channels, n_channels), optional
            The demixing matrix, it is updated if provided.
        A (torch.Tensor): shape (..., n_frequencies, n_channels, n_channels), optional
            The mixing matrix, it is updated if provided.
    Returns:
        X (torch.Tensor/ComplexTensor): shape (..., n_frequencies, n_channels, n_frames)
            The updated source estimates.
    """
    n_chan, n_freq, n_frames = X.shape[-3:]

    # Update now the demixing matrix
    for s in range(n_chan):
        v_num = (
            einsum(
                "...cfn,...cfn,...fn->...cf",
                X,
                new_complex_like(X, (weights, torch.zeros_like(weights))),
                X[..., s, :, :].conj(),
            )
            / n_frames
        )
        v_denom = (
            torch.einsum("...cfn,...fn->...cf", weights, mag_sq(X[..., s, :, :]))
            / n_frames
        )

        v = v_num / torch.clamp(v_denom, min=eps)
        v_s = 1.0 - torch.rsqrt(torch.clamp(v_denom[..., s, :], min=eps))
        v = cat((v[..., :s, :], v_s[..., None, :], v[..., s + 1 :, :]), dim=-2)

        # update demixed signals
        X = X - einsum("...cf,...fn->...cfn", v, X[..., s, :, :])

        if W is not None:
            W = W - einsum("...cf,...fd->...fcd", v, W[..., s, :])

        if A is not None:
            u = einsum("...fcd,...df->...fc", A, v) / torch.clamp(
                1.0 - v_s[..., None], min=eps
            )
            A = cat(
                (A[..., :s], A[..., [s]] + u[..., None], A[..., s + 1 :]), dim=-1
            )

    return X, W, A


def spatial_model_update_ip2(
    Xo: Union[torch.Tensor, ComplexTensor],
    weights: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    A: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-7,
):
    """Apply the spatial model update via the generalized eigenvalue decomposition.
    This method is specialized for two channels.
    Args:
        Xo (torch.Tensor/ComplexTensor): shape (..., n_frequencies, n_channels, n_frames)
            The microphone input signal with n_chan == 2.
        weights (torch.Tensor): shape (..., n_frequencies, n_channels, n_frames)
            The weights obtained from the source model to compute
            the weighted statistics.
    Returns:
        X (torch.Tensor/ComplexTensor): shape (..., n_frequencies, n_channels, n_frames)
            The updated source estimates.
    """
    assert Xo.shape[-3] == 2, "This method is specialized for two channels processing."

    V = []
    for k in [0, 1]:
        # shape: (n_batch, n_freq, n_chan, n_chan)
        Vloc = einsum(
            "...fn,...cfn,...dfn->...fcd", weights[..., k, :, :], Xo, Xo.conj()
        )
        Vloc = Vloc / Xo.shape[-1]
        # make sure V is hermitian symmetric
        Vloc = 0.5 * (Vloc + hermite(Vloc))
        V.append(Vloc)

    eigval, eigvec = eigh_2x2(V[1], V[0], eps=eps)

    # reverse order of eigenvectors
    eigvec = reverse(eigvec, dim=-1)

    scale_0 = abs(
        (eigvec[..., None, :, 0]).conj() @ (V[0] @ eigvec[..., :, None, 0])
    )
    scale_1 = abs(
        (eigvec[..., None, :, 1]).conj() @ (V[1] @ eigvec[..., :, None, 1])
    )
    scale = torch.cat((scale_0, scale_1), dim=-1)
    scale = torch.clamp(torch.sqrt(torch.clamp(scale, min=1e-7)), min=eps)
    eigvec = eigvec / scale

    if W is not None:
        W = hermite(eigvec)
        if A is not None:
            A = inv_2x2(W)

    X = einsum("...fcd,...dfn->...cfn", hermite(eigvec), Xo)

    return X, W, A


def demix(W, X_original):
    return einsum("...fcd,...dfn->...cfn", W, X_original)


def auxiva_iss_one_step(X, W, A, model, eps, two_chan_ip2, X_original):
    n_chan = W.shape[-1]

    # shape: (n_chan, n_freq, n_frames)
    # model takes as input a tensor of shape (..., n_frequencies, n_frames)
    weights = model(X)

    # we normalize the sources to have source to have unit variance prior to
    # computing the model
    g = torch.clamp(torch.mean(mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-5)
    X = divide(X, torch.sqrt(g))
    weights = weights * g

    if n_chan == 2 and two_chan_ip2:
        # Here are the exact/fast updates for two channels using the GEVD
        X, W, A = spatial_model_update_ip2(X_original, weights, W=W, A=A, eps=eps)

    else:
        # Iterative Source Steering updates
        X, W, A = spatial_model_update_iss(X, weights, W=W, A=A, eps=eps)

    return X, W, A


def auxiva_iss_one_step_dmc(W, A, model, eps, two_chan_ip2, X_original):
    """
    wrapper that does not take the separated signal as input
    to use with demixing matrix checkpointing
    """
    X = demix(W, X_original)
    X, W, A = auxiva_iss(X, W, A, model, eps, two_chan_ip2, X_original)
    return W, A


def wiener_weights(Y, model, eps):
    """
    Compute the Wiener weights using a noise model
    """
    mask = model(Y)
    Y_pwr = Y.real ** 2 + Y.imag ** 2
    noise_power = torch.mean(Y_pwr * mask, dim=-1, keepdim=True)
    signal_power = torch.mean(Y_pwr, dim=-1, keepdim=True)
    weights = 1.0 - noise_power / torch.clamp(signal_power, min=eps)
    return weights


def auxiva_iss(
    X: Union[torch.Tensor, ComplexTensor],
    n_iter: Optional[int] = 20,
    n_src: Optional[int] = None,
    model: Optional[callable] = None,
    eps: Optional[float] = 1e-6,
    two_chan_ip2: Optional[bool] = True,
    proj_back_mic: Optional[bool] = 0,
    use_dmc: Optional[bool] = False,
    use_wiener: Optional[bool] = False,
    checkpoints_iter: Optional[List[int]] = None,
    checkpoints_list: Optional[List] = None,
) -> Union[torch.Tensor, ComplexTensor]:

    """Blind source separation based on independent vector analysis with
    alternating updates of the mixing vectors.

    Args:
        X (torch.Tensor/ComplexTensor): shape (..., n_channels, n_frequencies, n_frames)
            STFT representation of the signal.
        n_iter (int) : optional
            The number of iterations (default 20).
        n_src (int): optional
            When n_src is less than the number of channels, the n_src most energetic
            are selected for the output.
        model: SourceModel
            The model of source distribution (default: Laplace).
        eps (float):
            A small constant to make divisions and the like numerically stable.
        two_chan_ip2 (bool):
            For the 2 channel case, use the more efficient IP2 algorithm (default: True).
            Ignored when using more than 2 channels.
        proj_back_mic (int):
            The microphone index to use as a reference when adjusting the scale and delay.
        use_dmc (bool): optional
            If True, use checkpointing of the demixing matrix to save memory
        checkpoints_iter (List[int]):
            Optionally, we can keep intermediate results for later analysis.
            Should be used together with checkpoints_list.
        checkpoints_list (List):
            An empty list can be passed and the intermediate signals are
            appended to the list for iterations numbers contained in checkpoints_iter.

    Returns:
        X (torch.Tensor/ComplexTensor): shape (..., n_channels, n_frequencies, n_frames)
            STFT representation of the signal after separation.
    """  # noqa: H405

    n_chan, n_freq, n_frames = X.shape[-3:]

    if model is None:
        model = LaplaceModel()

    # for now, only supports determined case
    assert callable(model)

    # keep track of original input signal
    X_original = X.clone()

    if proj_back_mic is not None or use_dmc:
        assert (
            0 <= proj_back_mic < n_chan
        ), "The reference microphone index must be between 0 and # channels - 1."
        W = X.new_full(X.shape[:-3] + (n_freq, n_chan, n_chan), 0)
        eye = torch.eye(n_chan)
        W[:] = new_complex_like(W, (eye, torch.zeros_like(eye)))
        A = W.clone()
    else:
        W = None
        A = None

    for epoch in range(n_iter):

        if checkpoints_iter is not None and epoch in checkpoints_iter:
            if use_dmc:
                X = demix(W, X_original)
            checkpoints_list.append(X)

        if use_dmc:
            # use demixing matrix checkpointing
            model_params = [p for p in self.model.parameters()]
            W, A = torch.utils.checkpoint.checkpoint(
                auxiva_iss_one_step_dmc,
                W,
                A,
                model,
                eps,
                two_chan_ip2,
                X_original,
                *model_params,
                preserve_rng_state=True
            )
        else:
            # use vanilla backprop
            X, W, A = auxiva_iss_one_step(X, W, A, model, eps, two_chan_ip2, X_original)

    if use_dmc:
        X = demix(W, X_original)

    if use_wiener:
        ww = wiener_weights(X, model, eps)
        X = X * ww

    if proj_back_mic is not None:
        # (batch, freq, -, src) -> (batch, src, freq, -)
        a = A[..., :, [proj_back_mic], :].permute((0, 3, 1, 2))
        X = a * X

    if n_src is not None and n_src < n_chan:
        # select sources based on energy
        X = select_most_energetic(X, num=n_src, dim=-3, dim_reduc=(-2, -1))

    return X
