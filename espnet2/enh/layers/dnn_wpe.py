from typing import Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import to_double, to_float
from espnet2.enh.layers.mask_estimator import MaskEstimator
from espnet2.enh.layers.wpe import wpe_one_iteration
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class DNN_WPE(torch.nn.Module):
    def __init__(
        self,
        wtype: str = "blstmp",
        widim: int = 257,
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        dropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask: bool = True,
        nmask: int = 1,
        nonlinear: str = "sigmoid",
        iterations: int = 1,
        normalization: bool = False,
        eps: float = 1e-6,
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        mask_flooring: bool = False,
        flooring_thres: float = 1e-6,
        use_torch_solver: bool = True,
    ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay
        self.eps = eps

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True
        self.diagonal_loading = diagonal_loading
        self.diag_eps = diag_eps
        self.mask_flooring = mask_flooring
        self.flooring_thres = flooring_thres
        self.use_torch_solver = use_torch_solver

        if self.use_dnn_mask:
            self.nmask = nmask
            self.mask_est = MaskEstimator(
                wtype,
                widim,
                wlayers,
                wunits,
                wprojs,
                dropout_rate,
                nmask=nmask,
                nonlinear=nonlinear,
            )
        else:
            self.nmask = 1

    def forward(
        self, data: Union[torch.Tensor, ComplexTensor], ilens: torch.LongTensor
    ) -> Tuple[
        Union[torch.Tensor, ComplexTensor],
        torch.LongTensor,
        Union[torch.Tensor, ComplexTensor],
    ]:
        """DNN_WPE forward function.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector

        Args:
            data: (B, T, C, F)
            ilens: (B,)
        Returns:
            enhanced (torch.Tensor or List[torch.Tensor]): (B, T, C, F)
            ilens: (B,)
            masks (torch.Tensor or List[torch.Tensor]): (B, T, C, F)
            power (List[torch.Tensor]): (B, F, T)
        """
        # (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)
        enhanced = [data for i in range(self.nmask)]
        masks = None
        power = None

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = [enh.real ** 2 + enh.imag ** 2 for enh in enhanced]
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                masks, _ = self.mask_est(data, ilens)
                # floor masks to increase numerical stability
                if self.mask_flooring:
                    masks = [m.clamp(min=self.flooring_thres) for m in masks]
                if self.normalization:
                    # Normalize along T
                    masks = [m / m.sum(dim=-1, keepdim=True) for m in masks]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = [p * masks[i] for i, p in enumerate(power)]

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = [p.mean(dim=-2).clamp(min=self.eps) for p in power]

            # enhanced: (..., C, T) -> (..., C, T)
            # NOTE(kamo): Calculate in double precision
            enhanced = [
                wpe_one_iteration(
                    to_double(data.contiguous()),
                    to_double(p),
                    taps=self.taps,
                    delay=self.delay,
                    inverse_power=self.inverse_power,
                )
                for p in power
            ]
            enhanced = [
                enh.to(dtype=data.dtype).masked_fill(make_pad_mask(ilens, enh.real), 0)
                for enh in enhanced
            ]

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = [enh.permute(0, 3, 2, 1) for enh in enhanced]
        if masks is not None:
            masks = (
                [m.transpose(-1, -3) for m in masks]
                if self.nmask > 1
                else masks[0].transpose(-1, -3)
            )
        if self.nmask == 1:
            enhanced = enhanced[0]

        return enhanced, ilens, masks, power

    def predict_mask(
        self, data: Union[torch.Tensor, ComplexTensor], ilens: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Predict mask for WPE dereverberation.

        Args:
            data (torch.complex64/ComplexTensor): (B, T, C, F), double precision
            ilens (torch.Tensor): (B,)
        Returns:
            masks (torch.Tensor or List[torch.Tensor]): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        """
        if self.use_dnn_mask:
            masks, ilens = self.mask_est(to_float(data.permute(0, 3, 2, 1)), ilens)
            # (B, F, C, T) -> (B, T, C, F)
            masks = [m.transpose(-1, -3) for m in masks]
            if self.nmask == 1:
                masks = masks[0]
        else:
            masks = None
        return masks, ilens
