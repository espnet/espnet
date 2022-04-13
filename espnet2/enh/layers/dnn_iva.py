"""DNN beamformer module."""
from distutils.version import LooseVersion
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import logging
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.iva import auxiva_iss
from espnet2.enh.layers.complex_utils import stack
from espnet2.enh.layers.complex_utils import to_double
from espnet2.enh.layers.complex_utils import to_float
from espnet2.enh.layers.mask_estimator import MaskEstimator


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class DNN_IVA(torch.nn.Module):
    """DNN mask based Beamformer.

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        http://proceedings.mlr.press/v70/ochiai17a/ochiai17a.pdf

    """

    def __init__(
        self,
        bidim,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        num_spk: int = 1,
        nonlinear: str = "sigmoid",
        dropout_rate: float = 0.0,
        badim: int = 320,
        ref_channel: int = -1,
        eps: float = 1e-6,
        mask_flooring: bool = False,
        flooring_thres: float = 1e-6,
        iva_iterations: int = 15,
        iva_train_iterations: int = None,
        iva_train_channels: int = None,
        use_dmc: bool = False,
    ):
        super().__init__()
        self.mask = MaskEstimator(
            btype,
            bidim,
            blayers,
            bunits,
            bprojs,
            dropout_rate,
            nmask=1,
            nonlinear=nonlinear,
        )
        self.ref_channel = ref_channel

        assert num_spk >= 1, num_spk
        self.num_spk = num_spk

        self.beamformer_type = "iva"

        self.eps = eps
        self.mask_flooring = mask_flooring
        self.flooring_thres = flooring_thres
        self.iterations = iva_iterations
        if iva_train_iterations is None:
            self.train_iterations = iva_iterations
        else:
            self.train_iterations = iva_train_iterations
        self.train_channels = iva_train_channels
        self.use_dmc = use_dmc

    def compute_mask(self, X):
        """
        wrapper for the mask network
        X:  shape == (B, C, F, T)
        """
        X = X.permute(0, 2, 1, 3)  # -> (B, F, C, T)

        X = torch.log(1.0 + X.real ** 2 + X.imag ** 2)

        mask, *_ = self.mask(X, self.current_ilens)  # shape is (B, F, C, T)
        mask = mask[0]

        if self.mask_flooring:
            mask = torch.clamp(mask, min=self.flooring_thres)

        mask = mask.permute(0, 2, 1, 3)  # -> (B, C, F, T)
        return mask

    def forward(
        self,
        data: Union[torch.Tensor, ComplexTensor],
        ilens: torch.LongTensor,
        powers: Optional[List[torch.Tensor]] = None,
        oracle_masks: Optional[List[torch.Tensor]] = None,
        iterations: Optional[int] = None,
    ) -> Tuple[Union[torch.Tensor, ComplexTensor], torch.LongTensor, torch.Tensor]:
        """DNN_IVA forward function.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (torch.complex64/ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
            powers (List[torch.Tensor] or None): used for wMPDR or WPD (B, F, T)
            oracle_masks (List[torch.Tensor] or None): oracle masks (B, F, C, T)
                if not None, oracle_masks will be used instead of self.mask
        Returns:
            enhanced (torch.complex64/ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)
            masks (torch.Tensor): (B, T, C, F)
        """
        # data (B, T, C, F) -> (B, C, F, T)
        data = data.permute(0, 2, 3, 1)
        # data_d = to_double(data)

        if iterations is None:
            if self.training:
                iterations = self.train_iterations
            else:
                iterations = self.iterations


        data = torch.view_as_complex(torch.stack((data.real, data.imag), dim=-1))

        if self.training: 
            if self.train_channels is not None and data.shape[-3] > self.train_channels:
                data = data[..., :self.train_channels, :, :]

        self.current_ilens = ilens

        enhanced = auxiva_iss(
            data,
            n_src=self.num_spk,
            n_iter=iterations,
            model=self.compute_mask,
            eps=self.eps,
            proj_back_mic=self.ref_channel,
        )

        enhanced = ComplexTensor(enhanced.real, enhanced.imag)

        enhanced = [e.transpose(-2, -1) for e in enhanced.transpose(0, 1)]

        return enhanced, ilens, None
