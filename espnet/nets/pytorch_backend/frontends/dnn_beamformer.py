from typing import Tuple

import numpy as np
import torch
import torch.nn
import torch.nn as nn
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.encoders import CNN
from espnet.nets.pytorch_backend.encoders import BRNN
from espnet.nets.pytorch_backend.encoders import BRNNP
from espnet.nets.pytorch_backend.frontends.beamformer \
    import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_power_spectral_density_matrix
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class DNN_MVDR(nn.Module):
    """

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783

    """
    def __init__(self,
                 bidim,
                 btype='blstmp',
                 blayers=3,
                 bunits=300,
                 bprojs=320,
                 dropout_rate=0.0,
                 badim=320,
                 ref_channel: int=None):
        super().__init__()
        self.mask = MaskEstimator(btype, bidim, blayers, bunits, bprojs,
                                  dropout_rate)
        self.ref = AttentionReference(bidim, bprojs, badim)
        self.ref_channel = ref_channel

    def forward(self, data: ComplexTensor, ilens: torch.LongTensor) \
            -> Tuple[ComplexTensor, torch.LongTensor]:
        """

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)
        """
        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)

        (speech, noise, hs), _ = self.mask(data, ilens)

        psd_speech = get_power_spectral_density_matrix(data, speech)
        psd_noise = get_power_spectral_density_matrix(data, noise)

        # u: (B, C)
        if self.ref_channel is None:
            u, _ = self.ref(psd_speech, hs, ilens)
        else:
            # (optional) Create onehot vector for fixed reference microphone
            u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                            device=data.device)
            u[..., self.ref_channel].fill_(1)

        ws = get_mvdr_vector(psd_speech, psd_noise, u)
        enhanced = apply_beamforming_vector(ws, data)

        # (..., F, T) -> (..., T, F)
        enhanced = enhanced.transpose(-1, -2)
        return enhanced, ilens


class AttentionReference(nn.Module):
    def __init__(self, bidim, bprojs, att_dim):
        super().__init__()
        self.mlp_psd = nn.Linear(bidim, att_dim)
        self.mlp_state = nn.Linear(bprojs, att_dim, bias=False)
        self.gvec = nn.Linear(att_dim, 1)

    def forward(self,
                psd_in: ComplexTensor,
                state_feat: torch.Tensor,
                ilens: torch.LongTensor,
                scaling: float=2.0) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            state_feat (torch.Tensor): (B, F, C, T)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        # (..., F, C, T) -> (..., C, T, F)
        state_feat = state_feat.permute(0, 2, 3, 1)

        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        assert state_feat.size()[:2] == (B, C), state_feat.size()
        # psd_in: (B, F, C, C)
        psd = psd_in.masked_fill(torch.eye(C, dtype=torch.uint8,
                                           device=psd_in.device), 0)
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real ** 2 + psd.imag ** 2) ** 0.5

        # state_feat: (..., C, T, F) -> (..., C, F)
        state_feat = \
            state_feat.sum(dim=-2) / ilens[:, None, None].type_as(state_feat)

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F) -> (B, C, F2)
        mlp_state = self.mlp_state(state_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd + mlp_state)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens


class MaskEstimator(torch.nn.Module):
    def __init__(self, btype, bidim, blayers, bunits, bprojs, dropout):
        super().__init__()
        # subsampling is not performed in mask estimation network
        subsample = np.ones(blayers + 1, dtype=np.int)

        if btype == 'blstm':
            self.blstm = BRNN(bidim, blayers, bunits, bprojs, dropout)
        elif btype == 'blstmp':
            self.blstm = BRNNP(bidim, blayers, bunits,
                                bprojs, subsample, dropout)
        elif btype == 'cnn':
            self.blstm = CNN(bidim, blayers, bunits, bprojs, residual=False)
        else:
            raise ValueError(
                "Error: need to specify an appropriate architecture: {}"
                .format(btype))
        self.btype = btype
        self.lo_S = torch.nn.Linear(bprojs, bidim)
        self.lo_N = torch.nn.Linear(bprojs, bidim)

    def forward(self, xs: ComplexTensor, ilens: torch.LongTensor) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     torch.LongTensor]:
        """
        Args:
            xs (ComplexTensor): (B, F, C, T)
            ilens (torch.Tensor): (B,)
        Returns:
            (ms_S, ms_N, xs), ilens

            ms_S (torch.Tensor): (B, F, C, T)
            ms_N (torch.Tensor): (B, F, C, T)
            xs (torch.Tensor): (B, F, C, T)
            ilens (torch.Tensor): (B,)
        """
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.permute(0, 2, 3, 1)

        B, C, T, _ = xs.size()
        # Calculate amplitude
        xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        # (B, C, T, F) -> (B * C, T, F)
        xs = xs.view(B * C, xs.size(-2), -1)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(B * C)
        xs, _ = self.blstm(xs, ilens_)
        xs = xs.view(B, C, T, -1)

        # ms_S, ms_N: (B, C, T, F)
        ms_S = self.lo_S(xs)
        ms_N = self.lo_N(xs)

        # Zero padding
        pad_mask = make_pad_mask(ilens, ms_S, length_dim=2)
        ms_S.masked_fill(pad_mask, 0.0)
        ms_N.masked_fill(pad_mask, 0.0)

        ms_S = torch.sigmoid(ms_S)
        ms_N = torch.sigmoid(ms_N)

        # (B, C, T, F) -> (B, F, C, T)
        xs = xs.permute(0, 3, 1, 2)
        ms_S = ms_S.permute(0, 3, 1, 2)
        ms_N = ms_N.permute(0, 3, 1, 2)

        return (ms_S, ms_N, xs), ilens
