from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from packaging.version import parse as V

from espnet2.enh.layers.fasnet import FaSNet_TAC
from espnet2.enh.layers.ifasnet import iFaSNet
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class FaSNetSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        enc_dim: int,
        feature_dim: int,
        hidden_dim: int,
        layer: int,
        segment_size: int,
        num_spk: int,
        win_len: int,
        context_len: int,
        fasnet_type: str,
        dropout: float = 0.0,
        sr: int = 16000,
        predict_noise: bool = False,
    ):
        """Filter-and-sum Network (FaSNet) Separator

        Args:
            input_dim: required by AbsSeparator. Not used in this model.
            enc_dim: encoder dimension
            feature_dim: feature dimension
            hidden_dim: hidden dimension in DPRNN
            layer: number of DPRNN blocks in iFaSNet
            segment_size: dual-path segment size
            num_spk: number of speakers
            win_len: window length in millisecond
            context_len: context length in millisecond
            fasnet_type: 'fasnet' or 'ifasnet'.
                Select from origin fasnet or Implicit fasnet
            dropout: dropout rate. Default is 0.
            sr: samplerate of input audio
            predict_noise: whether to output the estimated noise signal
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        assert fasnet_type in ["fasnet", "ifasnet"], "only support fasnet and ifasnet"

        FASNET = FaSNet_TAC if fasnet_type == "fasnet" else iFaSNet

        self.fasnet = FASNET(
            enc_dim=enc_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            layer=layer,
            segment_size=segment_size,
            nspk=num_spk + 1 if predict_noise else num_spk,
            win_len=win_len,
            context_len=context_len,
            sr=sr,
            dropout=dropout,
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): (Batch, samples, channels)
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            separated (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        assert input.dim() == 3, "only support input shape: (Batch, samples, channels)"
        # currently only support for fixed-array

        input = input.permute(0, 2, 1)

        none_mic = torch.zeros(1, dtype=input.dtype)

        separated = self.fasnet(input, none_mic)

        separated = list(separated.unbind(dim=1))

        others = {}
        if self.predict_noise:
            *separated, noise = separated
            others["noise1"] = noise

        return separated, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
