from collections import OrderedDict
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch

from espnet2.enh.layers.fasnet import FaSNet_TAC
from espnet2.enh.layers.ifasnet import iFaSNet
from espnet2.enh.separator.abs_separator import AbsSeparator


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


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
        """
        super().__init__()

        self._num_spk = num_spk

        assert fasnet_type in ["fasnet", "ifasnet"], "only support fasnet and ifasnet"

        FASNET = FaSNet_TAC if fasnet_type == "fasnet" else iFaSNet

        self.fasnet = FASNET(
            enc_dim=enc_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            layer=layer,
            segment_size=segment_size,
            nspk=num_spk,
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

        return separated, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
