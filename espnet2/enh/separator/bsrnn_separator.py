from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.bsrnn import BSRNN
from espnet2.enh.layers.complex_utils import is_complex, new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator


class BSRNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        num_channels: int = 16,
        num_layers: int = 6,
        target_fs: int = 48000,
        causal: bool = True,
        norm_type: str = "GN",
        ref_channel: Optional[int] = None,
    ):
        """Band-split RNN (BSRNN) separator.

        Reference:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim: (int) maximum number of frequency bins corresponding to
                `target_fs`
            num_spk: (int) number of speakers.
            num_channels: (int) feature dimension in the BandSplit block.
            num_layers: (int) number of processing layers.
            target_fs: (int) max sampling frequency that the model can handle.
            causal (bool): whether or not to apply causal modeling.
                if True, LSTM will be used instead of BLSTM for time modeling
            norm_type (str): type of the normalization layer (cfLN / cLN / BN / GN).
            ref_channel: (int) reference channel. not used for now.
        """
        super().__init__()

        self._num_spk = num_spk
        self.ref_channel = ref_channel

        self.bsrnn = BSRNN(
            input_dim=input_dim,
            num_channel=num_channels,
            num_layer=num_layers,
            target_fs=target_fs,
            causal=causal,
            num_spk=num_spk,
            norm_type=norm_type,
        )

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """BSRNN Forward.

        Args:
            input (torch.Tensor or ComplexTensor): STFT spectrum [B, T, (C,) F (,2)]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model.
                unused in this model.

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # B, T, (C,) F, 2
        if is_complex(input):
            feature = torch.stack([input.real, input.imag], dim=-1)
        else:
            assert input.size(-1) == 2, input.shape
            feature = input

        opt = {}
        if additional is not None and "fs" in additional:
            opt["fs"] = additional["fs"]
        masked = self.bsrnn(feature, **opt)
        # B, num_spk, T, F
        if not is_complex(input):
            masked = list(ComplexTensor(masked[..., 0], masked[..., 1]).unbind(1))
        else:
            masked = list(
                new_complex_like(input, (masked[..., 0], masked[..., 1])).unbind(1)
            )

        others = {}
        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
