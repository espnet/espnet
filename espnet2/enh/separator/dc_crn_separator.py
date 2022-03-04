from collections import OrderedDict
from distutils.version import LooseVersion
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.layers.dc_crn import DC_CRN
from espnet2.enh.separator.abs_separator import AbsSeparator


EPS = torch.finfo(torch.get_default_dtype()).eps
is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class DC_CRNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        input_channels: List = [2, 16, 32, 64, 128, 256],
        enc_hid_channels: int = 8,
        enc_kernel_size: Tuple = (1, 3),
        enc_padding: Tuple = (0, 1),
        enc_last_kernel_size: Tuple = (1, 4),
        enc_last_stride: Tuple = (1, 2),
        enc_last_padding: Tuple = (0, 1),
        enc_layers: int = 5,
        skip_last_kernel_size: Tuple = (1, 3),
        skip_last_stride: Tuple = (1, 1),
        skip_last_padding: Tuple = (0, 1),
        glstm_groups: int = 2,
        glstm_layers: int = 2,
        glstm_bidirectional: bool = False,
        glstm_rearrange: bool = False,
        mode: str = "masking",
        ref_channel: int = 0,
    ):
        """Densely-Connected Convolutional Recurrent Network (DC-CRN) Separator

        Reference:
            Deep Learning Based Real-Time Speech Enhancement for Dual-Microphone
            Mobile Phones; Tan et al., 2020
            https://web.cse.ohio-state.edu/~wang.77/papers/TZW.taslp21.pdf

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            input_channels (list): number of input channels for the stacked
                DenselyConnectedBlock layers
                Its length should be (`number of DenselyConnectedBlock layers`).
            enc_hid_channels (int): common number of intermediate channels for all
                DenselyConnectedBlock of the encoder
            enc_kernel_size (tuple): common kernel size for all DenselyConnectedBlock
                of the encoder
            enc_padding (tuple): common padding for all DenselyConnectedBlock
                of the encoder
            enc_last_kernel_size (tuple): common kernel size for the last Conv layer
                in all DenselyConnectedBlock of the encoder
            enc_last_stride (tuple): common stride for the last Conv layer in all
                DenselyConnectedBlock of the encoder
            enc_last_padding (tuple): common padding for the last Conv layer in all
                DenselyConnectedBlock of the encoder
            enc_layers (int): common total number of Conv layers for all
                DenselyConnectedBlock layers of the encoder
            skip_last_kernel_size (tuple): common kernel size for the last Conv layer
                in all DenselyConnectedBlock of the skip pathways
            skip_last_stride (tuple): common stride for the last Conv layer in all
                DenselyConnectedBlock of the skip pathways
            skip_last_padding (tuple): common padding for the last Conv layer in all
                DenselyConnectedBlock of the skip pathways
            glstm_groups (int): number of groups in each Grouped LSTM layer
            glstm_layers (int): number of Grouped LSTM layers
            glstm_bidirectional (bool): whether to use BLSTM or unidirectional LSTM
                in Grouped LSTM layers
            glstm_rearrange (bool): whether to apply the rearrange operation after each
                grouped LSTM layer
            output_channels (int): number of output channels (even number)
            mode (str): one of ("mapping", "masking")
                "mapping": complex spectral mapping
                "masking": complex masking
            ref_channel (int): index of the reference microphone
        """
        super().__init__()

        self._num_spk = num_spk
        self.mode = mode
        if mode not in ("mapping", "masking"):
            raise ValueError("mode=%s is not supported" % mode)
        self.ref_channel = ref_channel

        self.dc_crn = DC_CRN(
            input_dim=input_dim,
            input_channels=input_channels,
            enc_hid_channels=enc_hid_channels,
            enc_kernel_size=enc_kernel_size,
            enc_padding=enc_padding,
            enc_last_kernel_size=enc_last_kernel_size,
            enc_last_stride=enc_last_stride,
            enc_last_padding=enc_last_padding,
            enc_layers=enc_layers,
            skip_last_kernel_size=skip_last_kernel_size,
            skip_last_stride=skip_last_stride,
            skip_last_padding=skip_last_padding,
            glstm_groups=glstm_groups,
            glstm_layers=glstm_layers,
            glstm_bidirectional=glstm_bidirectional,
            glstm_rearrange=glstm_rearrange,
            output_channels=num_spk * 2,
        )

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """DC-CRN Separator Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [Batch, T, F]
                                                   or [Batch, T, C, F]
            ilens (torch.Tensor): input lengths [Batch,]

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(Batch, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        assert is_complex(input)
        is_multichannel = input.ndim == 4
        if is_multichannel:
            feature = torch.cat([input.real, input.imag], dim=2).permute(0, 2, 1, 3)
        else:
            feature = torch.stack([input.real, input.imag], dim=1)

        masks = self.dc_crn(feature)
        masks = [new_complex_like(input, m.unbind(dim=1)) for m in masks.unbind(dim=2)]

        if self.mode == "masking":
            if is_multichannel:
                masked = [input * m.unsqueeze(2) for m in masks]
            else:
                masked = [input * m for m in masks]
        else:
            masked = masks
            if is_multichannel:
                masks = [m.unsqueeze(2) / (input + EPS) for m in masked]
            else:
                masks = [m / (input + EPS) for m in masked]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
