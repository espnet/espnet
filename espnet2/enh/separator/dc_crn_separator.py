from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex, new_complex_like
from espnet2.enh.layers.dc_crn import DC_CRN
from espnet2.enh.separator.abs_separator import AbsSeparator

EPS = torch.finfo(torch.get_default_dtype()).eps
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class DC_CRNSeparator(AbsSeparator):
    """
    Densely-Connected Convolutional Recurrent Network (DC-CRN) Separator.

    This class implements the DC-CRN model for speech separation based on the 
    paper: "Deep Learning Based Real-Time Speech Enhancement for Dual-Microphone 
    Mobile Phones" by Tan et al., 2020. The model can operate in two modes: 
    complex spectral mapping or complex masking.

    Reference:
        Tan, Z., Wang, D., & Chen, Y. (2020). Deep Learning Based Real-Time 
        Speech Enhancement for Dual-Microphone Mobile Phones. 
        https://web.cse.ohio-state.edu/~wang.77/papers/TZW.taslp21.pdf

    Args:
        input_dim (int): Input feature dimension.
        num_spk (int): Number of speakers (default: 2).
        predict_noise (bool): Whether to output the estimated noise signal 
            (default: False).
        input_channels (list): Number of input channels for the stacked 
            DenselyConnectedBlock layers. Its length should be equal to 
            the number of DenselyConnectedBlock layers (default: [2, 16, 32, 64, 128, 256]).
        enc_hid_channels (int): Common number of intermediate channels for all 
            DenselyConnectedBlock of the encoder (default: 8).
        enc_kernel_size (tuple): Common kernel size for all DenselyConnectedBlock 
            of the encoder (default: (1, 3)).
        enc_padding (tuple): Common padding for all DenselyConnectedBlock of 
            the encoder (default: (0, 1)).
        enc_last_kernel_size (tuple): Common kernel size for the last Conv layer 
            in all DenselyConnectedBlock of the encoder (default: (1, 4)).
        enc_last_stride (tuple): Common stride for the last Conv layer in all 
            DenselyConnectedBlock of the encoder (default: (1, 2)).
        enc_last_padding (tuple): Common padding for the last Conv layer in all 
            DenselyConnectedBlock of the encoder (default: (0, 1)).
        enc_layers (int): Common total number of Conv layers for all 
            DenselyConnectedBlock layers of the encoder (default: 5).
        skip_last_kernel_size (tuple): Common kernel size for the last Conv layer 
            in all DenselyConnectedBlock of the skip pathways (default: (1, 3)).
        skip_last_stride (tuple): Common stride for the last Conv layer in all 
            DenselyConnectedBlock of the skip pathways (default: (1, 1)).
        skip_last_padding (tuple): Common padding for the last Conv layer in all 
            DenselyConnectedBlock of the skip pathways (default: (0, 1)).
        glstm_groups (int): Number of groups in each Grouped LSTM layer (default: 2).
        glstm_layers (int): Number of Grouped LSTM layers (default: 2).
        glstm_bidirectional (bool): Whether to use BLSTM or unidirectional LSTM 
            in Grouped LSTM layers (default: False).
        glstm_rearrange (bool): Whether to apply the rearrange operation after each 
            grouped LSTM layer (default: False).
        mode (str): One of ("mapping", "masking"). "mapping" for complex spectral 
            mapping and "masking" for complex masking (default: "masking").
        ref_channel (int): Index of the reference microphone (default: 0).

    Raises:
        ValueError: If the provided mode is not supported.

    Examples:
        >>> separator = DC_CRNSeparator(input_dim=512, num_spk=2)
        >>> input_tensor = torch.randn(10, 20, 512)  # Batch of 10, 20 time frames
        >>> ilens = torch.tensor([20] * 10)  # All sequences are of length 20
        >>> masked, ilens, others = separator(input_tensor, ilens)

    Note:
        The output masks can be used for separating the sources based on the 
        chosen mode of operation (masking or mapping).
    """
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        predict_noise: bool = False,
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
            predict_noise: whether to output the estimated noise signal
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
        self.predict_noise = predict_noise
        self.mode = mode
        if mode not in ("mapping", "masking"):
            raise ValueError("mode=%s is not supported" % mode)
        self.ref_channel = ref_channel

        num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
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
            output_channels=num_outputs * 2,
        )

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
        DC-CRN Separator Forward.

    This method processes the input features through the DC-CRN architecture to 
    separate the signals from multiple speakers. It can operate in either 
    masking or mapping mode based on the specified configuration.

    Args:
        input (Union[torch.Tensor, ComplexTensor]): Encoded feature tensor of 
            shape [Batch, T, F] for real input or [Batch, T, C, F] for complex 
            input, where T is the time dimension, F is the frequency dimension, 
            and C is the number of channels.
        ilens (torch.Tensor): Input lengths of shape [Batch,].
        additional (Optional[Dict]): Additional data that can be provided for 
            processing, defaults to None.

    Returns:
        Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, 
              OrderedDict]: A tuple containing:
            - masked (List[Union[torch.Tensor, ComplexTensor]]): List of tensors 
              representing the masked output for each speaker, with shapes 
              [(Batch, T, F), ...].
            - ilens (torch.Tensor): Tensor of input lengths with shape (B,).
            - others (OrderedDict): Dictionary containing additional predicted 
              data such as masks for each speaker:
                - 'mask_spk1': torch.Tensor(Batch, Frames, Freq)
                - 'mask_spk2': torch.Tensor(Batch, Frames, Freq)
                - ...
                - 'mask_spkn': torch.Tensor(Batch, Frames, Freq)

    Examples:
        >>> separator = DC_CRNSeparator(input_dim=64, num_spk=2)
        >>> input_tensor = torch.randn(8, 100, 64)  # Example input
        >>> ilens = torch.tensor([100] * 8)  # Lengths of each input
        >>> masked, ilens, others = separator.forward(input_tensor, ilens)

    Note:
        Ensure that the input tensor has the correct shape based on whether 
        it is real or complex. The function checks if the input is complex 
        and processes it accordingly.

    Raises:
        ValueError: If the mode is not one of ("mapping", "masking").
        """
        assert is_complex(input)
        is_multichannel = input.ndim == 4
        if is_multichannel:
            feature = torch.cat([input.real, input.imag], dim=2).permute(0, 2, 1, 3)
        else:
            feature = torch.stack([input.real, input.imag], dim=1)

        masks = self.dc_crn(feature)
        masks = [new_complex_like(input, m.unbind(dim=1)) for m in masks.unbind(dim=2)]

        if self.predict_noise:
            *masks, mask_noise = masks

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
        if self.predict_noise:
            mask_noise = mask_noise.unsqueeze(2) if is_multichannel else mask_noise
            if self.mode == "masking":
                others["noise1"] = input * mask_noise
            else:
                others["noise1"] = mask_noise

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
