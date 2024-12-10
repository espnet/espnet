from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.bsrnn import BSRNN
from espnet2.enh.layers.complex_utils import is_complex, new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator


class BSRNNSeparator(AbsSeparator):
    """
    Band-split RNN (BSRNN) separator for speech enhancement.

    This class implements a BSRNN-based speech separator designed to enhance 
    the quality of audio signals by separating different speakers. It leverages 
    a band-split architecture to effectively process audio signals.

    References:
        [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
        [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

    Attributes:
        num_spk (int): Number of speakers being separated.
        ref_channel (Optional[int]): Reference channel (currently unused).

    Args:
        input_dim (int): Maximum number of frequency bins corresponding to 
            `target_fs`.
        num_spk (int): Number of speakers. Default is 1.
        num_channels (int): Feature dimension in the BandSplit block. Default is 16.
        num_layers (int): Number of processing layers. Default is 6.
        target_fs (int): Max sampling frequency that the model can handle. 
            Default is 48000.
        causal (bool): Whether to apply causal modeling. If True, LSTM will be 
            used instead of BLSTM for time modeling. Default is True.
        norm_type (str): Type of the normalization layer (cfLN / cLN / BN / GN). 
            Default is "GN".
        ref_channel (Optional[int]): Reference channel. Not used for now.

    Examples:
        >>> separator = BSRNNSeparator(input_dim=1024, num_spk=2)
        >>> input_tensor = torch.randn(1, 100, 2, 512)  # Example STFT input
        >>> ilens = torch.tensor([100])
        >>> masked, ilens, others = separator(input_tensor, ilens)

    Raises:
        AssertionError: If the input tensor does not have the expected dimensions 
        when not using complex input.
    """
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
        """
        Perform the forward pass of the BSRNN separator.

        This method processes the input STFT spectrum and generates masks for 
        separating different speakers using the BSRNN model.

        Args:
            input (torch.Tensor or ComplexTensor): 
                The input STFT spectrum with shape [B, T, (C,) F (,2)],
                where B is the batch size, T is the number of time frames,
                C is the number of channels, F is the number of frequency bins,
                and the last dimension of size 2 represents real and imaginary 
                parts if using a ComplexTensor.
            ilens (torch.Tensor): 
                A tensor containing the input lengths for each sequence in the 
                batch, with shape [Batch].
            additional (Dict or None): 
                A dictionary containing other data that may be included in the 
                model. This parameter is unused in this implementation.

        Returns:
            masked (List[Union[torch.Tensor, ComplexTensor]]): 
                A list of tensors representing the separated signals, 
                each with shape [(B, T, F), ...] for each speaker.
            ilens (torch.Tensor): 
                The input lengths tensor with shape (B,).
            others (OrderedDict): 
                A dictionary containing additional predicted data, such as 
                masks for each speaker. The structure is as follows:
                OrderedDict[
                    'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                    'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                    ...
                    'mask_spkn': torch.Tensor(Batch, Frames, Freq),
                ]

        Examples:
            >>> input_tensor = torch.randn(10, 100, 2)  # Example input
            >>> ilens_tensor = torch.tensor([100] * 10)  # Example input lengths
            >>> masks, lengths, additional_outputs = separator.forward(input_tensor, ilens_tensor)

        Note:
            The `additional` argument is not utilized in this version of the 
            model, but it is included for compatibility with potential future 
            extensions.

        Raises:
            AssertionError: If the input tensor does not have the correct shape 
            or if it is not complex and does not have a last dimension of size 2.
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
