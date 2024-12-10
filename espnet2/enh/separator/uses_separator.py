from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex, new_complex_like
from espnet2.enh.layers.uses import USES
from espnet2.enh.separator.abs_separator import AbsSeparator


class USESSeparator(AbsSeparator):
    """
    Unconstrained Speech Enhancement and Separation (USES) Network.

        This class implements the USES architecture for speech enhancement 
        and separation tasks. It is designed to handle various input conditions 
        and is capable of separating multiple speakers from a mixture.

        Reference:
            [1] W. Zhang, K. Saijo, Z.-Q. Wang, S. Watanabe, and Y. Qian,
            “Toward Universal Speech Enhancement for Diverse Input Conditions,”
            in Proc. ASRU, 2023.

        Args:
            input_dim (int): Input feature dimension. Not used as the model is 
                independent of the input size.
            num_spk (int): Number of speakers.
            enc_channels (int): Feature dimension after the Conv1D encoder.
            bottleneck_size (int): Dimension of the bottleneck feature. Must be a 
                multiple of `att_heads`.
            num_blocks (int): Number of processing blocks.
            num_spatial_blocks (int): Number of processing blocks with channel modeling.
            ref_channel (int): Reference channel (used in channel modeling modules).
            segment_size (int): Number of frames in each non-overlapping segment. 
                This is used to segment long utterances into smaller chunks for 
                efficient processing.
            memory_size (int): Group size of global memory tokens. The basic use 
                of memory tokens is to store the history information from previous 
                segments. The memory tokens are updated by the output of the last 
                block after processing each segment.
            memory_types (int): Number of memory token groups. Each group corresponds 
                to a different type of processing.
            rnn_type (str): Select from 'RNN', 'LSTM', and 'GRU'.
            bidirectional (bool): Whether the inter-chunk RNN layers are bidirectional.
            hidden_size (int): Dimension of the hidden state.
            att_heads (int): Number of attention heads.
            dropout (float): Dropout ratio. Default is 0.
            norm_type (str): Type of normalization to use after each inter- or 
                intra-chunk NN block.
            activation (str): The nonlinear activation function.
            ch_mode (Union[str, List[str]]): Mode of channel modeling. Select from 
                "att" and "tac".
            ch_att_dim (int): Dimension of the channel attention.
            eps (float): Epsilon for layer normalization.
            additional (dict): Additional parameters for flexibility during inference.

        Examples:
            >>> separator = USESSeparator(input_dim=80, num_spk=2)
            >>> input_tensor = torch.randn(10, 256, 80, 64)  # Example input
            >>> ilens = torch.tensor([64] * 10)  # Example input lengths
            >>> outputs, lengths, others = separator(input_tensor, ilens)
    """
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        enc_channels: int = 256,
        bottleneck_size: int = 64,
        num_blocks: int = 6,
        num_spatial_blocks: int = 3,
        ref_channel: Optional[int] = None,
        segment_size: int = 64,
        memory_size: int = 20,
        memory_types: int = 1,
        # Transformer-related arguments
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        hidden_size: int = 128,
        att_heads: int = 4,
        dropout: float = 0.0,
        norm_type: str = "cLN",
        activation: str = "relu",
        ch_mode: Union[str, List[str]] = "att",
        ch_att_dim: int = 256,
        eps: float = 1e-5,
        additional: dict = {},
    ):
        """Unconstrained Speech Enhancement and Separation (USES) Network.

        Reference:
            [1] W. Zhang, K. Saijo, Z.-Q., Wang, S. Watanabe, and Y. Qian,
            “Toward Universal Speech Enhancement for Diverse Input Conditions,”
            in Proc. ASRU, 2023.

        Args:
            input_dim (int): input feature dimension.
                Not used as the model is independent of the input size.
            num_spk (int): number of speakers.
            enc_channels (int): feature dimension after the Conv1D encoder.
            bottleneck_size (int): dimension of the bottleneck feature.
                Must be a multiple of `att_heads`.
            num_blocks (int): number of processing blocks.
            num_spatial_blocks (int): number of processing blocks with channel modeling.
            ref_channel (int): reference channel (used in channel modeling modules).
            segment_size (int): number of frames in each non-overlapping segment.
                This is used to segment long utterances into smaller chunks for
                efficient processing.
            memory_size (int): group size of global memory tokens.
                The basic use of memory tokens is to store the history information from
                previous segments.
                The memory tokens are updated by the output of the last block after
                processing each segment.
            memory_types (int): numbre of memory token groups.
                Each group corresponds to a different type of processing, i.e.,
                    the first group is used for denoising without dereverberation,
                    the second group is used for denoising with dereverberation,
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional (bool): whether the inter-chunk RNN layers are bidirectional.
            hidden_size (int): dimension of the hidden state.
            att_heads (int): number of attention heads.
            dropout (float): dropout ratio. Default is 0.
            norm_type: type of normalization to use after each inter- or
                intra-chunk NN block.
            activation: the nonlinear activation function.
            ch_mode: str or list, mode of channel modeling. Select from "att" and "tac".
            ch_att_dim (int): dimension of the channel attention.
            ref_channel: Optional[int], index of the reference channel.
            eps (float): epsilon for layer normalization.
        """
        super().__init__()

        self._num_spk = num_spk
        self.enc_channels = enc_channels
        self.ref_channel = ref_channel

        # used to project each complex-valued time-frequency bin to an embedding
        self.post_encoder = torch.nn.Conv2d(2, enc_channels, (3, 3), padding=(1, 1))

        assert bottleneck_size % att_heads == 0, (bottleneck_size, att_heads)
        opt = {
            "memory_types": memory_types,
        }
        # arguments in `opt` can be updated at inference time to process different data
        opt.update(additional)
        self.uses = USES(
            enc_channels,
            output_size=enc_channels * num_spk,
            bottleneck_size=bottleneck_size,
            num_blocks=num_blocks,
            num_spatial_blocks=num_spatial_blocks,
            segment_size=segment_size,
            memory_size=memory_size,
            **opt,
            # Transformer-specific arguments
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            hidden_size=hidden_size,
            att_heads=att_heads,
            dropout=dropout,
            norm_type=norm_type,
            activation=activation,
            ch_mode=ch_mode,
            ch_att_dim=ch_att_dim,
            eps=eps,
        )

        # used to project each embedding back to the complex-valued time-frequency bin
        self.pre_decoder = torch.nn.ConvTranspose2d(
            enc_channels, 2, (3, 3), padding=(1, 1)
        )

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
        Performs the forward pass of the USESSeparator model.

        This method processes the input STFT spectrum to separate the 
        sources and produce enhanced audio signals.

        Args:
            input (torch.Tensor or ComplexTensor): 
                STFT spectrum with shape [B, T, (C,) F (,2)], where:
                - B is the batch size,
                - T is the number of time frames,
                - C is the number of microphone channels (optional),
                - F is the number of frequency bins,
                - 2 corresponds to the real and imaginary parts (optional if 
                  input is a complex tensor).
            ilens (torch.Tensor): 
                Input lengths with shape [Batch].
            additional (Dict or None): 
                Additional data included in the model. It can contain:
                - "mode": one of ("no_dereverb", "dereverb", "both"):
                    1. "no_dereverb": Use only the first memory group for 
                       denoising without dereverberation.
                    2. "dereverb": Use only the second memory group for 
                       denoising with dereverberation.
                    3. "both": Use both memory groups for denoising with 
                       and without dereverberation.

        Returns:
            masked (List[Union[torch.Tensor, ComplexTensor]]): 
                List of tensors with shape [(B, T, F), ...] for each speaker.
            ilens (torch.Tensor): 
                The input lengths tensor with shape (B,).
            others (OrderedDict): 
                A dictionary containing predicted data, e.g., masks:
                OrderedDict[
                    'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                    'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                    ...
                    'mask_spkn': torch.Tensor(Batch, Frames, Freq),
                ]

        Examples:
            >>> separator = USESSeparator(input_dim=256)
            >>> input_tensor = torch.randn(8, 100, 2, 256)  # Example input
            >>> ilens = torch.tensor([100] * 8)  # Example input lengths
            >>> outputs, lengths, additional_outputs = separator.forward(input_tensor, ilens)
            >>> print(len(outputs))  # Number of separated sources

        Raises:
            ValueError: If the input shape is invalid or if an unknown mode is 
            provided in `additional`.
        """
        # B, 2, T, (C,) F
        if is_complex(input):
            feature = torch.stack([input.real, input.imag], dim=1)
        else:
            assert input.size(-1) == 2, input.shape
            feature = input.moveaxis(-1, 1)

        # B, C, 2, F, T
        if feature.ndim == 4:
            feature = feature.moveaxis(-1, -2).unsqueeze(1)
        elif feature.ndim == 5:
            feature = feature.permute(0, 3, 1, 4, 2).contiguous()
        else:
            raise ValueError(f"Invalid input shape: {feature.shape}")

        B, C, RI, F, T = feature.shape
        feature = feature.reshape(-1, RI, F, T)
        feature = self.post_encoder(feature)  # B*C, enc_channels, F, T
        feature = feature.reshape(B, C, -1, F, T).contiguous()

        others = {}
        # B, enc_channels * num_spk, F, T
        if additional is not None:
            mode = additional.get("mode", "no_dereverb")
            if mode == "no_dereverb":
                processed = self.uses(feature, ref_channel=self.ref_channel)
            elif mode == "dereverb":
                processed = self.uses(feature, ref_channel=self.ref_channel, mem_idx=1)
            elif mode == "both":
                # For training with multii-condition data
                # 1. denoised output without dereverberation
                processed = self.uses(feature, ref_channel=self.ref_channel, mem_idx=0)

                # 2. denoised output with dereverberation
                processed2 = self.uses(feature, ref_channel=self.ref_channel, mem_idx=1)
                processed2 = processed2.reshape(
                    B * self.num_spk, self.enc_channels, F, T
                )
                processed2 = self.pre_decoder(processed2)
                specs2 = processed2.reshape(B, self.num_spk, 2, F, T).moveaxis(-1, -2)
                # B, num_spk, T, F
                if not is_complex(input):
                    for spk in range(specs2.size(1)):
                        others[f"dereverb{spk + 1}"] = ComplexTensor(
                            specs2[:, spk, 0], specs2[:, spk, 1]
                        )
                else:
                    for spk in range(specs2.size(1)):
                        others[f"dereverb{spk + 1}"] = new_complex_like(
                            input, (specs2[:, spk, 0], specs2[:, spk, 1])
                        )
            else:
                raise ValueError(mode)
        else:
            mode = ""
            processed = self.uses(feature, ref_channel=self.ref_channel)

        processed = processed.reshape(B * self.num_spk, self.enc_channels, F, T)
        processed = self.pre_decoder(processed)
        specs = processed.reshape(B, self.num_spk, 2, F, T).moveaxis(-1, -2)

        # B, num_spk, T, F
        if not is_complex(input):
            specs = list(ComplexTensor(specs[:, :, 0], specs[:, :, 1]).unbind(1))
        else:
            specs = list(
                new_complex_like(input, (specs[:, :, 0], specs[:, :, 1])).unbind(1)
            )

        return specs, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
