from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import TemporalConvNet, TemporalConvNetInformed
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class TDSpeakerBeamExtractor(AbsExtractor):
    """
    Time-Domain SpeakerBeam Extractor.

    This class implements a time-domain speaker beam extractor that utilizes a
    Temporal Convolutional Network (TCN) for separating audio signals from
    different speakers. It supports both speaker embeddings and enrollment audio
    for speaker adaptation.

    Attributes:
        use_spk_emb (bool): Flag indicating whether to use speaker embeddings.

    Args:
        input_dim (int): Input feature dimension.
        layer (int, optional): Number of layers in each stack (default: 8).
        stack (int, optional): Number of stacks (default: 3).
        bottleneck_dim (int, optional): Bottleneck dimension (default: 128).
        hidden_dim (int, optional): Number of convolution channels (default: 512).
        skip_dim (int, optional): Number of skip connection channels (default: 128).
        kernel (int, optional): Kernel size (default: 3).
        causal (bool, optional): Whether to use causal convolution (default: False).
        norm_type (str, optional): Normalization type; choose from 'BN', 'gLN', 'cLN'
            (default: 'gLN').
        pre_nonlinear (str, optional): Nonlinear function before mask estimation;
            select from 'prelu', 'relu', 'tanh', 'sigmoid', 'linear' (default: 'prelu').
        nonlinear (str, optional): Nonlinear function for mask estimation;
            select from 'relu', 'tanh', 'sigmoid', 'linear' (default: 'relu').
        i_adapt_layer (int, optional): Index of adaptation layer (default: 7).
        adapt_layer_type (str, optional): Type of adaptation layer; see
            espnet2.enh.layers.adapt_layers for options (default: 'mul').
        adapt_enroll_dim (int, optional): Dimensionality of the speaker embedding
            (default: 128).
        use_spk_emb (bool, optional): Whether to use speaker embeddings as enrollment
            (default: False).
        spk_emb_dim (int, optional): Dimension of input speaker embeddings; only
            used when `use_spk_emb` is True (default: 256).

    Raises:
        ValueError: If `pre_nonlinear` or `nonlinear` are not supported values.

    Examples:
        # Creating an instance of TDSpeakerBeamExtractor
        extractor = TDSpeakerBeamExtractor(
            input_dim=128,
            layer=8,
            stack=3,
            bottleneck_dim=128,
            hidden_dim=512,
            skip_dim=128,
            kernel=3,
            causal=False,
            norm_type='gLN',
            pre_nonlinear='prelu',
            nonlinear='relu',
            i_adapt_layer=7,
            adapt_layer_type='mul',
            adapt_enroll_dim=128,
            use_spk_emb=False,
            spk_emb_dim=256,
        )

        # Forward pass
        masked_output, ilens, others = extractor(
            input=torch.randn(16, 100, 128),  # Example input
            ilens=torch.tensor([100] * 16),    # Input lengths
            input_aux=torch.randn(16, 1, 128),  # Auxiliary input
            ilens_aux=torch.tensor([1] * 16),   # Auxiliary input lengths
            suffix_tag='',
            additional=None,
        )
    """

    def __init__(
        self,
        input_dim: int,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        skip_dim: int = 128,
        kernel: int = 3,
        causal: bool = False,
        norm_type: str = "gLN",
        pre_nonlinear: str = "prelu",
        nonlinear: str = "relu",
        # enrollment related arguments
        i_adapt_layer: int = 7,
        adapt_layer_type: str = "mul",
        adapt_enroll_dim: int = 128,
        use_spk_emb: bool = False,
        spk_emb_dim: int = 256,
    ):
        """Time-Domain SpeakerBeam Extractor.

        Args:
            input_dim: input feature dimension
            layer: int, number of layers in each stack
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            skip_dim: int, number of skip connection channels
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
            pre_nonlinear: the nonlinear function right before mask estimation
                       select from 'prelu', 'relu', 'tanh', 'sigmoid', 'linear'
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid', 'linear'
            i_adapt_layer: int, index of adaptation layer
            adapt_layer_type: str, type of adaptation layer
                see espnet2.enh.layers.adapt_layers for options
            adapt_enroll_dim: int, dimensionality of the speaker embedding
            use_spk_emb: bool, whether to use speaker embeddings as enrollment
            spk_emb_dim: int, dimension of input speaker embeddings
                         only used when `use_spk_emb` is True
        """
        super().__init__()

        if pre_nonlinear not in ("sigmoid", "prelu", "relu", "tanh", "linear"):
            raise ValueError("Not supporting pre_nonlinear={}".format(pre_nonlinear))
        if nonlinear not in ("sigmoid", "relu", "tanh", "linear"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.tcn = TemporalConvNetInformed(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            Sc=skip_dim,
            out_channel=None,
            norm_type=norm_type,
            causal=causal,
            pre_mask_nonlinear=pre_nonlinear,
            mask_nonlinear=nonlinear,
            i_adapt_layer=i_adapt_layer,
            adapt_layer_type=adapt_layer_type,
            adapt_enroll_dim=adapt_enroll_dim,
        )

        # Auxiliary network
        self.use_spk_emb = use_spk_emb
        if use_spk_emb:
            self.auxiliary_net = torch.nn.Conv1d(
                spk_emb_dim,
                adapt_enroll_dim if skip_dim is None else adapt_enroll_dim * 2,
                1,
            )
        else:
            self.auxiliary_net = TemporalConvNet(
                N=input_dim,
                B=bottleneck_dim,
                H=hidden_dim,
                P=kernel,
                X=layer,
                R=1,
                C=1,
                Sc=skip_dim,
                out_channel=(
                    adapt_enroll_dim if skip_dim is None else adapt_enroll_dim * 2
                ),
                norm_type=norm_type,
                causal=False,
                pre_mask_nonlinear=pre_nonlinear,
                mask_nonlinear="linear",
            )

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        input_aux: torch.Tensor,
        ilens_aux: torch.Tensor,
        suffix_tag: str = "",
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
        TD-SpeakerBeam Forward.

        This method processes the input features through the Time-Domain SpeakerBeam
        extractor and returns the masked output, input lengths, and additional
        predicted data such as masks and enrollment embeddings.

        Args:
            input (torch.Tensor or ComplexTensor):
                Encoded feature of shape [B, T, N], where B is the batch size,
                T is the time dimension, and N is the feature dimension.
            ilens (torch.Tensor):
                Input lengths of shape [Batch] indicating the valid length of
                each input feature in the batch.
            input_aux (torch.Tensor or ComplexTensor):
                Encoded auxiliary feature for the target speaker of shape
                [B, T, N] or [B, N]. This can be either a speaker embedding or
                enrollment audio.
            ilens_aux (torch.Tensor):
                Input lengths of auxiliary input for the target speaker of shape
                [Batch].
            suffix_tag (str, optional):
                Suffix to append to the keys in the `others` dictionary.
                Defaults to an empty string.
            additional (None or dict, optional):
                Additional parameters not used in this model.
                Defaults to None.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
                A tuple containing:
                - masked (List[Union(torch.Tensor, ComplexTensor)]):
                    A list of masked tensors of shape [(B, T, N), ...].
                - ilens (torch.Tensor):
                    The input lengths of shape (B,).
                - others (OrderedDict):
                    A dictionary containing predicted data, e.g., masks:
                    - f'mask{suffix_tag}': torch.Tensor(Batch, Frames, Freq),
                    - f'enroll_emb{suffix_tag}':
                      torch.Tensor(Batch, adapt_enroll_dim/adapt_enroll_dim*2).

        Note:
            When `self.use_spk_emb` is True, `aux_feature` is assumed to be
            a speaker embedding; otherwise, it is assumed to be an enrollment audio.

        Examples:
            >>> extractor = TDSpeakerBeamExtractor(input_dim=256)
            >>> input_tensor = torch.randn(10, 100, 256)  # [B, T, N]
            >>> ilens = torch.tensor([100] * 10)  # input lengths
            >>> input_aux = torch.randn(10, 1, 256)  # Auxiliary input
            >>> ilens_aux = torch.tensor([1] * 10)  # Auxiliary input lengths
            >>> masked_output, lengths, additional_outputs = extractor.forward(
            ...     input_tensor, ilens, input_aux, ilens_aux
            ... )
        """
        # if complex spectrum
        feature = abs(input) if is_complex(input) else input
        aux_feature = abs(input_aux) if is_complex(input_aux) else input_aux
        B, L, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, L
        # NOTE(wangyou): When `self.use_spk_emb` is True, `aux_feature` is assumed to be
        # a speaker embedding; otherwise, it is assumed to be an enrollment audio.
        if self.use_spk_emb:
            # B, N, L'=1
            if aux_feature.dim() == 2:
                aux_feature = aux_feature.unsqueeze(-1)
            elif aux_feature.size(-2) == 1:
                assert aux_feature.dim() == 3, aux_feature.shape
                aux_feature = aux_feature.transpose(1, 2)
        else:
            aux_feature = aux_feature.transpose(1, 2)  # B, N, L'

        enroll_emb = self.auxiliary_net(aux_feature).squeeze(1)  # B, N', L'
        if not self.use_spk_emb:
            enroll_emb.masked_fill_(make_pad_mask(ilens_aux, enroll_emb, -1), 0.0)
        enroll_emb = enroll_emb.mean(dim=-1)  # B, N'

        mask = self.tcn(feature, enroll_emb)  # B, N, L
        mask = mask.transpose(-1, -2)  # B, L, N

        masked = input * mask

        others = {
            "enroll_emb{}".format(suffix_tag): enroll_emb.detach(),
        }

        return masked, ilens, others
