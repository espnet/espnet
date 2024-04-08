from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import TemporalConvNet, TemporalConvNetInformed
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class TDSpeakerBeamExtractor(AbsExtractor):
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
        """TD-SpeakerBeam Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            input_aux (torch.Tensor or ComplexTensor): Encoded auxiliary feature
                for the target speaker [B, T, N] or [B, N]
            ilens_aux (torch.Tensor): input lengths of auxiliary input for the
                target speaker [Batch]
            suffix_tag (str): suffix to append to the keys in `others`
            additional (None or dict): additional parameters
                not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                f'mask{suffix_tag}': torch.Tensor(Batch, Frames, Freq),
                f'enroll_emb{suffix_tag}': torch.Tensor(Batch, adapt_enroll_dim/adapt_enroll_dim*2),
            ]
        """  # noqa: E501
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
