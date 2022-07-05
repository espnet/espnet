from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.rnn.encoders import RNN


class DPCLSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        num_spk: int = 2,
        nonlinear: str = "tanh",
        layer: int = 2,
        unit: int = 512,
        emb_D: int = 40,
        dropout: float = 0.0,
    ):
        """Deep Clustering Separator.

        References:
            [1] Deep clustering: Discriminative embeddings for segmentation and
                separation; John R. Hershey. et al., 2016;
                https://ieeexplore.ieee.org/document/7471631
            [2] Manifold-Aware Deep Clustering: Maximizing Angles Between Embedding
                Vectors Based on Regular Simplex; Tanaka, K. et al., 2021;
                https://www.isca-speech.org/archive/interspeech_2021/tanaka21_interspeech.html

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'blstm', 'lstm' etc.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            emb_D: int, dimension of the feature vector for a tf-bin.
            dropout: float, dropout ratio. Default is 0.
        """  # noqa: E501
        super().__init__()

        self._num_spk = num_spk

        self.blstm = RNN(
            idim=input_dim,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        self.linear = torch.nn.Linear(unit, input_dim * emb_D)

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

        self.D = emb_D

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. tf_embedding: OrderedDict[
                'tf_embedding': learned embedding of all T-F bins (B, T * F, D),
            ]
        """
        # if complex spectrum,
        if isinstance(input, ComplexTensor):
            feature = abs(input)
        else:
            feature = input
        B, T, F = input.shape
        # x:(B, T, F)
        x, ilens, _ = self.blstm(feature, ilens)
        # x:(B, T, F*D)
        x = self.linear(x)
        # x:(B, T, F*D)
        x = self.nonlinear(x)
        tf_embedding = x.view(B, -1, self.D)

        if self.training:
            masked = None
        else:
            # K-means for batch
            centers = tf_embedding[:, : self._num_spk, :].detach()
            dist = torch.empty(B, T * F, self._num_spk, device=tf_embedding.device)
            last_label = torch.zeros(B, T * F, device=tf_embedding.device)
            while True:
                for i in range(self._num_spk):
                    dist[:, :, i] = torch.sum(
                        (tf_embedding - centers[:, i, :].unsqueeze(1)) ** 2, dim=2
                    )
                label = dist.argmin(dim=2)
                if torch.sum(label != last_label) == 0:
                    break
                last_label = label
                for b in range(B):
                    for i in range(self._num_spk):
                        centers[b, i] = tf_embedding[b, label[b] == i].mean(dim=0)
            label = label.view(B, T, F)
            masked = []
            for i in range(self._num_spk):
                masked.append(input * (label == i))

        others = OrderedDict(
            {"tf_embedding": tf_embedding},
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
