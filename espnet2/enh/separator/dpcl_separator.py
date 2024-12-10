from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.rnn.encoders import RNN


class DPCLSeparator(AbsSeparator):
    """
    Deep Clustering Separator.

    This class implements the Deep Clustering method for source separation,
    which utilizes recurrent neural networks (RNNs) to estimate masks for 
    separating audio sources. It is particularly effective for scenarios with 
    multiple speakers.

    References:
        [1] Deep clustering: Discriminative embeddings for segmentation and
            separation; John R. Hershey. et al., 2016;
            https://ieeexplore.ieee.org/document/7471631
        [2] Manifold-Aware Deep Clustering: Maximizing Angles Between Embedding
            Vectors Based on Regular Simplex; Tanaka, K. et al., 2021;
            https://www.isca-speech.org/archive/interspeech_2021/tanaka21_interspeech.html

    Args:
        input_dim (int): Input feature dimension.
        rnn_type (str, optional): RNN type, select from 'blstm', 'lstm', etc.
            Defaults to 'blstm'.
        num_spk (int, optional): Number of speakers. Defaults to 2.
        nonlinear (str, optional): Nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid'. Defaults to 'tanh'.
        layer (int, optional): Number of stacked RNN layers. Defaults to 2.
        unit (int, optional): Dimension of the hidden state. Defaults to 512.
        emb_D (int, optional): Dimension of the feature vector for a tf-bin.
            Defaults to 40.
        dropout (float, optional): Dropout ratio. Defaults to 0.0.

    Raises:
        ValueError: If the nonlinear function is not one of the supported types.

    Examples:
        >>> separator = DPCLSeparator(input_dim=80, num_spk=2)
        >>> input_tensor = torch.randn(10, 100, 80)  # Batch of 10, 100 time steps
        >>> ilens = torch.tensor([100] * 10)  # All sequences have length 100
        >>> masked, ilens_out, others = separator(input_tensor, ilens)

    Attributes:
        num_spk (int): The number of speakers that the separator is configured for.

    Note:
        The `additional` argument in the forward method is not used in this model.
    """
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
        """
        Forward pass for the DPCLSeparator.

        This method processes the input features through the model and generates 
        separated outputs for each speaker. It employs a recurrent neural network 
        to encode the input features and applies K-means clustering to estimate 
        the speaker masks.

        Args:
            input (Union[torch.Tensor, ComplexTensor]): 
                Encoded feature of shape [B, T, F], where B is the batch size, 
                T is the number of time frames, and F is the number of frequency 
                bins. This can be either a standard tensor or a complex tensor.
            ilens (torch.Tensor): 
                Input lengths of shape [Batch], indicating the actual lengths 
                of the sequences in the batch.
            additional (Optional[Dict]): 
                Other data included in the model. This argument is currently 
                not used in this model.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, 
                  OrderedDict]:
                - masked (List[Union[torch.Tensor, ComplexTensor]]): A list 
                  of tensors of shape [(B, T, N), ...] representing the 
                  separated audio signals for each speaker.
                - ilens (torch.Tensor): Tensor of shape (B,) containing the 
                  lengths of the input sequences after processing.
                - others (OrderedDict): Contains additional predicted data, 
                  such as:
                    - 'tf_embedding': learned embedding of all T-F bins 
                      of shape (B, T * F, D).

        Examples:
            >>> separator = DPCLSeparator(input_dim=80, num_spk=2)
            >>> input_tensor = torch.randn(4, 100, 80)  # Example input
            >>> input_lengths = torch.tensor([100, 100, 80, 60])  # Lengths
            >>> masked, lengths, others = separator.forward(input_tensor, 
            ...                                             input_lengths)

        Note:
            This method currently does not utilize the 'additional' argument.

        Raises:
            ValueError: If the input 'nonlinear' activation function is not 
                        one of 'sigmoid', 'relu', or 'tanh'.
        """
        # if complex spectrum,
        if is_complex(input):
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
