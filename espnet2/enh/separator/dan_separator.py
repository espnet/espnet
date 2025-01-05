from collections import OrderedDict
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as Fun
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.rnn.encoders import RNN


class DANSeparator(AbsSeparator):
    """
        DANSeparator is a Deep Attractor Network for single-microphone speaker
    separation, which utilizes recurrent neural networks to estimate masks for
    each speaker in an audio mixture.

    This model is based on the research paper:
    - DEEP ATTRACTOR NETWORK FOR SINGLE-MICROPHONE SPEAKER SEPARATION;
      Zhuo Chen et al., 2017;
      https://pubmed.ncbi.nlm.nih.gov/29430212/

    Attributes:
        num_spk (int): The number of speakers to separate.

    Args:
        input_dim (int): Input feature dimension.
        rnn_type (str): Type of RNN, options include 'blstm', 'lstm', etc.
        num_spk (int): Number of speakers to separate. Default is 2.
        nonlinear (str): Nonlinear function for mask estimation. Options
                         include 'relu', 'tanh', 'sigmoid'. Default is 'tanh'.
        layer (int): Number of stacked RNN layers. Default is 2.
        unit (int): Dimension of the hidden state. Default is 512.
        emb_D (int): Dimension of the attribute vector for one time-frequency bin.
                     Default is 40.
        dropout (float): Dropout ratio. Default is 0.0.

    Raises:
        ValueError: If the nonlinear function is not one of the supported types.

    Examples:
        separator = DANSeparator(input_dim=80, num_spk=2, rnn_type='blstm')
        input_tensor = torch.randn(10, 100, 80)  # Batch of 10, 100 time steps, 80 features
        ilens = torch.tensor([100] * 10)  # Input lengths for each batch
        masked, ilens_out, others = separator(input_tensor, ilens)

        # masked will contain the separated signals for each speaker
        # ilens_out will contain the lengths of the output signals
        # others will contain the predicted masks for each speaker
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
        """Deep Attractor Network Separator

        Reference:
            DEEP ATTRACTOR NETWORK FOR SINGLE-MICROPHONE SPEAKER SEPARATION;
            Zhuo Chen. et al., 2017;
            https://pubmed.ncbi.nlm.nih.gov/29430212/

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'blstm', 'lstm' etc.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            emb_D: int, dimension of the attribute vector for one tf-bin.
            dropout: float, dropout ratio. Default is 0.
        """
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
                Forward pass for the Deep Attractor Network (DAN) separator.

        This method processes the input audio features and computes the estimated
        masks for each speaker. It utilizes a recurrent neural network (RNN) to
        generate embeddings, from which attractors are derived to separate the
        speakers' contributions in the mixed signal.

        Args:
            input (Union[torch.Tensor, ComplexTensor]):
                Encoded feature tensor of shape [B, T, F], where B is the batch
                size, T is the number of time frames, and F is the number of
                frequency bins.
            ilens (torch.Tensor):
                A tensor containing the lengths of the input sequences for each
                batch element, shape [Batch].
            additional (Optional[Dict], optional):
                A dictionary containing additional data that may be used in the
                model. For example, it may include:
                - "feature_ref": List of reference spectra of shape
                  List[(B, T, F)].

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor,
                  OrderedDict]:
                - masked (List[Union[torch.Tensor, ComplexTensor]]):
                  A list of tensors, each of shape [(B, T, N), ...], where N
                  corresponds to the number of speakers.
                - ilens (torch.Tensor):
                  A tensor of shape (B,) representing the lengths of the input
                  sequences.
                - others (OrderedDict):
                  A dictionary containing predicted data, such as masks for each
                  speaker, with the following structure:
                  OrderedDict[
                      'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                      'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                      ...
                      'mask_spkn': torch.Tensor(Batch, Frames, Freq),
                  ]

        Examples:
            >>> model = DANSeparator(input_dim=256)
            >>> input_tensor = torch.randn(10, 100, 256)  # Example input
            >>> ilens = torch.tensor([100] * 10)  # All sequences are of length 100
            >>> masked, ilens_out, others = model.forward(input_tensor, ilens)

        Note:
            Ensure that the input feature tensor and ilens are properly
            aligned and of correct dimensions.

        Raises:
            ValueError: If the input nonlinear activation function is not one of
            'sigmoid', 'relu', or 'tanh'.
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
        # tf_embedding:(B, T*F, D)
        tf_embedding = x.contiguous().view(B, T * F, -1)

        # Compute the attractors
        if self.training:
            assert additional is not None and "feature_ref" in additional
            origin = additional["feature_ref"]
            abs_origin = [abs(o) for o in origin]
            Y_t = torch.zeros(B, T, F, device=origin[0].device)
            for i in range(self._num_spk):
                flags = [abs_origin[i] >= o for o in abs_origin]
                Y = reduce(lambda x, y: x * y, flags)
                Y = Y.int() * i
                Y_t += Y
            Y_t = Y_t.contiguous().flatten().long()
            Y = Fun.one_hot(Y_t, num_classes=self._num_spk)
            Y = Y.contiguous().view(B, -1, self._num_spk).float()

            # v_y:(B, D, spks)
            v_y = torch.bmm(torch.transpose(tf_embedding, 1, 2), Y)
            # sum_y:(B, D, spks)
            sum_y = torch.sum(Y, 1, keepdim=True).expand_as(v_y)
            # attractor:(B, D, spks)
            attractor = v_y / (sum_y + 1e-8)
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
            attractor = centers.permute(0, 2, 1)

        # calculate the distance between embeddings and attractors
        # dist:(B, T*F, spks)
        dist = torch.bmm(tf_embedding, attractor)
        masks = torch.softmax(dist, dim=2)
        masks = masks.contiguous().view(B, T, F, self._num_spk).unbind(dim=3)

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
