from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.rnn.encoders import RNN


class DPCLE2ESeparator(AbsSeparator):
    """
    Deep Clustering End-to-End Separator.

    This class implements a deep clustering approach for single-channel
    multi-speaker separation. The model utilizes a recurrent neural network
    (RNN) architecture to learn speaker-specific masks from the input audio
    features.

    References:
        Single-Channel Multi-Speaker Separation using Deep Clustering;
        Yusuf Isik et al., 2016;
        https://www.isca-speech.org/archive/interspeech_2016/isik16_interspeech.html

    Args:
        input_dim (int): Input feature dimension.
        rnn_type (str): Type of RNN to use. Options include 'blstm', 'lstm', etc.
        num_spk (int): Number of speakers in the input audio.
        predict_noise (bool): Whether to output the estimated noise signal.
        nonlinear (str): Nonlinear function for mask estimation.
            Options: 'relu', 'tanh', 'sigmoid'.
        layer (int): Number of stacked RNN layers. Default is 2.
        unit (int): Dimension of the hidden state.
        emb_D (int): Dimension of the feature vector for a tf-bin.
        dropout (float): Dropout ratio. Default is 0.0.
        alpha (float): Clustering hardness parameter.
        max_iteration (int): Maximum iterations for soft k-means.
        threshold (float): Threshold to end the soft k-means process.

    Returns:
        None.

    Examples:
        separator = DPCLE2ESeparator(input_dim=257, num_spk=2)
        input_features = torch.randn(10, 100, 257)  # (Batch, Time, Frequency)
        input_lengths = torch.tensor([100] * 10)  # Lengths of each input sequence
        masked_outputs, lengths, others = separator(input_features, input_lengths)

    Note:
        This separator is designed to work with both real and complex input
        tensors. Ensure the input features are properly formatted before
        passing them to the forward method.

    Raises:
        ValueError: If an unsupported nonlinear activation function is provided.
    """

    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "tanh",
        layer: int = 2,
        unit: int = 512,
        emb_D: int = 40,
        dropout: float = 0.0,
        alpha: float = 5.0,
        max_iteration: int = 500,
        threshold: float = 1.0e-05,
    ):
        """Deep Clustering End-to-End Separator

        References:
            Single-Channel Multi-Speaker Separation using Deep Clustering;
            Yusuf Isik. et al., 2016;
            https://www.isca-speech.org/archive/interspeech_2016/isik16_interspeech.html

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'blstm', 'lstm' etc.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            emb_D: int, dimension of the feature vector for a tf-bin.
            dropout: float, dropout ratio. Default is 0.
            alpha: float, the clustering hardness parameter.
            max_iteration: int, the max iterations of soft kmeans.
            threshold: float, the threshold to end the soft k-means process.
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

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

        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.enh_blstm = RNN(
            idim=input_dim * (self.num_outputs + 1),
            elayers=1,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        self.enh_linear = torch.nn.Linear(unit, input_dim * self.num_outputs)

        self.D = emb_D
        self.alpha = alpha
        self.max_iteration = max_iteration
        self.threshold = threshold

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
        Forward pass for the DPCLE2ESeparator.

        This method takes the encoded features and performs the forward pass
        through the model to separate the sources. It applies a soft K-means
        clustering algorithm to estimate the masks for each speaker.

        Args:
            input (Union[torch.Tensor, ComplexTensor]): Encoded feature tensor
                of shape [B, T, F] where B is batch size, T is time frames,
                and F is the number of frequency bins.
            ilens (torch.Tensor): Tensor of input lengths of shape [Batch].
            additional (Optional[Dict], optional): Additional information that
                can be passed to the forward method. Defaults to None.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor,
            OrderedDict]: A tuple containing:
                - masked (List[Union[torch.Tensor, ComplexTensor]]): List of
                  tensors, each of shape (B, T, F) representing the
                  separated sources [(B, T, N), ...].
                - ilens (torch.Tensor): Tensor containing the lengths of each
                  output in the batch.
                - others (OrderedDict): Contains additional predicted data,
                  e.g., masks for each speaker:
                    - 'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                    - 'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                    ...
                    - 'mask_spkn': torch.Tensor(Batch, Frames, Freq).

        Examples:
            >>> separator = DPCLE2ESeparator(input_dim=128)
            >>> input_tensor = torch.randn(10, 100, 128)  # Batch of 10
            >>> ilens = torch.tensor([100] * 10)  # All sequences of length 100
            >>> masked, ilens_out, others = separator.forward(input_tensor, ilens)

        Note:
            The output masks can be applied to the input features to obtain
            the estimated sources.

        Raises:
            ValueError: If the input is not a valid tensor or if ilens does
            not match the batch size.
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input
        B, T, F = input.shape

        # 1st Stage
        # x:(B, T, F)
        x, ilens, _ = self.blstm(feature, ilens)
        # x:(B, T, F*D)
        x = self.linear(x)
        # x:(B, T, F*D)
        x = self.nonlinear(x)
        V = x.view(B, -1, self.D)

        # Soft KMeans
        centers = V[:, : self.num_outputs, :]
        gamma = torch.zeros(B, T * F, self.num_outputs, device=input.device)
        count = 0
        while True:
            # Compute weight
            gamma_exp = torch.empty(B, T * F, self.num_outputs, device=input.device)
            new_centers = torch.empty(B, self.num_outputs, self.D, device=input.device)
            for i in range(self.num_outputs):
                gamma_exp[:, :, i] = torch.exp(
                    -self.alpha
                    * torch.sum(V - centers[:, i, :].unsqueeze(1) ** 2, dim=2)
                )
            # To avoid grad becomes nan, we add a small constant in denominator
            gamma = gamma_exp / (torch.sum(gamma_exp, dim=2, keepdim=True) + 1.0e-8)
            # Update centers
            for i in range(self.num_outputs):
                new_centers[:, i, :] = torch.sum(
                    V * gamma[:, :, i].unsqueeze(2), dim=1
                ) / (torch.sum(gamma[:, :, i].unsqueeze(2), dim=1) + 1.0e-8)

            if (
                torch.pow(new_centers - centers, 2).sum() < self.threshold
                or count > self.max_iteration
            ):
                break

            count += 1
            centers = new_centers

        masks = gamma.contiguous().view(B, T, F, self.num_outputs).unbind(dim=3)
        masked = [feature * m for m in masks]
        masked.append(feature)

        # 2nd Stage
        # cat_source:(B, T, (spks+1)*F)
        cat_source = torch.cat(masked, dim=2)
        # cat_x:(B, T, spks*F)
        cat_x, ilens, _ = self.enh_blstm(cat_source, ilens)
        # z:(B, T, spks*F)
        z = self.enh_linear(cat_x)
        z = z.contiguous().view(B, T, F, self.num_outputs)

        enh_masks = torch.softmax(z, dim=3).unbind(dim=3)
        if self.predict_noise:
            *enh_masks, mask_noise = enh_masks
        enh_masked = [input * m for m in enh_masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(enh_masks))], enh_masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return enh_masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
