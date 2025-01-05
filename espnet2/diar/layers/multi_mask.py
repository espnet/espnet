# This is an implementation of the multiple 1x1 convolution layer architecture
# in https://arxiv.org/pdf/2203.17068.pdf

from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.diar.layers.abs_mask import AbsMask


class MultiMask(AbsMask):
    """
    Multiple 1x1 convolution layer Module.

    This module corresponds to the final 1x1 convolution block and
    non-linear function in TCNSeparator. It has multiple 1x1
    convolution blocks, one of which is selected according to the
    specified number of speakers to handle a flexible number of
    speakers.

    Args:
        input_dim (int): Number of filters in the autoencoder.
        bottleneck_dim (int, optional): Number of channels in the
            bottleneck 1x1 convolution block. Defaults to 128.
        max_num_spk (int, optional): Maximum number of
            mask_conv1x1 modules (should be >= maximum number of
            speakers in the dataset). Defaults to 3.
        mask_nonlinear (str, optional): Non-linear function to use
            for generating masks. Defaults to "relu".

    Attributes:
        max_num_spk (int): The maximum number of speakers supported
            by the model.

    Examples:
        >>> model = MultiMask(input_dim=256, bottleneck_dim=128,
        ...                    max_num_spk=3, mask_nonlinear='relu')
        >>> input_tensor = torch.randn(10, 64, 256)  # (M, K, N)
        >>> ilens = torch.tensor([64] * 10)  # Lengths for each input
        >>> bottleneck_feat = torch.randn(10, 64, 128)  # (M, K, B)
        >>> masked, ilens_out, others = model(input_tensor, ilens,
        ...                                     bottleneck_feat, num_spk=2)

    Raises:
        ValueError: If an unsupported mask non-linear function is
            specified.

    Returns:
        Tuple[List[Union[torch.Tensor, ComplexTensor]],
                torch.Tensor, OrderedDict]:
            - masked (List[Union[torch.Tensor, ComplexTensor]]):
                List of masked outputs for each speaker.
            - ilens (torch.Tensor): Lengths of the input sequences.
            - others (OrderedDict): Additional predicted data,
                including masks for each speaker.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 128,
        max_num_spk: int = 3,
        mask_nonlinear="relu",
    ):
        """Multiple 1x1 convolution layer Module.

        This module corresponds to the final 1x1 conv block and
        non-linear function in TCNSeparator.
        This module has multiple 1x1 conv blocks. One of them is selected
        according to the given num_spk to handle flexible num_spk.

        Args:
            input_dim: Number of filters in autoencoder
            bottleneck_dim: Number of channels in bottleneck 1 * 1-conv block
            max_num_spk: Number of mask_conv1x1 modules
                        (>= Max number of speakers in the dataset)
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self._max_num_spk = max_num_spk
        self.mask_nonlinear = mask_nonlinear
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.ModuleList()
        for z in range(1, max_num_spk + 1):
            self.mask_conv1x1.append(
                nn.Conv1d(bottleneck_dim, z * input_dim, 1, bias=False)
            )

    @property
    def max_num_spk(self) -> int:
        """
        Multiple 1x1 convolution layer Module.

        This module corresponds to the final 1x1 conv block and
        non-linear function in TCNSeparator. It consists of multiple
        1x1 convolution blocks, allowing for flexible handling of a
        varying number of speakers.

        Attributes:
            max_num_spk (int): Maximum number of speakers that can be
                processed by this module.
            mask_nonlinear (str): The non-linear activation function used
                to generate the masks.

        Args:
            input_dim (int): Number of filters in the autoencoder.
            bottleneck_dim (int, optional): Number of channels in the
                bottleneck 1x1 convolution block. Defaults to 128.
            max_num_spk (int, optional): Maximum number of
                mask_conv1x1 modules (>= Max number of speakers in the
                dataset). Defaults to 3.
            mask_nonlinear (str, optional): Non-linear function to use
                for mask generation. Defaults to "relu".

        Examples:
            >>> multi_mask = MultiMask(input_dim=64, bottleneck_dim=128,
            ...                          max_num_spk=3, mask_nonlinear="relu")
            >>> input_tensor = torch.randn(10, 128, 64)  # [M, K, N]
            >>> ilens = torch.tensor([64] * 10)  # lengths of the inputs
            >>> bottleneck_feat = torch.randn(10, 64, 128)  # [M, K, B]
            >>> masked, ilens, others = multi_mask(input_tensor, ilens,
            ...                                      bottleneck_feat, num_spk=2)

        Raises:
            ValueError: If an unsupported mask non-linear function is
                specified.

        Note:
            This module is part of the MultiMask package and is intended
            for use in speech separation tasks, particularly in scenarios
            involving multiple speakers.
        """
        return self._max_num_spk

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        bottleneck_feat: torch.Tensor,
        num_spk: int,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
        Processes input through the multiple 1x1 convolution layers.

        This method applies the forward pass for the MultiMask module, which
        consists of multiple 1x1 convolution layers that generate masks for
        separating audio signals from different speakers. The number of masks
        generated corresponds to the specified number of speakers.

        Args:
            input: A tensor of shape [M, K, N], where M is the batch size,
                   K is the number of frequency bins, and N is the number
                   of time steps.
            ilens (torch.Tensor): A tensor of shape (M,) containing the
                                  lengths of the input sequences.
            bottleneck_feat: A tensor of shape [M, K, B], representing the
                             bottleneck features, where B is the bottleneck
                             dimension.
            num_spk: An integer indicating the number of speakers
                     (training: oracle, inference: estimated by another module).

        Returns:
            masked (List[Union[torch.Tensor, ComplexTensor]]): A list of tensors
            of shape [(M, K, N), ...], where each tensor corresponds to the
            input masked by the estimated speaker masks.
            ilens (torch.Tensor): A tensor of shape (M,) containing the
                                  lengths of the input sequences.
            others (OrderedDict): An ordered dictionary containing additional
            predicted data, such as masks for each speaker:
                - 'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                - 'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                - 'mask_spkn': torch.Tensor(Batch, Frames, Freq).

        Raises:
            ValueError: If the specified non-linear function for mask
                        generation is unsupported.

        Examples:
            >>> input_tensor = torch.randn(2, 64, 128)  # Example input
            >>> ilens = torch.tensor([128, 128])  # Example input lengths
            >>> bottleneck_feat = torch.randn(2, 64, 128)  # Example bottleneck
            >>> num_spk = 2  # Number of speakers
            >>> masked_output, lengths, masks = multi_mask.forward(
            ...     input_tensor, ilens, bottleneck_feat, num_spk)

        Note:
            This API is designed to be compatible with the TasNet framework.
        """
        M, K, N = input.size()
        bottleneck_feat = bottleneck_feat.transpose(1, 2)  # [M, B, K]
        score = self.mask_conv1x1[num_spk - 1](
            bottleneck_feat
        )  # [M, B, K] -> [M, num_spk*N, K]
        # add other outputs of the module list with factor 0.0
        # to enable distributed training
        for z in range(self._max_num_spk):
            if z != num_spk - 1:
                score += 0.0 * F.interpolate(
                    self.mask_conv1x1[z](bottleneck_feat).transpose(1, 2),
                    size=num_spk * N,
                ).transpose(1, 2)
        score = score.view(M, num_spk, N, K)  # [M, num_spk*N, K] -> [M, num_spk, N, K]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = torch.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = torch.tanh(score)
        else:
            raise ValueError("Unsupported mask non-linear function")

        masks = est_mask.transpose(2, 3)  # [M, num_spk, K, N]
        masks = masks.unbind(dim=1)  # List[M, K, N]

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others
