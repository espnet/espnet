#! /usr/bin/python
# -*- encoding: utf-8 -*-
# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class AAMSoftmax(AbsLoss):
    """
        Additive Angular Margin Softmax (AAMSoftmax) for deep face recognition.

    This class implements the AAMSoftmax loss function, which incorporates an
    additive angular margin to enhance the discriminative power of deep face
    recognition systems. The concept is based on the paper by Deng et al. (2019)
    entitled "ArcFace: Additive Angular Margin Loss for Deep Face Recognition."

    Attributes:
        test_normalize (bool): Indicates whether to normalize during testing.
        m (float): Margin value for AAMSoftmax.
        s (float): Scale value for AAMSoftmax.
        in_feats (int): Dimensionality of speaker embedding.
        weight (torch.nn.Parameter): Learnable parameter for class weights.
        ce (nn.CrossEntropyLoss): Cross-entropy loss function.
        easy_margin (bool): Flag to use easy margin or standard margin.
        cos_m (float): Cosine of the margin.
        sin_m (float): Sine of the margin.
        th (float): Threshold for cosine values.
        mm (float): Modified margin value for non-monotonic cases.

    Args:
        nout (int): Dimensionality of speaker embedding.
        nclasses (int): Number of speakers in the training set.
        margin (float, optional): Margin value of AAMSoftmax (default: 0.3).
        scale (float, optional): Scale value of AAMSoftmax (default: 15).
        easy_margin (bool, optional): If True, use easy margin (default: False).

    Returns:
        torch.Tensor: The computed loss value.

    Examples:
        >>> aamsoftmax = AAMSoftmax(nout=512, nclasses=10)
        >>> x = torch.randn(32, 512)  # Batch of 32 embeddings
        >>> labels = torch.randint(0, 10, (32,))  # Random labels for 10 classes
        >>> loss = aamsoftmax(x, labels)
        >>> print(loss)

    Note:
        This implementation is adapted from the original code from the
        VoxCeleb trainer and Face_Pytorch repository.

    Todo:
        - Add more tests for edge cases and performance.
    """

    def __init__(
        self, nout, nclasses, margin=0.3, scale=15, easy_margin=False, **kwargs
    ):
        super().__init__(nout)

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nout
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(nclasses, nout), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print("Initialised AAMSoftmax margin %.3f scale %.3f" % (self.m, self.s))

    def forward(self, x, label=None):
        """
        Computes the forward pass of the Additive Angular Margin Softmax (AAMSoftmax).

        This method calculates the loss based on the input features and their
        corresponding labels. It applies the AAMSoftmax transformation to the
        input embeddings, incorporating the additive angular margin for improved
        discriminative power in face recognition tasks.

        Args:
            x (torch.Tensor): Input features of shape (batch_size, nout),
                where `nout` is the dimensionality of the speaker embedding.
            label (torch.Tensor, optional): Ground truth labels of shape
                (batch_size,) or (batch_size, 1). Defaults to None.

        Returns:
            torch.Tensor: The computed loss value.

        Raises:
            AssertionError: If the size of `label` does not match the first
            dimension of `x` or if `x` does not have the expected number of
            features.

        Examples:
            >>> aamsoftmax = AAMSoftmax(nout=512, nclasses=10)
            >>> features = torch.randn(32, 512)  # Batch of 32 samples
            >>> labels = torch.randint(0, 10, (32,))  # Random labels for 10 classes
            >>> loss = aamsoftmax(features, labels)
            >>> print(loss)

        Note:
            The method normalizes the input features and weights before
            calculating cosine similarities. The `label` tensor should contain
            class indices corresponding to the input features.

        Todo:
            Consider adding support for multi-label classification in the
            future versions.
        """
        if len(label.size()) == 2:
            label = label.squeeze(1)

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        return loss
