# code from WeSpeaker: https://github.com/wenet-e2e/wespeaker/blob/
# c9ec537b53fe1e04525be74b2550ee95bed3a891/wespeaker/models/projections.py#L243

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class ArcMarginProduct_intertopk_subcenter(AbsLoss):
    """
    Implement of large margin arc distance with intertopk and subcenter.

    This class implements the ArcMarginProduct with intertopk and subcenter
    techniques for improved speaker verification. It leverages concepts from
    the referenced papers to enhance the model's performance.

    References:
        MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
        FOR SPEAKER VERIFICATION. https://arxiv.org/pdf/2110.05042.pdf
        Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy
        Web Faces. https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        scale (float, optional): Norm of input feature. Defaults to 32.0.
        margin (float, optional): Margin for cos(theta + margin). Defaults to 0.2.
        easy_margin (bool, optional): Use easy margin if True. Defaults to False.
        K (int, optional): Number of sub-centers. Defaults to 3.
        mp (float, optional): Margin penalty of hard samples. Defaults to 0.06.
        k_top (int, optional): Number of hard samples. Defaults to 5.
        do_lm (bool, optional): Whether to perform large margin finetune.
                                 Defaults to False.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output classes.
        scale (float): Scaling factor for output.
        margin (float): Margin value.
        K (int): Number of sub-centers.
        k_top (int): Number of top-k samples.
        mp (float): Margin penalty for hard samples.
        do_lm (bool): Flag for large margin finetuning.
        weight (torch.Tensor): Weight parameter for the classifier.
        easy_margin (bool): Flag for easy margin setting.
        cos_m (float): Cosine of the margin.
        sin_m (float): Sine of the margin.
        th (float): Threshold for cosine value.
        mm (float): Margin adjustment factor.
        mmm (float): Additional margin adjustment factor.
        cos_mp (float): Cosine for margin penalty.
        sin_mp (float): Sine for margin penalty.
        ce (nn.CrossEntropyLoss): Cross-entropy loss function.

    Examples:
        >>> arc_margin = ArcMarginProduct_intertopk_subcenter(
        ...     nout=512, nclasses=10, scale=32.0, margin=0.2, K=3, k_top=5
        ... )
        >>> input_tensor = torch.randn(16, 512)  # Batch of 16 samples
        >>> label_tensor = torch.randint(0, 10, (16, 1))  # Random labels
        >>> loss = arc_margin.forward(input_tensor, label_tensor)

    Note:
        Ensure that the input tensor is normalized before passing it to the
        forward method to achieve optimal results.

    Raises:
        ValueError: If the input and label sizes do not match.

    Todo:
        - Add support for additional loss functions.
        - Optimize the implementation for larger batch sizes.
    """

    def __init__(
        self,
        nout,
        nclasses,
        scale=32.0,
        margin=0.2,
        easy_margin=False,
        K=3,
        mp=0.06,
        k_top=5,
        do_lm=False,
    ):
        super().__init__(nout)
        self.in_features = nout
        self.out_features = nclasses
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm

        # intertopk + subcenter
        self.K = K
        if do_lm:  # if do LMF, remove hard sample penalty
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top

        # initial classifier
        self.weight = nn.Parameter(torch.FloatTensor(self.K * nclasses, nout))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin
        )  # this can make the output more continuous
        ########
        self.m = self.margin
        ########
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

        self.ce = nn.CrossEntropyLoss()

    def update(self, margin=0.2):
        """
            Implement of large margin arc distance with intertopk and subcenter.

        This class implements the ArcMarginProduct with enhancements for speaker
        verification tasks. It includes intertopk and subcenter techniques to improve
        the robustness of the margin-based loss function.

        Reference:
            MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
            FOR SPEAKER VERIFICATION.
            https://arxiv.org/pdf/2110.05042.pdf
            Sub-center ArcFace: Boosting Face Recognition by
            Large-Scale Noisy Web Faces.
            https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

        Attributes:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            scale (float): Norm of input feature.
            margin (float): Margin for the arc margin calculation.
            K (int): Number of sub-centers.
            k_top (int): Number of hard samples.
            mp (float): Margin penalty of hard samples.
            do_lm (bool): Whether to perform large margin finetune.

        Args:
            nout (int): Number of output features.
            nclasses (int): Number of classes for classification.
            scale (float, optional): Norm of input feature. Defaults to 32.0.
            margin (float, optional): Margin for the arc margin calculation.
                Defaults to 0.2.
            easy_margin (bool, optional): If True, use easy margin. Defaults to False.
            K (int, optional): Number of sub-centers. Defaults to 3.
            mp (float, optional): Margin penalty of hard samples. Defaults to 0.06.
            k_top (int, optional): Number of hard samples. Defaults to 5.
            do_lm (bool, optional): If True, performs large margin finetune.
                Defaults to False.

        Examples:
            arc_margin = ArcMarginProduct_intertopk_subcenter(nout=512, nclasses=10)
            arc_margin.update(margin=0.3)
        """
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # hard sample margin is increasing as margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, input, label):
        """
        Computes the forward pass for the ArcMarginProduct.

        This method applies the ArcMarginProduct calculation to the input
        features and computes the loss based on the provided labels. It
        normalizes the input features and weight parameters, calculates
        the cosine and sine values for the margin, and applies the
        necessary transformations to generate the output.

        Args:
            input (torch.Tensor): The input features of shape (batch_size,
                in_features).
            label (torch.Tensor): The ground truth labels of shape
                (batch_size, num_classes) or (batch_size, 1).

        Returns:
            torch.Tensor: The computed loss value.

        Raises:
            ValueError: If the input tensor does not match the expected
                dimensions.

        Examples:
            >>> model = ArcMarginProduct_intertopk_subcenter(nout=512, nclasses=10)
            >>> input = torch.randn(32, 512)  # Example input for batch size 32
            >>> label = torch.randint(0, 10, (32,))  # Random labels for 10 classes
            >>> loss = model(input, label)
            >>> print(loss)

        Note:
            Ensure that the input tensor is normalized before passing it
            to this method for optimal performance.

        Todo:
            - Add support for additional margin configurations in future
                implementations.
        """
        if len(label.size()) == 2:
            label = label.squeeze(1)
        cosine = F.linear(
            F.normalize(input), F.normalize(self.weight)
        )  # (batch, out_dim * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K)
        )  # (batch, out_dim, k)
        cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.k_top > 0:
            # topk (j != y_i)
            _, top_k_index = torch.topk(
                cosine - 2 * one_hot, self.k_top
            )  # exclude j = y_i
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

            # sum
            output = (
                (one_hot * phi)
                + (top_k_one_hot * phi_mp)
                + ((1.0 - one_hot - top_k_one_hot) * cosine)
            )
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.ce(output, label)
        return loss
