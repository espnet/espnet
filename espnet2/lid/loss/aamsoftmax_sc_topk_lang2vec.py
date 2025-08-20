import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class AAMSoftmaxSCTopKLang2Vec(AbsLoss):
    r"""
    AAMSoftmax with intertopk and subcenter, and lang2vec prediction.

    The AAMSoftmax part is same with ArcMarginProduct_intertopk_subcenter
    in `aamsoftmax_subcenter_intertopk.py`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        cos(theta + margin)
        K: number of sub-centers
        k_top: number of hard samples
        mp: margin penalty of hard samples
        do_lm: whether do large margin finetune

        NOTE(qingzheng):
        There are no place in ESPnet been found use the update function
        to update the cos_mp and sin_mp, so the the cos_mp are always
        cos(0.0) and sin_mp are sin(0.0). So, actually, the marginize on
        the negative classes in topK is not used.
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
        lang2vec_dim: int = None,
        lang2vec_type: str = None,  # geo, phonology_knn, syntax_knn, inventory_knn
        lang2vec_weight: float = None,
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

        self.lang2vec_dim = lang2vec_dim
        self.lang2vec_type = lang2vec_type
        self.lang2vec_weight = lang2vec_weight

        if (
            self.lang2vec_dim is not None
            and self.lang2vec_type is not None
            and self.lang2vec_weight is not None
        ):
            if lang2vec_type == "geo":
                self.lang2vec_head = nn.Sequential(
                    nn.Linear(nout, lang2vec_dim),
                )
                self.lang2vec_loss = nn.MSELoss()
            elif lang2vec_type in ["phonology_knn", "syntax_knn", "inventory_knn"]:
                self.lang2vec_head = nn.Sequential(
                    nn.Linear(nout, lang2vec_dim),
                )
                self.lang2vec_loss = (
                    nn.BCEWithLogitsLoss()
                )  # BCEWithLogitsLoss combines sigmoid and binary cross entropy, which use the log-sum-exp trick for numerical stability.
            else:
                raise ValueError(
                    f"Unknown lang2vec type: {lang2vec_type},"
                    "support lang2vec types: geo, phonology_knn,"
                    "syntax_knn, inventory_knn"
                )

    def update(self, margin=0.2):
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

    def _calc_subcenter_cosine(self, input, weight=None):
        """
        Calculate the cosine similarity between input and K subcenters.
        If the weight is not provided (None), it will use the default
        weight (in __init__). The weight input here is for the self-conditioning
        LID on the intermediate layers, which may need to use independent
        weights. If weight is not provided here and use self-conditioning
        LID, it means the weight is shared across all intermediate layers
        and the downstream AAMSoftmax.
        Args:
            input: (batch, in_features)
            weight: (K * nclasses, in_features)
        Returns:
            cosine: (batch, nclasses)
        """
        # there are k subcenter per class in weight
        if weight is not None:
            cosine = F.linear(
                F.normalize(input), F.normalize(weight)
            )  # (batch, nclasses * k)
        else:
            cosine = F.linear(
                F.normalize(input), F.normalize(self.weight)
            )  # (batch, nclasses * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K)
        )  # (batch, nclasses, k)

        # subcenter max pooling, compute k max cosine, use the max one
        # k subcenter per class in weight, cosine is the simlarity to these
        # subcenters then select the top one. This is for intra-class
        # marginization.
        cosine, _ = torch.max(cosine, dim=2)  # (batch, nclasses)

        return cosine

    def _add_margin(self, cosine, label):
        """
        Add margin to the cosine similarity. Only add margin during training.
        Args:
            cosine: (batch, nclasses)
            label: (batch,)
        Returns:
            output: (batch, nclasses)
        """
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # NOTE(qingzheng): \ means drop trend, / means increase trend
        # phi: cos(theta + m), true class, +m, /, cos(+m) \,
        # -logcos(+m) /, original goal is \, hence is the penalty
        phi = cosine * self.cos_m - sine * self.sin_m

        # NOTE(qingzheng): phi_mp: for the topk samples, negative class
        # cos(theta - mp), -mp, \, cos(-mp) /, -logcos(-mp) \,
        # original goal is / (for negative class, we want max loss),
        # hence is the penalty
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp  # cos(theta - mp)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.k_top > 0:
            # topk (j != y_i), the top k expect the true class
            _, top_k_index = torch.topk(
                cosine - 2 * one_hot, self.k_top
            )  # exclude j = y_i
            top_k_one_hot = cosine.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

            # sum
            output = (
                (one_hot * phi)  # true class, phi
                + (top_k_one_hot * phi_mp)  # topk class (negative), phi_mp
                + (
                    (1.0 - one_hot - top_k_one_hot) * cosine
                )  # other class, cosine, without margin
            )
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.scale

        return output

    def forward(self, input, label=None, lang2vec=None, weight=None):
        """
        weight should be provided if use self-conditioning LID and apply
        independent weights across all intermediate layers.
        """

        cosine = self._calc_subcenter_cosine(input, weight)

        pred_lids = torch.argmax(cosine, dim=1)  # (batch,)

        if label is not None:
            if len(label.size()) == 2:
                label = label.squeeze(1)
            accuracy = (pred_lids == label).float().mean()
        else:  # inference
            loss = None
            accuracy = None
            return loss, accuracy, pred_lids

        output = self._add_margin(cosine, label)

        loss = self.ce(output, label)
        class_loss = loss  # classification loss
        lang2vec_loss = None

        if (
            lang2vec is not None
            and self.lang2vec_dim is not None
            and self.lang2vec_type is not None
            and self.lang2vec_weight is not None
        ):
            assert (
                0 < self.lang2vec_weight < 1
            ), f"lang2vec_weight should be in (0, 1), but got {self.lang2vec_weight}"
            lang2vec_loss = self.lang2vec_loss(self.lang2vec_head(input), lang2vec)
            loss *= 1 - self.lang2vec_weight
            loss += self.lang2vec_weight * lang2vec_loss

        return loss, accuracy, pred_lids, class_loss, lang2vec_loss
