# Copyright 2025 Qingzheng Wang
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Add lang2vec supervision to ESPnet's existing subcenter AAMSoftmax loss."""

import torch.nn as nn

from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)


class AAMSoftmaxSCTopKLang2Vec(ArcMarginProduct_intertopk_subcenter):
    """Use the standard classification loss plus an auxiliary lang2vec head."""

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
        lang2vec_type: str = None,
        lang2vec_weight: float = None,
    ):
        super().__init__(
            nout=nout,
            nclasses=nclasses,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
            K=K,
            mp=mp,
            k_top=k_top,
            do_lm=do_lm,
        )
        self.lang2vec_dim = lang2vec_dim
        self.lang2vec_type = lang2vec_type
        self.lang2vec_weight = lang2vec_weight
        if all(
            value is not None
            for value in (lang2vec_dim, lang2vec_type, lang2vec_weight)
        ):
            if not 0 <= lang2vec_weight <= 1:
                raise ValueError("lang2vec_weight must be in [0, 1].")
            self.lang2vec_head = nn.Sequential(nn.Linear(nout, lang2vec_dim))
            if lang2vec_type == "geo":
                self.lang2vec_loss = nn.MSELoss()
            elif lang2vec_type in ("phonology_knn", "syntax_knn", "inventory_knn"):
                self.lang2vec_loss = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unknown lang2vec type: {lang2vec_type}")

    def forward(self, input, label=None, lang2vec=None):
        """Return total loss, accuracy, predictions, and separate loss components."""
        class_loss, accuracy, predictions = super().forward(input, label)
        loss = class_loss
        auxiliary = None
        if (
            label is not None
            and lang2vec is not None
            and hasattr(self, "lang2vec_head")
        ):
            auxiliary = self.lang2vec_loss(self.lang2vec_head(input), lang2vec)
            loss = (
                1 - self.lang2vec_weight
            ) * class_loss + self.lang2vec_weight * auxiliary
        return loss, accuracy, predictions, class_loss, auxiliary
