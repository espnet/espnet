# encoding: utf-8
"""Class Declaration of Transformer's Label Smootion loss."""

import chainer.links as L


class LayerNorm(L.LayerNormalization):
    def __init__(self, dims, eps=1e-12):
        super(LayerNorm, self).__init__(size=dims, eps=eps)

    def __call__(self, e):
        return super(LayerNorm, self).__call__(e)
