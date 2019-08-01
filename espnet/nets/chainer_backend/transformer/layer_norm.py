# encoding: utf-8

import chainer.links as L


class LayerNorm(L.LayerNormalization):
    def __init__(self, dims, eps=1e-12):
        super(LayerNorm, self).__init__(size=dims, eps=eps)

    def forward(self, e):
        return super(LayerNorm, self).forward(e)
