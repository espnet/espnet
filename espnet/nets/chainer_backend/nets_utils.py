"""Chainer net_utils module."""

import chainer.functions as F


def _subsamplex(x, n):
    x = [F.get_item(xx, (slice(None, None, n), slice(None))) for xx in x]
    ilens = [xx.shape[0] for xx in x]
    return x, ilens
