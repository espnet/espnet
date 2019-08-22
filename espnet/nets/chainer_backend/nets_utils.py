import chainer.functions as F
import logging


def _subsamplex(x, n):
    x = [F.get_item(xx, (slice(None, None, n), slice(None))) for xx in x]
    ilens = [xx.shape[0] for xx in x]
    return x, ilens


def mask_by_length(xs, lengths, fill=0):
    """Mask ndarray according to length.

    Args:
        xs (array): Batch of input ndarray (B, `*`).
        lengths (List): Batch of lengths (B,).
        fill (int or float): Value to fill masked part.

    Returns:
        Tensor: Batch of masked input ndarray (B, `*`).

    Examples:
        >>> x = xp.arange(5).repeat(3, 1) + 1
        >>> x
        ndarray([[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]])
        >>> lengths = [5, 3, 2]
        >>> mask_by_length(x, lengths)
        ndarray([[1, 2, 3, 4, 5],
                [1, 2, 3, 0, 0],
                [1, 2, 0, 0, 0]])

    """
    assert xs.shape[0] == len(lengths)
    xp = xs.xp
    ret = xp.full(xs.shape, fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l].data
    return ret