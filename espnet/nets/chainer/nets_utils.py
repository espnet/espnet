import chainer.functions as F


def _subsamplex(x, n):
    x = [F.get_item(xx, (slice(None, None, n), slice(None))) for xx in x]
    ilens = [xx.shape[0] for xx in x]
    return x, ilens


# TODO(kan-bayashi): no need to use linear tensor
def linear_tensor(linear, x):
    """Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x ... x M matrix)
    :return:
    :param Variable y: Tensor (D_1 x D_2 x ... x N matrix)
    """
    y = linear(F.reshape(x, (-1, x.shape[-1])))
    return F.reshape(y, (x.shape[:-1] + (-1,)))
