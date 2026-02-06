"""Add_deltas module."""

import numpy as np


def delta(feat, window):
    """Process the deta of the feats."""
    assert window > 0
    delta_feat = np.zeros_like(feat)
    for i in range(1, window + 1):
        delta_feat[:-i] += i * feat[i:]
        delta_feat[i:] += -i * feat[:-i]
        delta_feat[-i:] += i * feat[-1]
        delta_feat[:i] += -i * feat[0]
    delta_feat /= 2 * sum(i**2 for i in range(1, window + 1))
    return delta_feat


def add_deltas(x, window=2, order=2):
    """Append the deltas to the input."""
    feats = [x]
    for _ in range(order):
        feats.append(delta(feats[-1], window))
    return np.concatenate(feats, axis=1)


class AddDeltas(object):
    """Add Deltas class."""

    def __init__(self, window=2, order=2):
        """Initialize the class."""
        self.window = window
        self.order = order

    def __repr__(self):
        """Return a printable representation of the class."""
        return "{name}(window={window}, order={order}".format(
            name=self.__class__.__name__, window=self.window, order=self.order
        )

    def __call__(self, x):
        """Process the call method."""
        return add_deltas(x, window=self.window, order=self.order)
