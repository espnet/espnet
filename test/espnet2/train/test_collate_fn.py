import numpy as np
from espnet2.train.collate_fn import common_collate_fn


def test_common_collate_fn():
    data = [dict(a=np.random.randn(3, 5),
                 b=np.random.randn(4).astype(np.long)),
            dict(a=np.random.randn(2, 5),
                 b=np.random.randn(4).astype(np.long))]
    t = common_collate_fn(data)

    desired = dict(a=np.stack([data[0]['a'],
                               np.pad(data[1]['a'], [(0, 1), (0, 0)],
                                      mode='constant')]),
                   b=np.stack([data[0]['b'], data[1]['b']]),
                   a_lengths=np.array([3, 2], dtype=np.long),
                   b_lengths=np.array([4, 4], dtype=np.long),
                   )

    np.testing.assert_array_equal(t['a'], desired['a'])
    np.testing.assert_array_equal(t['a_lengths'], desired['a_lengths'])
    np.testing.assert_array_equal(t['b'], desired['b'])
    np.testing.assert_array_equal(t['b_lengths'], desired['b_lengths'])
