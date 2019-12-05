import numpy as np
import pytest

from espnet2.train.collate_fn import common_collate_fn
from espnet2.train.collate_fn import CommonCollateFn


@pytest.mark.parametrize(
    'float_pad_value, int_pad_value, not_sequence',
    [(0., -1, ()),
     (3., 2, ('a',)),
     (np.inf, 100, ('a', 'b')),
     ])
def test_common_collate_fn(float_pad_value, int_pad_value, not_sequence):
    data = [dict(a=np.random.randn(3, 5),
                 b=np.random.randn(4).astype(np.long)),
            dict(a=np.random.randn(2, 5),
                 b=np.random.randn(3).astype(np.long))]
    t = common_collate_fn(data, float_pad_value=float_pad_value,
                          int_pad_value=int_pad_value,
                          not_sequence=not_sequence)

    desired = dict(a=np.stack([data[0]['a'],
                               np.pad(data[1]['a'], [(0, 1), (0, 0)],
                                      mode='constant',
                                      constant_values=float_pad_value
                                      )]),
                   b=np.stack([data[0]['b'],
                               np.pad(data[1]['b'], [(0, 1)],
                                      mode='constant',
                                      constant_values=int_pad_value)]),
                   a_lengths=np.array([3, 2], dtype=np.long),
                   b_lengths=np.array([4, 3], dtype=np.long),
                   )

    np.testing.assert_array_equal(t['a'], desired['a'])
    np.testing.assert_array_equal(t['b'], desired['b'])

    if 'a' not in not_sequence:
        np.testing.assert_array_equal(t['a_lengths'], desired['a_lengths'])
    if 'b' not in not_sequence:
        np.testing.assert_array_equal(t['b_lengths'], desired['b_lengths'])


@pytest.mark.parametrize(
    'float_pad_value, int_pad_value, not_sequence',
    [(0., -1, ()),
     (3., 2, ('a',)),
     (np.inf, 100, ('a', 'b')),
     ])
def test_(float_pad_value, int_pad_value, not_sequence):
    _common_collate_fn = CommonCollateFn(
        float_pad_value=float_pad_value, int_pad_value=int_pad_value,
        not_sequence=not_sequence)
    data = [dict(a=np.random.randn(3, 5),
                 b=np.random.randn(4).astype(np.long)),
            dict(a=np.random.randn(2, 5),
                 b=np.random.randn(3).astype(np.long))]
    t = _common_collate_fn(data)

    desired = dict(a=np.stack([data[0]['a'],
                               np.pad(data[1]['a'], [(0, 1), (0, 0)],
                                      mode='constant',
                                      constant_values=float_pad_value
                                      )]),
                   b=np.stack([data[0]['b'],
                               np.pad(data[1]['b'], [(0, 1)],
                                      mode='constant',
                                      constant_values=int_pad_value)]),
                   a_lengths=np.array([3, 2], dtype=np.long),
                   b_lengths=np.array([4, 3], dtype=np.long),
                   )

    np.testing.assert_array_equal(t['a'], desired['a'])
    np.testing.assert_array_equal(t['b'], desired['b'])

    if 'a' not in not_sequence:
        np.testing.assert_array_equal(t['a_lengths'], desired['a_lengths'])
    if 'b' not in not_sequence:
        np.testing.assert_array_equal(t['b_lengths'], desired['b_lengths'])


@pytest.mark.parametrize(
    'float_pad_value, int_pad_value, not_sequence',
    [(0., -1, ()),
     (3., 2, ('a',)),
     (np.inf, 100, ('a', 'b')),
     ])
def test_CommonCollateFn_repr(float_pad_value, int_pad_value, not_sequence):
    print(CommonCollateFn(float_pad_value=float_pad_value,
                          int_pad_value=int_pad_value,
                          not_sequence=not_sequence))
