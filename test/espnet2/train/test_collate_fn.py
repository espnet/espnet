import numpy as np
import pytest

from espnet2.train.collate_fn import CommonCollateFn, HuBERTCollateFn, common_collate_fn


@pytest.mark.parametrize(
    "float_pad_value, int_pad_value, not_sequence",
    [(0.0, -1, ()), (3.0, 2, ("a",)), (np.inf, 100, ("a", "b"))],
)
def test_common_collate_fn(float_pad_value, int_pad_value, not_sequence):
    data = [
        ("id", dict(a=np.random.randn(3, 5), b=np.random.randn(4).astype(np.long))),
        ("id2", dict(a=np.random.randn(2, 5), b=np.random.randn(3).astype(np.long))),
    ]
    t = common_collate_fn(
        data,
        float_pad_value=float_pad_value,
        int_pad_value=int_pad_value,
        not_sequence=not_sequence,
    )

    desired = dict(
        a=np.stack(
            [
                data[0][1]["a"],
                np.pad(
                    data[1][1]["a"],
                    [(0, 1), (0, 0)],
                    mode="constant",
                    constant_values=float_pad_value,
                ),
            ]
        ),
        b=np.stack(
            [
                data[0][1]["b"],
                np.pad(
                    data[1][1]["b"],
                    [(0, 1)],
                    mode="constant",
                    constant_values=int_pad_value,
                ),
            ]
        ),
        a_lengths=np.array([3, 2], dtype=np.long),
        b_lengths=np.array([4, 3], dtype=np.long),
    )

    np.testing.assert_array_equal(t[1]["a"], desired["a"])
    np.testing.assert_array_equal(t[1]["b"], desired["b"])

    if "a" not in not_sequence:
        np.testing.assert_array_equal(t[1]["a_lengths"], desired["a_lengths"])
    if "b" not in not_sequence:
        np.testing.assert_array_equal(t[1]["b_lengths"], desired["b_lengths"])


@pytest.mark.parametrize(
    "float_pad_value, int_pad_value, not_sequence",
    [(0.0, -1, ()), (3.0, 2, ("a",)), (np.inf, 100, ("a", "b"))],
)
def test_(float_pad_value, int_pad_value, not_sequence):
    _common_collate_fn = CommonCollateFn(
        float_pad_value=float_pad_value,
        int_pad_value=int_pad_value,
        not_sequence=not_sequence,
    )
    data = [
        ("id", dict(a=np.random.randn(3, 5), b=np.random.randn(4).astype(np.long))),
        ("id2", dict(a=np.random.randn(2, 5), b=np.random.randn(3).astype(np.long))),
    ]
    t = _common_collate_fn(data)

    desired = dict(
        a=np.stack(
            [
                data[0][1]["a"],
                np.pad(
                    data[1][1]["a"],
                    [(0, 1), (0, 0)],
                    mode="constant",
                    constant_values=float_pad_value,
                ),
            ]
        ),
        b=np.stack(
            [
                data[0][1]["b"],
                np.pad(
                    data[1][1]["b"],
                    [(0, 1)],
                    mode="constant",
                    constant_values=int_pad_value,
                ),
            ]
        ),
        a_lengths=np.array([3, 2], dtype=np.long),
        b_lengths=np.array([4, 3], dtype=np.long),
    )

    np.testing.assert_array_equal(t[1]["a"], desired["a"])
    np.testing.assert_array_equal(t[1]["b"], desired["b"])

    if "a" not in not_sequence:
        np.testing.assert_array_equal(t[1]["a_lengths"], desired["a_lengths"])
    if "b" not in not_sequence:
        np.testing.assert_array_equal(t[1]["b_lengths"], desired["b_lengths"])


@pytest.mark.parametrize(
    "float_pad_value, int_pad_value, not_sequence",
    [(0.0, -1, ()), (3.0, 2, ("a",)), (np.inf, 100, ("a", "b"))],
)
def test_CommonCollateFn_repr(float_pad_value, int_pad_value, not_sequence):
    print(
        CommonCollateFn(
            float_pad_value=float_pad_value,
            int_pad_value=int_pad_value,
            not_sequence=not_sequence,
        )
    )


@pytest.mark.parametrize(
    "float_pad_value, int_pad_value, not_sequence, label_downsampling, pad, rand_crop",
    [
        (0.0, -1, (), 1, True, False),
        (3.0, 2, ("a",), 1, False, False),
        (np.inf, 100, ("a", "b"), 2, True, False),
    ],
)
def test_HuBERT_(
    float_pad_value, int_pad_value, not_sequence, label_downsampling, pad, rand_crop
):
    _hubert_collate_fn = HuBERTCollateFn(
        float_pad_value=float_pad_value,
        int_pad_value=int_pad_value,
        not_sequence=not_sequence,
        label_downsampling=label_downsampling,
        pad=pad,
        rand_crop=rand_crop,
    )
    data = [
        (
            "id",
            dict(
                speech=np.random.randn(16000), text=np.random.randn(49).astype(np.long)
            ),
        ),
        (
            "id2",
            dict(
                speech=np.random.randn(22000), text=np.random.randn(67).astype(np.long)
            ),
        ),
    ]
    t = _hubert_collate_fn(data)

    if pad:
        desired = dict(
            speech=np.stack(
                [
                    np.pad(
                        data[0][1]["speech"],
                        (0, 6000),
                        mode="constant",
                        constant_values=float_pad_value,
                    ),
                    data[1][1]["speech"],
                ]
            ),
            text=np.stack(
                [
                    np.pad(
                        data[0][1]["text"],
                        (0, 18),
                        mode="constant",
                        constant_values=int_pad_value,
                    )[::label_downsampling],
                    data[1][1]["text"][::label_downsampling],
                ]
            ),
            speech_lengths=np.array([16000, 22000], dtype=np.long),
            text_lengths=np.array([49, 67], dtype=np.long),
        )
    else:
        desired = dict(
            speech=np.stack(
                [
                    data[0][1]["speech"],
                    data[1][1]["speech"][:16000],
                ]
            ),
            text=np.stack(
                [
                    data[0][1]["text"][::label_downsampling],
                    data[1][1]["text"][:49:label_downsampling],
                ]
            ),
            speech_lengths=np.array([16000, 16000], dtype=np.long),
            text_lengths=np.array([49, 49], dtype=np.long),
        )

    if label_downsampling > 1:
        desired["text_lengths"] = (
            desired["text_lengths"] + 1 - label_downsampling
        ) // label_downsampling + 1

    np.testing.assert_array_equal(t[1]["speech"], desired["speech"])
    np.testing.assert_array_equal(t[1]["text"], desired["text"])

    if "speech" not in not_sequence:
        np.testing.assert_array_equal(t[1]["speech_lengths"], desired["speech_lengths"])
    if "text" not in not_sequence:
        np.testing.assert_array_equal(t[1]["text_lengths"], desired["text_lengths"])


@pytest.mark.parametrize(
    "float_pad_value, int_pad_value, not_sequence, label_downsampling, pad, rand_crop",
    [
        (0.0, -1, (), 1, True, True),
        (3.0, 2, ("a",), 1, False, False),
        (np.inf, 100, ("a", "b"), 2, True, False),
    ],
)
def test_HuBERTCollateFn_repr(
    float_pad_value, int_pad_value, not_sequence, label_downsampling, pad, rand_crop
):
    print(
        HuBERTCollateFn(
            float_pad_value=float_pad_value,
            int_pad_value=int_pad_value,
            not_sequence=not_sequence,
            label_downsampling=label_downsampling,
            pad=pad,
            rand_crop=rand_crop,
        )
    )
