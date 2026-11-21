import re

import numpy as np
import pytest

from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
from espnet2.train.collate_fn import CommonCollateFn


class Dataset:
    def __init__(self):
        self.data = {
            "a": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            "b": np.array([8, 9, 10, 11, 12]),
        }

    def __getitem__(self, item):
        return item, {"dummy": self.data["a"]}


@pytest.mark.parametrize("chunk_length", [3, "2,4", "3-5"])
def test_ChunkIterFactory(chunk_length):
    dataset = Dataset()
    collatefn = CommonCollateFn()
    batches = [["a"], ["b"]]
    iter_factory = ChunkIterFactory(
        dataset=dataset,
        batches=batches,
        batch_size=2,
        chunk_length=chunk_length,
        collate_fn=collatefn,
    )

    for key, batch in iter_factory.build_iter(0):
        for k, v in batch.items():
            if chunk_length == 3:
                assert v.shape == (2, 3)
            elif chunk_length == "2,4":
                assert v.ndim == 2 and v.shape[0] == 2 and v.shape[1] in (2, 4)
            elif chunk_length == "3-5":
                assert v.ndim == 2 and v.shape[0] == 2 and v.shape[1] in (3, 4, 5)


class Dataset2:
    def __init__(self):
        self.data = {
            "a": {
                "label": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                "aux": np.array([0, -1, -2, -3]),
            },
            "b": {"label": np.array([8, 9, 10, 11, 12]), "aux": np.array([-1, -2])},
        }

    def __getitem__(self, item):
        return item, self.data[item]


def test_ChunkIterFactory_partial_chunking():
    dataset = Dataset2()
    collatefn = CommonCollateFn()
    batches = [["a"], ["b"]]
    iter_factory = ChunkIterFactory(
        dataset=dataset,
        batches=batches,
        batch_size=2,
        chunk_length=3,
        collate_fn=collatefn,
        excluded_key_prefixes=["aux"],
    )

    for key, batch in iter_factory.build_iter(0):
        for k, v in batch.items():
            if iter_factory.excluded_key_pattern is not None and re.fullmatch(
                iter_factory.excluded_key_pattern, k
            ):
                assert v.shape in ((2, 2), (2, 4))
            else:
                assert v.shape == (2, 3)


class Dataset3:
    def __init__(self, with_category=True):
        self.data = {
            "a": {"data": np.array([0, 1, 2, 3]), "utt2category": np.array([0])},
            "b": {"data": np.array([8, 9, 10, 11, 12]), "utt2category": np.array([1])},
            "c": {"data": np.array([0, 3, 5, 9, 7, 4]), "utt2category": np.array([2])},
            "d": {"data": np.array([1, 9, 3, 1, 2, 6]), "utt2category": np.array([1])},
            "e": {"data": np.array([10, 12, 20, 18, 3]), "utt2category": np.array([2])},
            "f": {"data": np.array([4, 5, 6, 7, 8, 9]), "utt2category": np.array([0])},
        }
        if not with_category:
            for v in self.data.values():
                del v["utt2category"]

    def __getitem__(self, item):
        return item, self.data[item]


def test_ChunkIterFactory_utt2category():
    dataset = Dataset3(with_category=True)
    collatefn = CommonCollateFn()
    batches = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]
    iter_factory = ChunkIterFactory(
        dataset=dataset,
        batches=batches,
        batch_size=2,
        chunk_length=3,
        num_cache_chunks=4,
        collate_fn=collatefn,
    )

    for key, batch in iter_factory.build_iter(0):
        for k, v in batch.items():
            if k == "data":
                assert v.shape == (2, 3)
            elif k == "utt2category":
                val = v[0].item()
                assert all([vv.item() == val for vv in v])


class SpeechLabelDataset:
    """speech[t] = t + 1 (so a padded zero is visible) and labels[t] = t."""

    def __init__(self, lengths):
        self.lengths = lengths

    def __getitem__(self, item):
        n = self.lengths[item]
        speech = np.arange(1, n + 1, dtype=np.float32)
        labels = np.stack([np.arange(n), -np.arange(n)], 1).astype(np.float32)
        return item, {"speech": speech, "spk_labels": labels}


@pytest.mark.parametrize("shuffle", [False, True])
def test_left_context_is_given_to_speech_only(shuffle):
    """The speech chunk is the label chunk with C samples in front of it.

    Labels keep [a, a + W); speech gets [a - C, a + W), with zeros before the
    start of the recording, so after removing the first C samples the speech
    and the labels cover the same span.
    """
    W, C = 4, 3
    dataset = SpeechLabelDataset({"r1": 20, "r2": 11})
    iter_factory = ChunkIterFactory(
        dataset=dataset,
        batches=[["r1"], ["r2"]],
        batch_size=2,
        chunk_length=W,
        chunk_shift_ratio=1.0,
        shuffle=shuffle,
        collate_fn=CommonCollateFn(),
        left_context=C,
    )
    n = 0
    for _, batch in iter_factory.build_iter(0):
        speech, labels = batch["speech"], batch["spk_labels"]
        assert speech.shape[1] == C + W and labels.shape[1] == W
        # the chunks of a key have one length, so no *_lengths are given
        assert "speech_lengths" not in batch
        for s, lab in zip(speech, labels):
            start = int(lab[0, 0])  # label value = sample index
            # the window itself
            np.testing.assert_array_equal(s[C:].numpy(), lab[:, 0].numpy() + 1)
            # its context: the C samples before it, zeros before the start
            expected = np.arange(start - C, start) + 1
            expected[expected <= 0] = 0
            np.testing.assert_array_equal(s[:C].numpy(), expected)
            n += 1
    assert n > 0


def test_left_context_scales_with_the_sampling_rate():
    """With default_fs, the context is in samples at default_fs, as the chunk."""

    class Dataset8k(SpeechLabelDataset):
        def __getitem__(self, item):
            _, d = super().__getitem__(item)
            d["utt2fs"] = np.array([8000])
            return item, d

    iter_factory = ChunkIterFactory(
        dataset=Dataset8k({"r": 40}),
        batches=[["r"]],
        batch_size=1,
        chunk_length=8,
        chunk_shift_ratio=1.0,
        collate_fn=CommonCollateFn(),
        default_fs=16000,
        left_context=6,
        excluded_key_prefixes=["utt2fs"],
    )
    for _, batch in iter_factory.build_iter(0):
        assert batch["spk_labels"].shape[1] == 4  # 8 at 16 kHz -> 4 at 8 kHz
        assert batch["speech"].shape[1] == 3 + 4  # 6 at 16 kHz -> 3 at 8 kHz


def test_left_context_must_be_non_negative():
    with pytest.raises(ValueError):
        ChunkIterFactory(
            dataset=SpeechLabelDataset({"r": 10}),
            batches=[["r"]],
            batch_size=1,
            chunk_length=4,
            left_context=-1,
        )


@pytest.mark.parametrize("as_tensor", [False, True])
def test_with_left_context_pads_numpy_and_tensors(as_tensor):
    import torch

    v = np.arange(1, 13, dtype=np.float32).reshape(6, 2)  # (time, channel)
    if as_tensor:
        v = torch.from_numpy(v)
    out = ChunkIterFactory._with_left_context(v, 1, 3, 2)
    out = np.asarray(out)
    assert out.shape == (5, 2)
    np.testing.assert_array_equal(out[0], [0, 0])  # before the start
    np.testing.assert_array_equal(out[1:], np.arange(1, 9).reshape(4, 2))
    inside = np.asarray(ChunkIterFactory._with_left_context(v, 3, 2, 2))
    np.testing.assert_array_equal(inside, np.arange(3, 11).reshape(4, 2))
