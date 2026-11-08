"""--chunk_random_access: chunks read on their own (Omni-IO entries, RTTM labels)."""

import argparse
import itertools

import numpy as np
import pytest

from espnet2.fileio.omniio_audio import OmniIOAudioScpReader, OmniIOAudioWriter
from espnet2.fileio.rttm import RttmReader
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.tasks.abs_task import IteratorOptions
from espnet2.tasks.diar import DiarizationTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.dataset import ESPnetDataset

RATE = 16000


def _recordings():
    rng = np.random.default_rng(0)
    # values on the 16-bit grid, so FLAC stores them exactly
    return {
        "recA": rng.integers(-2000, 2000, 10500) / 32768.0,
        "recB": rng.integers(-2000, 2000, 4000) / 32768.0,
        "recC": rng.integers(-2000, 2000, 1500) / 32768.0,  # shorter than a chunk
    }


TURNS = {  # espnet RTTM: samples, end inclusive; two speakers each, as collate needs
    "recA": [("s1", 100, 1999), ("s1", 1500, 2599), ("s2", 2400, 3999)],
    "recB": [("s3", 0, 999), ("s4", 3000, 3999)],
    "recC": [("s1", 0, 999), ("s2", 1000, 1499)],
}


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    sigs = _recordings()
    with OmniIOAudioWriter(tmp_path / "arc", d / "wav.scp", format="flac") as w:
        for rec, sig in sigs.items():
            w[rec] = RATE, sig
    with (d / "espnet_rttm").open("w") as f:
        for rec, turns in TURNS.items():
            for spk, s, e in turns:
                f.write(f"SPEAKER {rec} 1 {s} {e} <NA> <NA> {spk} <NA>\n")
            f.write(f"END {rec} <NA> <NA> {len(sigs[rec])} <NA> <NA> <NA> <NA>\n")
    (d / "speech_shape").write_text(
        "".join(f"{rec} {len(sig)}\n" for rec, sig in sigs.items())
    )
    return d, sigs


@pytest.mark.parametrize("start, end", [(0, 1), (0, 4096), (4095, 4097), (100, 10500)])
def test_omniio_read_span_equals_the_slice(data, start, end):
    d, sigs = data
    reader = OmniIOAudioScpReader(d / "wav.scp")
    rate, span = reader.read_span("recA", start, end)
    assert rate == RATE
    np.testing.assert_array_equal(span, reader["recA"][1][start:end])
    np.testing.assert_array_equal(span, sigs["recA"][start:end].astype(np.float32))


@pytest.mark.parametrize("start, end", [(-1, 10), (5, 5), (0, 10501)])
def test_omniio_read_span_outside_raises(data, start, end):
    d, _ = data
    with pytest.raises(ValueError):
        OmniIOAudioScpReader(d / "wav.scp").read_span("recA", start, end)


@pytest.mark.parametrize("start, end", [(0, 2000), (2000, 4000), (4000, 6000), (5, 9)])
def test_rttm_read_span_equals_the_slice(data, start, end):
    d, _ = data
    reader = RttmReader(str(d / "espnet_rttm"))
    span = reader.read_span("recA", start, end)
    assert span.shape == (end - start, 2)  # every speaker of the recording
    np.testing.assert_array_equal(span, reader["recA"][start:end])
    with pytest.raises(ValueError):
        reader.read_span("recA", 0, 10501)


def test_dataset_item_of_a_span(data):
    d, sigs = data
    dataset = ESPnetDataset(
        [
            (str(d / "wav.scp"), "speech", "omniio"),
            (str(d / "espnet_rttm"), "spk_labels", "rttm"),
        ]
    )
    uid, whole = dataset["recA"]
    cid, span = dataset[("recA", 2000, 4000)]
    assert (uid, cid) == ("recA", "recA-0000002000")
    np.testing.assert_array_equal(span["speech"], whole["speech"][2000:4000])
    np.testing.assert_array_equal(span["spk_labels"], whole["spk_labels"][2000:4000])


def _factory(
    d, chunk_length=2000, shift_ratio=1.0, batch_size=2, train=True, extra=(), args=None
):
    parser = DiarizationTask.get_parser()
    args = args or parser.parse_args(
        list(extra)
        + [
            "--iterator_type",
            "chunk",
            "--chunk_random_access",
            "true",
            "--chunk_length",
            str(chunk_length),
            "--chunk_shift_ratio",
            str(shift_ratio),
            "--num_spk",
            "2",
        ]
    )
    options = _options(d, batch_size, train)
    return DiarizationTask.build_chunk_iter_factory(args, options, "train")


def _options(d, batch_size, train):
    return IteratorOptions(
        preprocess_fn=None,
        collate_fn=CommonCollateFn(float_pad_value=0.0, int_pad_value=-1),
        data_path_and_name_and_type=[
            (str(d / "wav.scp"), "speech", "omniio"),
            (str(d / "espnet_rttm"), "spk_labels", "rttm"),
        ],
        shape_files=[str(d / "speech_shape")],
        batch_size=batch_size,
        batch_bins=0,
        batch_type="unsorted",
        max_cache_size=0.0,
        max_cache_fd=32,
        allow_multi_rates=False,
        distributed=False,
        num_batches=None,
        num_iters_per_epoch=None,
        train=train,
    )


@pytest.mark.parametrize("shift_ratio", [1.0, 0.5])
def test_chunks_cover_each_sequence_from_its_start(data, shift_ratio):
    d, sigs = data
    W, S = 2000, int(2000 * shift_ratio)
    batches = list(_factory(d, shift_ratio=shift_ratio, train=False).build_iter(1))
    ids = list(itertools.chain(*(ids for ids, _ in batches)))
    want = [
        f"{rec}-{st:010d}"
        for rec in sorted(sigs)
        for st in range(0, len(sigs[rec]) - W + 1, S)
    ]
    assert ids == want  # recC is shorter than a chunk; tails are dropped
    rttm = RttmReader(str(d / "espnet_rttm"))
    for ids_, batch in batches:
        assert batch["speech"].shape[1] == W
        assert batch["spk_labels"].shape[1:] == (W, 2)
        for i, cid in enumerate(ids_):
            rec, st = cid.rsplit("-", 1)
            st = int(st)
            np.testing.assert_allclose(
                batch["speech"][i].numpy(), sigs[rec][st : st + W], atol=1e-7
            )
            np.testing.assert_array_equal(
                batch["spk_labels"][i].numpy(), rttm[rec][st : st + W]
            )


def test_batches_are_those_of_the_unsorted_sampler(data, tmp_path):
    """Same mini-batches as --batch_type unsorted over a key file of the chunks."""
    d, sigs = data
    keys = [
        f"{rec}-{st:010d}"
        for rec in sorted(sigs)
        for st in range(0, len(sigs[rec]) - 2000 + 1, 2000)
    ]
    key_file = tmp_path / "keys"
    key_file.write_text("".join(f"{k} 2000\n" for k in keys))
    want = [list(b) for b in UnsortedBatchSampler(batch_size=2, key_file=str(key_file))]
    factory = _factory(d, batch_size=2)
    got = [list(ids) for ids, _ in factory.build_iter(1, shuffle=False)]
    assert got == want
    # shuffled: the same mini-batches in another order, reproducible per epoch
    e1 = [tuple(ids) for ids, _ in factory.build_iter(1)]
    assert sorted(e1) == sorted(tuple(b) for b in want)
    assert e1 == [tuple(ids) for ids, _ in factory.build_iter(1)]


def test_bad_chunk_length(data):
    d, _ = data
    with pytest.raises(ValueError):
        _factory(d, chunk_length="1000,2000")
    with pytest.raises(RuntimeError):
        _factory(d, chunk_length=20000)


def test_parser_default_is_off():
    args = DiarizationTask.get_parser().parse_args([])
    assert isinstance(args, argparse.Namespace) and args.chunk_random_access is False


@pytest.mark.parametrize("context", [500, 2500])
def test_left_context_on_the_speech_only(data, context):
    """--chunk_left_context C on the speech only.

    The speech chunk is [a - C, a + W), zeros before the start of the
    recording; the labels keep [a, a + W).
    """
    d, sigs = data
    W, C = 2000, context
    factory = _factory(d, train=False, extra=["--chunk_left_context", str(C)])
    rttm = RttmReader(str(d / "espnet_rttm"))
    n = 0
    for ids, batch in factory.build_iter(1):
        assert batch["speech"].shape[1] == C + W
        assert batch["spk_labels"].shape[1] == W
        for i, cid in enumerate(ids):
            rec, st = cid.rsplit("-", 1)
            st = int(st)
            sig = sigs[rec].astype(np.float32)
            want = np.concatenate(
                [np.zeros(max(C - st, 0), np.float32), sig[max(st - C, 0) : st + W]]
            )
            np.testing.assert_array_equal(batch["speech"][i].numpy(), want)
            np.testing.assert_array_equal(
                batch["spk_labels"][i].numpy(), rttm[rec][st : st + W]
            )
            n += 1
    assert n == 7  # recA: 5 chunks, recB: 2 chunks


def test_left_context_needs_fixed_rate(data):
    d, _ = data
    with pytest.raises(ValueError):
        _factory(
            d, extra=["--chunk_left_context", "500", "--chunk_default_fs", "16000"]
        )


@pytest.mark.parametrize("train", [True, False])
def test_model_on_the_speech_after_its_context(data, train):
    """Chunk iterator to diar model with the context removed from the speech.

    Once the C samples of context are removed from the speech (as
    BatchWavAugment(context=C) does), the frames match the labels, in
    training and in validation.
    """
    import torch

    d, _ = data
    W, C = 2000, 512
    parser = DiarizationTask.get_parser()
    args = parser.parse_args(
        [
            "--iterator_type",
            "chunk",
            "--chunk_random_access",
            "true",
            "--chunk_length",
            str(W),
            "--chunk_shift_ratio",
            "1.0",
            "--chunk_left_context",
            str(C),
            "--num_spk",
            "2",
            "--encoder",
            "transformer",
            "--encoder_conf",
            "input_layer=linear",
            "--encoder_conf",
            "num_blocks=1",
            "--encoder_conf",
            "linear_units=16",
            "--encoder_conf",
            "output_size=8",
            "--encoder_conf",
            "attention_heads=2",
            "--decoder",
            "linear",
            "--frontend_conf",
            "fs=16k",
        ]
    )
    model = DiarizationTask.build_model(args)
    model.train(train)
    factory = _factory(d, train=train, args=args)
    for _, batch in factory.build_iter(1):
        speech = batch["speech"][:, C:]  # the model's crop of the context
        assert speech.shape[1] == batch["spk_labels"].shape[1] == W
        with torch.set_grad_enabled(train):
            loss, stats, _ = model(
                speech=speech,
                speech_lengths=torch.full((speech.shape[0],), W),
                spk_labels=batch["spk_labels"],
                spk_labels_lengths=torch.full((speech.shape[0],), W),
            )
        assert torch.isfinite(loss)


def _write_recording(tmp_path, rate, length):
    """One recording `rec` of `length` samples at `rate` and its two-speaker label."""
    rng = np.random.default_rng(rate + length)
    sig = rng.integers(-2000, 2000, length) / 32768.0
    d = tmp_path / "data"
    with OmniIOAudioWriter(tmp_path / "arc", d / "wav.scp", format="flac") as w:
        w["rec"] = rate, sig
    a, b = length // 5, length // 2
    (d / "espnet_rttm").write_text(
        f"SPEAKER rec 1 {a} {b} <NA> <NA> s1 <NA>\n"
        f"SPEAKER rec 1 {b // 2} {length - 1} <NA> <NA> s2 <NA>\n"
        f"END rec <NA> <NA> {length} <NA> <NA> <NA> <NA>\n"
    )
    (d / "speech_shape").write_text(f"rec {length}\n")
    return d, sig.astype(np.float32)


@pytest.mark.parametrize(
    "rate, chunk_length, shift_ratio, context",
    [
        # every rate: overlapping chunks with left context
        (8000, 1000, 0.5, 300),
        (22050, 1000, 0.5, 300),
        (44100, 1000, 0.5, 300),
        (48000, 1000, 0.5, 300),
        # chunk lengths, at 16 kHz
        (16000, 1, 1.0, 0),  # one sample
        (16000, 4096, 1.0, 0),  # one FLAC block (4096 samples)
        (16000, 4097, 0.25, 5000),  # across blocks, context longer than a start
        (16000, 10500, 1.0, 0),  # the whole recording
    ],
)
def test_chunks_at_any_rate_and_length(
    tmp_path, rate, chunk_length, shift_ratio, context
):
    """Chunk positions, audio and labels at 8-48 kHz and chunk lengths.

    Chunk lengths go from one sample to the whole recording; each run has one
    fixed sampling rate.
    """
    L, W, S = 10500, chunk_length, int(chunk_length * shift_ratio)
    d, sig = _write_recording(tmp_path, rate, L)
    extra = ["--chunk_left_context", str(context)] if context else []
    factory = _factory(
        d, chunk_length=W, shift_ratio=shift_ratio, batch_size=3, train=False,
        extra=extra + ["--num_workers", "0"],
    )  # fmt: skip
    rttm = RttmReader(str(d / "espnet_rttm"))
    starts = []
    for ids, batch in factory.build_iter(1):
        for i, cid in enumerate(ids):
            st = int(cid.rsplit("-", 1)[1])
            starts.append(st)
            want = np.concatenate(
                [
                    np.zeros(max(context - st, 0), np.float32),
                    sig[max(st - context, 0) : st + W],
                ]
            )
            np.testing.assert_array_equal(batch["speech"][i].numpy(), want)
            np.testing.assert_array_equal(
                batch["spk_labels"][i].numpy(), rttm["rec"][st : st + W]
            )
    assert starts == list(range(0, L - W + 1, S))
