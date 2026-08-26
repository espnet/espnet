import numpy as np
import pytest

from espnet2.train.collate_fn import CommonCollateFn, HuBERTCollateFn, common_collate_fn


@pytest.mark.parametrize(
    "float_pad_value, int_pad_value, not_sequence",
    [(0.0, -1, ()), (3.0, 2, ("a",)), (np.inf, 100, ("a", "b"))],
)
def test_common_collate_fn(float_pad_value, int_pad_value, not_sequence):
    data = [
        ("id", dict(a=np.random.randn(3, 5), b=np.random.randn(4).astype(np.int64))),
        ("id2", dict(a=np.random.randn(2, 5), b=np.random.randn(3).astype(np.int64))),
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
        a_lengths=np.array([3, 2], dtype=np.int64),
        b_lengths=np.array([4, 3], dtype=np.int64),
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
        ("id", dict(a=np.random.randn(3, 5), b=np.random.randn(4).astype(np.int64))),
        ("id2", dict(a=np.random.randn(2, 5), b=np.random.randn(3).astype(np.int64))),
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
        a_lengths=np.array([3, 2], dtype=np.int64),
        b_lengths=np.array([4, 3], dtype=np.int64),
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
    (
        "float_pad_value, int_pad_value, not_sequence, label_downsampling, pad,"
        "rand_crop, window_size, window_shift, sample_rate"
    ),
    [
        (0.0, -1, (), 1, True, True, 25, 20, 16),
        (3.0, 2, ("a",), 1, False, False, 25, 20, 16),
        (np.inf, 100, ("a", "b"), 2, True, False, 25, 20, 16),
    ],
)
def test_HuBERT_(
    float_pad_value,
    int_pad_value,
    not_sequence,
    label_downsampling,
    pad,
    rand_crop,
    window_size,
    window_shift,
    sample_rate,
):
    _hubert_collate_fn = HuBERTCollateFn(
        float_pad_value=float_pad_value,
        int_pad_value=int_pad_value,
        not_sequence=not_sequence,
        label_downsampling=label_downsampling,
        pad=pad,
        rand_crop=rand_crop,
        window_size=window_size,
        window_shift=window_shift,
        sample_rate=sample_rate,
    )
    data = [
        (
            "id",
            dict(
                speech=np.random.randn(16000), text=np.random.randn(49).astype(np.int64)
            ),
        ),
        (
            "id2",
            dict(
                speech=np.random.randn(22000), text=np.random.randn(67).astype(np.int64)
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
            speech_lengths=np.array([16000, 22000], dtype=np.int64),
            text_lengths=np.array([49, 67], dtype=np.int64),
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
            speech_lengths=np.array([16000, 16000], dtype=np.int64),
            text_lengths=np.array([49, 49], dtype=np.int64),
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
    (
        "float_pad_value, int_pad_value, not_sequence, label_downsampling, pad, "
        "rand_crop, window_size, window_shift, sample_rate"
    ),
    [
        (0.0, -1, (), 1, True, True, 25, 20, 16),
        (3.0, 2, ("a",), 1, False, False, 80, 40, 16),
        (np.inf, 100, ("a", "b"), 2, True, False, 25, 20, 16),
    ],
)
def test_HuBERTCollateFn_repr(
    float_pad_value,
    int_pad_value,
    not_sequence,
    label_downsampling,
    pad,
    rand_crop,
    window_size,
    window_shift,
    sample_rate,
):
    print(
        HuBERTCollateFn(
            float_pad_value=float_pad_value,
            int_pad_value=int_pad_value,
            not_sequence=not_sequence,
            label_downsampling=label_downsampling,
            pad=pad,
            rand_crop=rand_crop,
            window_size=window_size,
            window_shift=window_shift,
            sample_rate=sample_rate,
        )
    )


# ---------------------------------------------------------------------------
# WavLM masked speech denoising / speaker mixing (`mix_speech`).
#
# WavLM's objective differs from HuBERT's only in that the *input* waveform is
# corrupted -- with either a segment of another utterance in the batch
# ("separation") or a sampled acoustic noise ("denoising") -- while the k-means
# targets stay those of the clean primary utterance. See
# https://arxiv.org/abs/2110.13900 sec. 2.2.
#
# These tests use constant-valued waveforms so the mixing gain is exact:
# `_add_noise_wavlm` scales the interferer to sit `noise_db` dB below the
# primary, i.e. scale = 10**(-noise_db/20) * sqrt(P_primary / P_interferer}, so
# with noise_db == 0 a primary of 0.1 mixed with an interferer of 0.2 gains
# exactly sqrt(0.01/0.04) * 0.2 == 0.1 over the mixed region.
# ---------------------------------------------------------------------------

WAVLM_LEN = 4000


def _const_batch(*values, length=WAVLM_LEN, n_labels=25):
    """One utterance per value, each a constant waveform of that amplitude."""
    return [
        (
            f"utt{i}",
            dict(
                speech=np.full(length, v, dtype=np.float32),
                text=np.arange(n_labels, dtype=np.int64),
            ),
        )
        for i, v in enumerate(values)
    ]


def _wavlm_collate_fn(**kwargs):
    """A collate_fn that isolates mixing: no cropping, no label downsampling."""
    conf = dict(
        float_pad_value=0.0,
        int_pad_value=-1,
        label_downsampling=1,
        pad=True,
        rand_crop=False,
        crop_audio=False,
        mix_speech=True,
        noise_apply_prob=1.0,
        dynamic_mixing_prob=0.0,
        dynamic_mixing_gain_db=0.0,
        noise_scp=None,
        rir_scp=None,
        train=True,
    )
    conf.update(kwargs)
    return HuBERTCollateFn(**conf)


def _wavlm_seed(seed):
    """Seed both RNGs `_add_noise_wavlm` draws from.

    It picks the interfering utterance with Python's `random` and everything else
    (branch, crop, gain) with `np.random`, so seeding only one leaves the mixture
    dependent on whatever ran before.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)


def _mixed_region(clean, mixed):
    """(start, stop) of the samples mixing changed, or None if unchanged."""
    changed = np.flatnonzero(~np.isclose(clean, mixed))
    if changed.size == 0:
        return None
    start, stop = int(changed[0]), int(changed[-1]) + 1
    # WavLM mixes a single contiguous crop, not a scattering of samples.
    assert changed.size == stop - start, "mixed region is not contiguous"
    return start, stop


def _write_noise_scp(tmp_path, value, length=WAVLM_LEN, fs=16000):
    import soundfile

    wav = tmp_path / "noise.wav"
    soundfile.write(wav, np.full(length, value, dtype=np.float32), fs)
    scp = tmp_path / "noise.scp"
    scp.write_text(f"noise1 {wav}\n")
    return str(scp)


def test_wavlm_mixing_disabled_leaves_speech_untouched():
    data = _const_batch(0.1, 0.2)
    clean = [d["speech"].copy() for _, d in data]
    _, out = _wavlm_collate_fn(mix_speech=False)(data)
    for i, ref in enumerate(clean):
        np.testing.assert_array_equal(out["speech"][i].numpy(), ref)


def test_wavlm_mixing_is_train_only():
    """Validation must see clean speech, or valid loss is not comparable."""
    data = _const_batch(0.1, 0.2)
    clean = [d["speech"].copy() for _, d in data]
    _, out = _wavlm_collate_fn(train=False)(data)
    for i, ref in enumerate(clean):
        np.testing.assert_array_equal(out["speech"][i].numpy(), ref)


@pytest.mark.parametrize("noise_apply_prob, expect_mixed", [(0.0, False), (1.0, True)])
def test_wavlm_mixing_respects_apply_prob(noise_apply_prob, expect_mixed):
    data = _const_batch(0.1, 0.2)
    clean = [d["speech"].copy() for _, d in data]
    _wavlm_seed(0)
    _, out = _wavlm_collate_fn(noise_apply_prob=noise_apply_prob)(data)
    mixed_any = any(
        _mixed_region(clean[i], out["speech"][i].numpy()) is not None
        for i in range(len(clean))
    )
    assert mixed_any is expect_mixed


def test_wavlm_separation_mixes_in_another_batch_utterance():
    """The interferer is another utterance, scaled to the requested SNR."""
    data = _const_batch(0.1, 0.2)
    clean = [d["speech"].copy() for _, d in data]
    _wavlm_seed(0)
    _, out = _wavlm_collate_fn()(data)

    # utt0 can only have drawn utt1 and vice versa, so with noise_db == 0 dB the
    # gain over the mixed region is fully determined.
    expected_gain = [
        np.sqrt(0.1**2 / 0.2**2) * 0.2,  # utt0 + scaled utt1 == 0.1
        np.sqrt(0.2**2 / 0.1**2) * 0.1,  # utt1 + scaled utt0 == 0.2
    ]
    for i in range(2):
        mixed = out["speech"][i].numpy()
        region = _mixed_region(clean[i], mixed)
        assert region is not None, f"utt{i} was not mixed"
        start, stop = region
        np.testing.assert_allclose(
            mixed[start:stop],
            clean[i][start:stop] + expected_gain[i],
            rtol=1e-5,
            atol=1e-7,
        )
        # Outside the mixed crop the primary is untouched, bit for bit.
        np.testing.assert_array_equal(mixed[:start], clean[i][:start])
        np.testing.assert_array_equal(mixed[stop:], clean[i][stop:])


def test_wavlm_mixed_region_is_at_most_half_the_primary():
    """WavLM crops "at most 50%" of the primary for the interference."""
    data = _const_batch(0.1, 0.2)
    clean = [d["speech"].copy() for _, d in data]
    for seed in range(25):
        _wavlm_seed(seed)
        _, out = _wavlm_collate_fn()(
            [(u, dict(speech=d["speech"].copy(), text=d["text"])) for u, d in data]
        )
        for i in range(2):
            region = _mixed_region(clean[i], out["speech"][i].numpy())
            assert region is not None
            start, stop = region
            assert 1 <= stop - start <= WAVLM_LEN // 2
            assert 0 <= start and stop <= WAVLM_LEN


def test_wavlm_mixing_leaves_kmeans_targets_untouched():
    """The model must still predict the *clean* primary's cluster ids."""
    data = _const_batch(0.1, 0.2)
    labels = [d["text"].copy() for _, d in data]
    _wavlm_seed(0)
    _, out = _wavlm_collate_fn()(data)
    assert _mixed_region(
        np.full(WAVLM_LEN, 0.1, np.float32), out["speech"][0].numpy()
    ) is not None
    for i, ref in enumerate(labels):
        np.testing.assert_array_equal(out["text"][i].numpy(), ref)


def test_wavlm_denoising_uses_the_noise_corpus(tmp_path):
    """dynamic_mixing_prob=1 routes to the sampled-noise (denoising) branch.

    The noise file is negative-valued so the branch is identifiable: at 0 dB the
    interferer is power-normalized, so amplitude alone cannot tell a 0.2
    utterance from a 0.5 noise -- only the sign of the residual can.
    """
    noise_scp = _write_noise_scp(tmp_path, -0.5)
    data = _const_batch(0.1, 0.2)
    clean = [d["speech"].copy() for _, d in data]
    _wavlm_seed(0)
    _, out = _wavlm_collate_fn(
        noise_scp=noise_scp, dynamic_mixing_prob=1.0, noise_db_range="6"
    )(data)

    for i, amp in enumerate([0.1, 0.2]):
        mixed = out["speech"][i].numpy()
        region = _mixed_region(clean[i], mixed)
        assert region is not None
        start, stop = region
        # 6 dB below the noise file (-0.5), not below the other utterance.
        gain = 10 ** (-6 / 20) * np.sqrt(amp**2 / 0.5**2) * -0.5
        np.testing.assert_allclose(
            mixed[start:stop], clean[i][start:stop] + gain, rtol=1e-5, atol=1e-7
        )
        assert gain < 0, "residual should carry the noise file's sign"


def test_wavlm_separation_preferred_when_dynamic_mixing_prob_is_zero(tmp_path):
    """With a noise corpus configured but prob 0, the interferer is in-batch."""
    noise_scp = _write_noise_scp(tmp_path, -0.5)
    data = _const_batch(0.1, 0.2)
    clean = [d["speech"].copy() for _, d in data]
    from_utterance = np.sqrt(0.1**2 / 0.2**2) * 0.2
    from_noise = np.sqrt(0.1**2 / 0.5**2) * -0.5

    for seed in range(10):
        _wavlm_seed(seed)
        _, out = _wavlm_collate_fn(noise_scp=noise_scp, dynamic_mixing_prob=0.0)(
            [(u, dict(speech=d["speech"].copy(), text=d["text"])) for u, d in data]
        )
        mixed = out["speech"][0].numpy()
        start, stop = _mixed_region(clean[0], mixed)
        np.testing.assert_allclose(
            mixed[start:stop],
            clean[0][start:stop] + from_utterance,
            rtol=1e-5,
            atol=1e-7,
        )
    # The two branches really are distinguishable by the residual's sign.
    assert np.sign(from_utterance) != np.sign(from_noise)


@pytest.mark.parametrize("noise_db_range", ["0_20", "-5_5"])
def test_wavlm_denoising_snr_within_configured_range(tmp_path, noise_db_range):
    """`noise_db` is the SNR of primary over interferer, in dB."""
    noise_scp = _write_noise_scp(tmp_path, 0.5)
    low, high = (float(v) for v in noise_db_range.split("_"))
    collate = _wavlm_collate_fn(
        noise_scp=noise_scp, dynamic_mixing_prob=1.0, noise_db_range=noise_db_range
    )
    for seed in range(20):
        data = _const_batch(0.1, 0.2)
        clean = data[0][1]["speech"].copy()
        _wavlm_seed(seed)
        _, out = collate(data)
        mixed = out["speech"][0].numpy()
        start, stop = _mixed_region(clean, mixed)
        residual = mixed[start:stop] - clean[start:stop]
        snr_db = 10 * np.log10((clean[start:stop] ** 2).mean() / (residual**2).mean())
        assert low - 1e-4 <= snr_db <= high + 1e-4, snr_db


def test_wavlm_mixing_does_not_mutate_the_input_batch():
    """Regression: mixing in place corrupts the interferer pool.

    `data` doubles as the pool the separation branch draws from, so mutating a
    waveform in place means later utterances get mixed with an already-mixed
    neighbour instead of clean speech.
    """
    data = _const_batch(0.1, 0.2, 0.3)
    before = [d["speech"].copy() for _, d in data]
    _wavlm_seed(0)
    _wavlm_collate_fn()(data)
    for (_, sample), ref in zip(data, before):
        np.testing.assert_array_equal(sample["speech"], ref)


def test_wavlm_mixing_does_not_accumulate_over_epochs():
    """Regression: ESPnetDataset hands back the same array when caching.

    With `max_cache_size` set, `__getitem__` returns the cached ndarray itself,
    so in-place mixing would compound epoch after epoch instead of resampling
    fresh interference each time.
    """
    cached = _const_batch(0.1, 0.2, 0.3)
    collate = _wavlm_collate_fn()

    _wavlm_seed(1234)
    first = collate(cached)[1]["speech"].numpy().copy()  # same dicts as a cache hit

    for epoch in range(5):
        _wavlm_seed(epoch)
        collate(cached)
        # the cached arrays themselves must stay pristine
        for (_, sample), amp in zip(cached, [0.1, 0.2, 0.3]):
            np.testing.assert_array_equal(sample["speech"], np.float32(amp))

    # Replaying a seed reproduces the mixture exactly; had the interference been
    # written back into the cached arrays it would have compounded instead.
    _wavlm_seed(1234)
    replay = collate(cached)[1]["speech"].numpy()
    np.testing.assert_array_equal(replay, first)


def test_wavlm_single_utterance_batch_without_noise_corpus():
    """Separation is impossible; leave the batch alone rather than crash."""
    data = _const_batch(0.1)
    clean = data[0][1]["speech"].copy()
    _wavlm_seed(0)
    _, out = _wavlm_collate_fn()(data)
    np.testing.assert_array_equal(out["speech"][0].numpy(), clean)


def test_wavlm_single_utterance_batch_falls_back_to_noise(tmp_path):
    noise_scp = _write_noise_scp(tmp_path, -0.5)
    data = _const_batch(0.1)
    clean = data[0][1]["speech"].copy()
    _wavlm_seed(0)
    _, out = _wavlm_collate_fn(
        noise_scp=noise_scp, dynamic_mixing_prob=0.0, noise_db_range="6"
    )(data)
    start, stop = _mixed_region(clean, out["speech"][0].numpy())
    gain = 10 ** (-6 / 20) * np.sqrt(0.1**2 / 0.5**2) * -0.5
    np.testing.assert_allclose(
        out["speech"][0].numpy()[start:stop],
        clean[start:stop] + gain,
        rtol=1e-5,
        atol=1e-7,
    )
