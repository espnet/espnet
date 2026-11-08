import logging

import numpy as np
import pytest
import soundfile
import torch

from espnet2.layers.batch_wav_augment import BatchWavAugment
from espnet2.train.preprocessor import detect_non_silence


def write_noises(tmp_path, specs, fs=16000):
    """specs: list of (num_samples, num_channels); returns the scp path."""
    rng = np.random.default_rng(0)
    lines = []
    for i, (n, ch) in enumerate(specs):
        path = tmp_path / f"noise{i}.wav"
        wav = (rng.uniform(-0.5, 0.5, size=(n, ch)) * 32767).astype(np.int16)
        soundfile.write(path, wav, fs, subtype="PCM_16")
        lines.append(f"n{i} {path}\n")
    scp = tmp_path / "noise.scp"
    scp.write_text("".join(lines))
    return str(scp)


def test_eval_mode_and_zero_prob_leave_speech_unchanged(tmp_path):
    scp = write_noises(tmp_path, [(8000, 1)])
    speech = torch.randn(2, 4000) * 0.1
    lengths = torch.tensor([4000, 3000])
    aug = BatchWavAugment(scp, noise_apply_prob=1.0).eval()
    out, out_lengths = aug(speech, lengths)
    assert out is speech and out_lengths is lengths
    assert aug._cache == {}  # never loaded outside training

    aug = BatchWavAugment(scp, noise_apply_prob=0.0).train()
    out, _ = aug(speech, lengths)
    assert out is speech


def test_snr_within_range_and_padding_untouched(tmp_path):
    scp = write_noises(tmp_path, [(20000, 1), (3000, 1), (9000, 2)])
    torch.manual_seed(0)
    b, t = 16, 6000
    speech = torch.randn(b, t) * 0.01  # no silence, far from the peak limit
    lengths = torch.full((b,), t)
    lengths[::4] = 4500
    valid = torch.arange(t) < lengths[:, None]
    speech = speech * valid
    aug = BatchWavAugment(scp, noise_apply_prob=1.0, noise_db_range="0_10").train()
    out, _ = aug(speech, lengths)
    assert out.shape == speech.shape and out.dtype == speech.dtype
    assert torch.all(out[~valid] == 0)
    added = out - speech
    for i in range(b):
        n = int(lengths[i])
        ps = speech[i, :n].pow(2).mean()
        pn = added[i, :n].pow(2).mean()
        snr = 10 * torch.log10(ps / pn)
        assert -0.01 <= snr <= 10.01, (i, snr)


def test_short_noise_is_repeated(tmp_path):
    scp = write_noises(tmp_path, [(1000, 1)])
    aug = BatchWavAugment(scp, noise_db_range="0").train()
    speech = torch.full((1, 3500), 0.01)
    out, _ = aug(speech, torch.tensor([3500]))
    added = (out - speech)[0]
    # period 1000: the added noise repeats
    torch.testing.assert_close(added[:1000], added[1000:2000])
    torch.testing.assert_close(added[1000:2000], added[2000:3000])


def test_multichannel_noise_gives_one_noise_per_channel(tmp_path):
    scp = write_noises(tmp_path, [(5000, 3), (4000, 1)])
    aug = BatchWavAugment(scp).train()
    aug(torch.randn(1, 2000) * 0.01, None)
    bank, starts, lengths = aug.bank("noise", torch.device("cpu"))
    assert lengths.tolist() == [5000, 5000, 5000, 4000]
    assert starts.tolist() == [0, 5000, 10000, 15000]
    assert bank.dtype == torch.int16 and bank.numel() == 19000


@pytest.mark.parametrize("t", [800, 5000, 16000])
def test_non_silent_power_matches_detect_non_silence(tmp_path, t):
    scp = write_noises(tmp_path, [(1000, 1)])
    aug = BatchWavAugment(scp)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, t)).astype(np.float32) * 0.1
    x[0, t // 3 : 2 * t // 3] = 0  # a silent middle
    x[1] *= np.linspace(0, 1, t, dtype=np.float32) ** 4  # fades in from silence
    x[2] = 0  # all silent
    expected = []
    for row in x:
        mask = detect_non_silence(row[None])[0]
        expected.append((row[mask] ** 2).mean() if mask.any() else 0.0)
    got = aug._non_silent_power(torch.from_numpy(x), torch.ones(3, t, dtype=torch.bool))
    np.testing.assert_allclose(got.numpy(), np.array(expected), rtol=1e-4, atol=1e-12)


def test_peak_is_normalised(tmp_path):
    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchWavAugment(scp, noise_db_range="-5").train()
    speech = torch.randn(2, 4000) * 0.9
    out, _ = aug(speech, torch.tensor([4000, 4000]))
    assert out.abs().max() <= 1.0 + 1e-6


def test_noise_is_not_in_the_state_dict(tmp_path):
    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchWavAugment(scp).train()
    aug(torch.randn(1, 2000) * 0.01, None)
    assert ("noise", "host") in aug._cache
    assert len(aug.state_dict()) == 0
    # not a buffer: DistributedDataParallel does not broadcast it
    assert list(aug.buffers()) == []


def test_bad_db_range(tmp_path):
    with pytest.raises(ValueError):
        BatchWavAugment("unused.scp", noise_db_range="10_0")


@pytest.mark.parametrize("lengths", [[16000, 9000, 5000, 1500, 700], [3000, 2999]])
def test_non_silent_power_of_padded_rows(tmp_path, lengths):
    """Each padded row: the power of the utterance alone (detect_non_silence)."""
    scp = write_noises(tmp_path, [(1000, 1)])
    aug = BatchWavAugment(scp)
    rng = np.random.default_rng(2)
    t = max(lengths)
    x = np.zeros((len(lengths), t), dtype=np.float32)
    for i, n in enumerate(lengths):
        row = rng.normal(size=n).astype(np.float32) * 0.1
        row[n // 2 :] *= 0.001  # ends nearly silent
        x[i, :n] = row
    valid = torch.arange(t) < torch.tensor(lengths)[:, None]
    expected = []
    for i, n in enumerate(lengths):
        row = x[i, :n]
        mask = detect_non_silence(row[None])[0]
        expected.append((row[mask] ** 2).mean())
    got = aug._non_silent_power(torch.from_numpy(x), valid)
    np.testing.assert_allclose(got.numpy(), np.array(expected), rtol=1e-4)


def test_float_noise_is_not_lost(tmp_path):
    """A floating-point WAV of small amplitude keeps its waveform in the bank."""
    rng = np.random.default_rng(3)
    wav = (rng.uniform(-1, 1, size=4000) * 1e-3).astype(np.float32)
    soundfile.write(tmp_path / "float.wav", wav, 16000, subtype="FLOAT")
    (tmp_path / "noise.scp").write_text(f"f {tmp_path / 'float.wav'}\n")
    aug = BatchWavAugment(str(tmp_path / "noise.scp")).train()
    bank, _, _ = aug.bank("noise", torch.device("cpu"))
    assert int(bank.abs().max()) == 32767  # peak-normalised, not rounded to 0
    corr = np.corrcoef(bank.numpy().astype(np.float64), wav)[0, 1]
    assert corr > 0.9999


def test_replicas_share_the_loaded_noise(tmp_path):
    """A shallow copy (as DataParallel replicates) reuses and fills the cache."""
    import copy

    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchWavAugment(scp).train()
    replica = copy.copy(aug)
    replica(torch.randn(1, 2000) * 0.01, None)
    assert ("noise", "host") in aug._cache  # the replica loaded it for the original


def test_long_noise_uses_every_offset(tmp_path):
    """A noise one sample longer than the batch: both offsets 0 and 1 occur."""
    rng = np.random.default_rng(4)
    t = 64
    wav = (rng.uniform(-0.5, 0.5, size=t + 1) * 32767).astype(np.int16)
    soundfile.write(tmp_path / "n.wav", wav, 16000, subtype="PCM_16")
    (tmp_path / "noise.scp").write_text(f"n {tmp_path / 'n.wav'}\n")
    aug = BatchWavAugment(str(tmp_path / "noise.scp"), noise_db_range="0").train()
    torch.manual_seed(0)
    speech = torch.full((200, t), 0.01)
    added = aug(speech, None)[0] - speech
    bank, _, _ = aug.bank("noise", torch.device("cpu"))
    ref = bank.float()
    seen = set()
    for row in added:
        corr = [
            torch.nn.functional.cosine_similarity(row, ref[o : o + t], dim=0)
            for o in (0, 1)
        ]
        seen.add(int(torch.stack(corr).argmax()))
    assert seen == {0, 1}


def test_concurrent_replicas_load_the_noise_once(tmp_path, monkeypatch):
    """Replicas running in threads (DataParallel) read the noise list once."""
    import copy
    import threading
    import time

    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchWavAugment(scp).train()
    calls = []
    load = aug._load

    def slow_load(kind):
        calls.append(threading.get_ident())
        time.sleep(0.2)  # long enough for every thread to arrive meanwhile
        return load(kind)

    monkeypatch.setattr(aug, "_load", slow_load)
    replicas = [copy.copy(aug) for _ in range(4)]
    results = [None] * len(replicas)

    def run(i):
        results[i] = replicas[i].bank("noise", torch.device("cpu"))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(replicas))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert all(r[0] is results[0][0] for r in results)


def test_module_can_be_deep_copied_after_loading(tmp_path):
    """No lock or other non-copyable state on the module."""
    import copy

    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchWavAugment(scp).train()
    aug(torch.randn(1, 2000) * 0.01, None)
    clone = copy.deepcopy(aug)
    out, _ = clone(torch.randn(1, 2000) * 0.01, None)
    assert out.shape == (1, 2000)


def test_repr_names_the_settings(tmp_path):
    aug = BatchWavAugment("noise.scp", noise_apply_prob=0.5, noise_db_range="-5_20")
    text = repr(aug)
    assert "noise.scp" in text and "0.5" in text and "(-5.0, 20.0)" in text


def test_noise_at_another_rate_is_resampled(tmp_path):
    """8 kHz noise for 16 kHz speech: twice as many samples in the bank."""
    pytest.importorskip("librosa")
    scp = write_noises(tmp_path, [(4000, 1)], fs=8000)
    aug = BatchWavAugment(scp, fs=16000).train()
    _, _, lengths = aug.bank("noise", torch.device("cpu"))
    assert abs(int(lengths[0]) - 8000) <= 2


def test_empty_noise_is_skipped_and_no_noise_raises(tmp_path):
    empty = tmp_path / "empty.wav"
    soundfile.write(empty, np.zeros((0, 1), dtype=np.int16), 16000, subtype="PCM_16")
    scp = write_noises(tmp_path, [(3000, 1)])
    with open(scp, "a") as f:
        f.write(f"empty {empty}\n")
    aug = BatchWavAugment(scp).train()
    _, _, lengths = aug.bank("noise", torch.device("cpu"))
    assert lengths.tolist() == [3000]

    only_empty = tmp_path / "only_empty.scp"
    only_empty.write_text(f"empty {empty}\n")
    with pytest.raises(RuntimeError):
        BatchWavAugment(str(only_empty)).train()(torch.zeros(1, 100), None)


def test_multichannel_speech_is_refused(tmp_path):
    scp = write_noises(tmp_path, [(3000, 1)])
    with pytest.raises(ValueError):
        BatchWavAugment(scp).train()(torch.zeros(1, 100, 2), None)


@pytest.mark.parametrize("t", [500, 2999, 7000])
def test_host_and_device_banks_give_the_same_noise(tmp_path, t):
    """Cutting the noise from host memory equals indexing the device bank."""
    # noises shorter than, equal to and longer than the batch width
    scp = write_noises(tmp_path, [(300, 1), (3000, 2), (9000, 1)])
    speech = torch.randn(6, t) * 0.05
    lengths = torch.tensor([t, t, t - 1, t // 2, t, t // 3 + 1])
    outs = []
    for keep in (False, True):
        aug = BatchWavAugment(scp, noise_db_range="0_10", keep_on_device=keep)
        torch.manual_seed(3)
        outs.append(aug.train()(speech, lengths)[0])
    torch.testing.assert_close(outs[0], outs[1])


_accelerators = [
    d
    for d, ok in [
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    ]
    if ok
]


@pytest.mark.skipif(not _accelerators, reason="needs a CUDA or MPS device")
@pytest.mark.parametrize("device", _accelerators)
@pytest.mark.parametrize("keep_on_device", [False, True])
def test_noise_on_an_accelerator(tmp_path, device, keep_on_device):
    """Both modes add noise at the requested SNR to a batch on an accelerator."""
    scp = write_noises(tmp_path, [(20000, 1), (3000, 1)])
    aug = BatchWavAugment(
        scp, noise_db_range="0_10", keep_on_device=keep_on_device
    ).train()
    speech = (torch.randn(4, 6000) * 0.01).to(device)
    out, _ = aug(speech, torch.full((4,), 6000, device=device))
    assert out.device.type == device
    added = (out - speech).cpu()
    snr = 10 * torch.log10(speech.cpu().pow(2).mean(1) / added.pow(2).mean(1))
    assert torch.all(snr > -0.01) and torch.all(snr < 10.01)
    # keep_on_device copies the bank to the accelerator ("cuda:0", "mps:0", ...)
    on_device = any(k == "noise" and d.startswith(device) for k, d in aug._cache)
    assert on_device == keep_on_device


# --- Same result as CommonPreprocessor for the same random choices ----------


def _write(path, wav, fs=16000, subtype="FLOAT"):
    soundfile.write(path, wav, fs, subtype=subtype)
    return str(path)


def _cpu_reference(tmp_path, monkeypatch, wav, rir_path, noise_path, offset, snr):
    """CommonPreprocessor on one utterance, with its random draws fixed.

    It applies the impulse response rir_path (or none) and then the noise
    noise_path (or none) at ``offset`` (np.random.randint's value) and ``snr``.
    """
    from espnet2.train.preprocessor import CommonPreprocessor

    rir_scp = tmp_path / "cpu_rir.scp"
    noise_scp = tmp_path / "cpu_noise.scp"
    rir_scp.write_text(f"r {rir_path}\n" if rir_path else "")
    noise_scp.write_text(f"n {noise_path}\n" if noise_path else "")
    pre = CommonPreprocessor(
        train=True,
        rir_scp=str(rir_scp) if rir_path else None,
        noise_scp=str(noise_scp) if noise_path else None,
        noise_db_range=str(snr),
        short_noise_thres=0.0,
    )
    choices = iter([p for p in (rir_path, noise_path) if p])
    monkeypatch.setattr(np.random, "random", lambda: 0.0)  # apply both
    monkeypatch.setattr(np.random, "choice", lambda items: next(choices))
    monkeypatch.setattr(np.random, "randint", lambda low, high=None: offset)
    out = pre._speech_process({"speech": wav.astype(np.float64)})["speech"]
    monkeypatch.undo()
    return np.asarray(out, dtype=np.float64).reshape(-1)


@pytest.mark.parametrize(
    "use_rir, noise_len, length",
    [
        (True, None, 16000),  # reverberation only
        (False, 40000, 16000),  # noise only, longer than the utterance
        (False, 16000, 16000),  # noise of the same length
        (False, 3000, 16000),  # shorter noise, repeated
        (True, 40000, 16000),  # both
        (True, 3000, 12345),  # both, a shorter utterance
    ],
)
def test_same_result_as_common_preprocessor(
    tmp_path, monkeypatch, use_rir, noise_len, length
):
    """Given the same choices, the GPU module equals CommonPreprocessor."""
    rng = np.random.default_rng(5)
    t = 16000
    wav = (rng.normal(size=length) * 0.05).astype(np.float32)
    wav[length // 3 : length // 2] *= 0.001  # a nearly silent stretch
    rir = None
    if use_rir:
        decay = np.exp(-np.arange(4000) / 600.0)
        rir = (rng.normal(size=4000) * decay).astype(np.float32)
        rir[0] = 1.0
    noise = None
    if noise_len:
        noise = (rng.uniform(-0.3, 0.3, size=noise_len)).astype(np.float32)
    rir_path = _write(tmp_path / "rir.wav", rir) if use_rir else None
    noise_path = _write(tmp_path / "noise.wav", noise) if noise_len else None
    snr = 5.0

    # CommonPreprocessor's offset: a start index into a longer noise, or the
    # number of noise samples put before the utterance for a shorter one.
    cpu_offset = 0
    if noise_len and noise_len > length:
        cpu_offset = 7777
    elif noise_len and noise_len < length:
        cpu_offset = 1234
    expected = _cpu_reference(
        tmp_path, monkeypatch, wav, rir_path, noise_path, cpu_offset, snr
    )

    aug = BatchWavAugment(
        noise_scp=_scp(tmp_path, "noise", noise_path),
        rir_scp=_scp(tmp_path, "rir", rir_path),
        noise_db_range=str(snr),
        rir_shift=False,  # CommonPreprocessor keeps the first samples
    ).train()
    speech = torch.zeros(1, t)
    speech[0, :length] = torch.from_numpy(wav)
    valid = torch.arange(t)[None] < length
    draws = {}
    if use_rir:
        draws.update(rir_apply=torch.tensor([True]), rir_which=torch.tensor([0]))
    if noise_len:
        # GPU offset: where in the noise the utterance starts; for a shorter
        # noise the CPU's padding of p samples starts it at (-p) mod n.
        gpu_offset = cpu_offset if noise_len >= length else (-cpu_offset) % noise_len
        draws.update(
            noise_apply=torch.tensor([True]),
            noise_which=torch.tensor([0]),
            noise_offset=torch.tensor([gpu_offset]),
            snr=torch.tensor([snr]),
        )
    out = aug.apply(speech, valid, draws)[0, :length].double().numpy()

    assert out.shape == expected.shape
    # the reference did apply something
    assert np.abs(expected - wav).max() > 0.01
    # float32 FFT and int16 noise against float64 on the CPU: about 1e-6
    err = np.abs(out - expected).max() / np.abs(expected).max()
    assert err < 1e-4, err


def _scp(tmp_path, name, path):
    if path is None:
        return None
    scp = tmp_path / f"{name}.scp"
    scp.write_text(f"{name} {path}\n")
    return str(scp)


def test_batch_equals_each_utterance_alone_with_rir_and_noise(tmp_path):
    """A padded batch gives each row what apply() gives the row alone."""
    rng = np.random.default_rng(6)
    rir = (rng.normal(size=2000) * np.exp(-np.arange(2000) / 300.0)).astype(np.float32)
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", rir))
    noise_scp = write_noises(tmp_path, [(20000, 1), (3000, 1)])
    aug = BatchWavAugment(noise_scp=noise_scp, rir_scp=rir_scp).train()
    lengths = [16000, 9000, 15990]
    speech = torch.zeros(3, 16000)
    for b, n in enumerate(lengths):
        speech[b, :n] = torch.from_numpy(rng.normal(size=n).astype(np.float32) * 0.05)
    valid = torch.arange(16000)[None] < torch.tensor(lengths)[:, None]
    torch.manual_seed(0)
    draws = aug.draw(torch.tensor(lengths))
    batched = aug.apply(speech, valid, draws)
    for b, n in enumerate(lengths):
        one = {k: v[b : b + 1] for k, v in draws.items()}
        alone = aug.apply(speech[b : b + 1, :n], valid[b : b + 1, :n], one)
        torch.testing.assert_close(batched[b, :n], alone[0], rtol=1e-4, atol=1e-6)
        assert torch.all(batched[b, n:] == 0)


def test_needs_noise_or_rir():
    with pytest.raises(ValueError):
        BatchWavAugment()


def test_rir_only_keeps_the_power(tmp_path):
    """Reverberation alone keeps the non-silent power of each utterance."""
    rng = np.random.default_rng(7)
    rir = (rng.normal(size=3000) * np.exp(-np.arange(3000) / 500.0)).astype(np.float32)
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", rir))
    for keep in (False, True):
        aug = BatchWavAugment(rir_scp=rir_scp, keep_on_device=keep).train()
        speech = torch.from_numpy(rng.normal(size=(2, 8000)).astype(np.float32)) * 0.05
        valid = torch.ones(2, 8000, dtype=torch.bool)
        out, _ = aug(speech, None)
        assert not torch.allclose(out, speech)
        torch.testing.assert_close(
            aug._non_silent_power(out, valid),
            aug._non_silent_power(speech, valid),
            rtol=1e-4,
            atol=1e-9,
        )


def test_one_column_list_holds_the_path(tmp_path):
    # A line with the path only, as CommonPreprocessor accepts it
    path = _write(tmp_path / "n.wav", np.linspace(-0.5, 0.5, 800, dtype=np.float32))
    one = tmp_path / "one.scp"
    one.write_text(f"{path}\n")
    two = tmp_path / "two.scp"
    two.write_text(f"n {path}\n")
    a = BatchWavAugment(str(one)).bank("noise", torch.device("cpu"))
    b = BatchWavAugment(str(two)).bank("noise", torch.device("cpu"))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("value", [1315, 8, 5.0])
def test_non_string_noise_db_range_warns(tmp_path, caplog, value):
    (tmp_path / "noise.scp").write_text(f"n {tmp_path / 'n.wav'}\n")
    with caplog.at_level(logging.WARNING):
        aug = BatchWavAugment(str(tmp_path / "noise.scp"), noise_db_range=value)
    assert "quote a range" in caplog.text
    assert aug.noise_db_low == aug.noise_db_high == float(value)


def test_string_noise_db_range_does_not_warn(tmp_path, caplog):
    (tmp_path / "noise.scp").write_text(f"n {tmp_path / 'n.wav'}\n")
    with caplog.at_level(logging.WARNING):
        BatchWavAugment(str(tmp_path / "noise.scp"), noise_db_range="13_15")
    assert "quote a range" not in caplog.text


# --- Noise offsets by the length of each utterance ---------------------------


def test_noise_offset_by_the_utterance_length_not_the_batch_width(tmp_path):
    """A 5000-sample utterance in a 16000-sample batch, with a 9000-sample noise.

    CommonPreprocessor gives the utterance alone a contiguous stretch of the
    noise, starting at 0..4000; the noise is not repeated. The widest row, for
    which the noise is shorter, starts anywhere in the noise.
    """
    scp = write_noises(tmp_path, [(9000, 1)])
    aug = BatchWavAugment(scp, noise_db_range="0").train()
    torch.manual_seed(0)
    offsets = torch.stack(
        [aug.draw(torch.tensor([16000, 5000]))["noise_offset"] for _ in range(2000)]
    )
    assert int(offsets[:, 1].min()) >= 0 and 3900 < int(offsets[:, 1].max()) <= 4000
    assert int(offsets[:, 0].max()) > 4000

    # the noise added to the short row is one contiguous stretch of the bank
    bank, _, _ = aug.bank("noise", torch.device("cpu"))
    ref = bank.float()
    speech = torch.zeros(2, 16000)
    speech[:, :5000] = 0.01
    speech[0] = 0.01
    for seed in range(20):
        torch.manual_seed(seed)
        added = (aug(speech, torch.tensor([16000, 5000]))[0] - speech)[1, :5000]
        best = max(
            float(torch.nn.functional.cosine_similarity(added, ref[o : o + 5000], 0))
            for o in range(0, 4001)
        )
        assert best > 0.9999, (seed, best)


def test_noise_offsets_are_uniform_over_the_fitting_starts(tmp_path):
    """noise_offset of a long noise takes every value of 0..n_len-length."""
    scp = write_noises(tmp_path, [(105, 1)])
    aug = BatchWavAugment(scp).train()
    torch.manual_seed(0)
    offsets = torch.cat(
        [aug.draw(torch.tensor([100]))["noise_offset"] for _ in range(600)]
    )
    assert sorted(set(offsets.tolist())) == [0, 1, 2, 3, 4, 5]


# --- Reverberation edge cases -------------------------------------------------


def _rir_aug(tmp_path, rir, fs=16000, **kwargs):
    path = _write(tmp_path / "rir.wav", np.asarray(rir, dtype=np.float32), fs=fs)
    return BatchWavAugment(rir_scp=_scp(tmp_path, "rir", path), **kwargs).train()


def _reverberate(aug, speech, which=0):
    b, t = speech.shape
    valid = torch.ones(b, t, dtype=torch.bool)
    draws = dict(
        rir_apply=torch.ones(b, dtype=torch.bool),
        rir_which=torch.full((b,), which),
    )
    return aug.apply(speech, valid, draws)


def test_delta_rir_leaves_speech_unchanged(tmp_path):
    aug = _rir_aug(tmp_path, [1.0, 0.0, 0.0, 0.0])
    speech = torch.from_numpy(np.random.default_rng(8).normal(size=(2, 3000))) * 0.05
    speech = speech.float()
    torch.testing.assert_close(_reverberate(aug, speech), speech, atol=1e-6, rtol=1e-5)


def test_delayed_delta_rir_shifts_by_one_sample(tmp_path):
    """rir_shift=False: speech [1, 2, 3, 4] with [0, 1] gives [0, 1, 2, 3].

    The power is then scaled back to the input's, so the result is checked up
    to that factor.
    """
    aug = _rir_aug(tmp_path, [0.0, 1.0], rir_shift=False)
    speech = torch.tensor([[1.0, 2.0, 3.0, 4.0]]) * 0.1
    out = _reverberate(aug, speech)[0]
    assert abs(float(out[0])) < 1e-6
    torch.testing.assert_close(out / out[1], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_rir_longer_than_speech(tmp_path):
    """A 5000-sample impulse response on 1000 samples: np.convolve(...)[:1000]."""
    rng = np.random.default_rng(9)
    rir = (rng.normal(size=5000) * np.exp(-np.arange(5000) / 800.0)).astype(np.float32)
    aug = _rir_aug(tmp_path, rir, rir_shift=False)
    x = (rng.normal(size=1000) * 0.05).astype(np.float32)
    out = _reverberate(aug, torch.from_numpy(x)[None])[0].double().numpy()
    expected = np.convolve(x.astype(np.float64), rir.astype(np.float64))[:1000]
    expected *= np.linalg.norm(out) / np.linalg.norm(expected)
    np.testing.assert_allclose(out, expected, atol=1e-5 * np.abs(expected).max())


def test_multichannel_rir_gives_one_rir_per_channel(tmp_path):
    """Channel 0 is a delta and channel 1 a one-sample delay: two RIRs."""
    rir = np.zeros((4, 2), dtype=np.float32)
    rir[0, 0] = 1.0
    rir[1, 1] = 1.0
    path = _write(tmp_path / "rir2.wav", rir)
    aug = BatchWavAugment(rir_scp=_scp(tmp_path, "rir", path), rir_shift=False)
    aug.train()
    _, _, lengths = aug.bank("rir", torch.device("cpu"))
    assert lengths.tolist() == [4, 4]
    speech = torch.tensor([[1.0, 2.0, 3.0, 4.0]]) * 0.1
    torch.testing.assert_close(_reverberate(aug, speech, which=0), speech)
    shifted = _reverberate(aug, speech, which=1)[0]
    torch.testing.assert_close(shifted / shifted[1], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_rir_at_another_rate_is_resampled(tmp_path):
    """An 8 kHz impulse response of 400 samples for 16 kHz speech: about 800."""
    pytest.importorskip("librosa")
    rir = np.exp(-np.arange(400) / 50.0).astype(np.float32)
    aug = _rir_aug(tmp_path, rir, fs=8000)  # the module runs at 16 kHz
    _, _, lengths = aug.bank("rir", torch.device("cpu"))
    assert abs(int(lengths[0]) - 800) <= 2


def test_rir_apply_prob_zero_does_not_load_the_rirs(tmp_path, monkeypatch):
    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchWavAugment(
        noise_scp=scp, rir_scp=str(tmp_path / "missing.scp"), rir_apply_prob=0.0
    ).train()
    load = aug._load

    def no_rir(kind):
        assert kind != "rir", "the impulse responses were loaded"
        return load(kind)

    monkeypatch.setattr(aug, "_load", no_rir)
    speech = torch.randn(2, 4000) * 0.01
    out, _ = aug(speech, None)
    assert not torch.allclose(out, speech)  # the noise is still added
    assert "rir_apply" not in aug.draw(torch.tensor([4000, 4000]))


def test_rir_apply_prob_one_reverberates_every_row(tmp_path):
    aug = _rir_aug(tmp_path, [0.0, 1.0], rir_apply_prob=1.0, rir_shift=False)
    torch.manual_seed(0)
    assert bool(aug.draw(torch.full((64,), 100))["rir_apply"].all())
    speech = torch.rand(8, 100) * 0.1 + 0.05
    out, _ = aug(speech, None)
    assert torch.all(out[:, 0].abs() < 1e-6)  # every row is delayed by a sample


def test_draws_stay_on_the_host_except_the_offsets(tmp_path):
    """The apply flags are tested on the host: no wait for the device."""
    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchWavAugment(noise_scp=scp, rir_scp=None).train()
    draws = aug.draw(torch.tensor([4000, 3000]))
    for key in ("noise_apply", "noise_which", "snr", "noise_offset"):
        assert draws[key].device.type == "cpu", key


# --- rir_shift: aligned with the dry speech, as Kaldi's --shift-output ------


def test_rir_shift_is_the_default_and_undoes_a_pure_delay(tmp_path):
    """A delta delayed by 37 samples leaves the speech unchanged by default."""
    rir = np.zeros(200, dtype=np.float32)
    rir[37] = 1.0
    aug = _rir_aug(tmp_path, rir)
    assert aug.rir_shift
    speech = torch.from_numpy(np.random.default_rng(10).normal(size=(2, 3000)))
    speech = speech.float() * 0.05
    torch.testing.assert_close(_reverberate(aug, speech), speech, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("t", [1000, 6000])
def test_rir_shift_equals_kaldi_shift_output(tmp_path, t):
    """full convolution[peak : peak + t], peak = argmax |rir| (here negative).

    Kaldi's wav-reverberate --shift-output keeps these samples; the power is
    then scaled back to the input's.
    """
    rng = np.random.default_rng(11)
    rir = (rng.normal(size=4000) * np.exp(-np.arange(4000) / 700.0)).astype(np.float32)
    rir[:120] *= 0.01  # a quiet stretch before the direct path
    rir[120] = -3.0  # the direct path, negative
    aug = _rir_aug(tmp_path, rir)
    x = (rng.normal(size=t) * 0.05).astype(np.float32)
    out = _reverberate(aug, torch.from_numpy(x)[None])[0].double().numpy()
    full = np.convolve(x.astype(np.float64), rir.astype(np.float64))
    expected = full[120 : 120 + t]
    expected *= np.linalg.norm(out) / np.linalg.norm(expected)
    np.testing.assert_allclose(out, expected, atol=1e-5 * np.abs(expected).max())


def test_rir_shift_per_row_and_channel(tmp_path):
    """Each row is shifted by its own impulse response's peak."""
    rir = np.zeros((300, 2), dtype=np.float32)
    rir[5, 0] = 1.0  # channel 0: delay 5
    rir[250, 1] = 1.0  # channel 1: delay 250
    path = _write(tmp_path / "rir2.wav", rir)
    aug = BatchWavAugment(rir_scp=_scp(tmp_path, "rir", path)).train()
    assert aug._rir_peaks().tolist() == [5, 250]
    speech = torch.rand(2, 1000) * 0.1 + 0.05
    valid = torch.ones(2, 1000, dtype=torch.bool)
    draws = dict(
        rir_apply=torch.ones(2, dtype=torch.bool), rir_which=torch.tensor([0, 1])
    )
    torch.testing.assert_close(aug.apply(speech, valid, draws), speech)


def test_rir_shift_keeps_padding_zero_and_batch_equals_alone(tmp_path):
    rng = np.random.default_rng(12)
    rir = (rng.normal(size=2000) * np.exp(-np.arange(2000) / 300.0)).astype(np.float32)
    rir[:60] *= 0.01
    rir[60] = 2.0
    aug = _rir_aug(tmp_path, rir)
    lengths = [4000, 2500]
    speech = torch.zeros(2, 4000)
    for b, n in enumerate(lengths):
        speech[b, :n] = torch.from_numpy(rng.normal(size=n).astype(np.float32)) * 0.05
    valid = torch.arange(4000)[None] < torch.tensor(lengths)[:, None]
    draws = dict(
        rir_apply=torch.ones(2, dtype=torch.bool), rir_which=torch.tensor([0, 0])
    )
    batched = aug.apply(speech, valid, draws)
    one = dict(rir_apply=torch.ones(1, dtype=torch.bool), rir_which=torch.tensor([0]))
    alone = aug.apply(speech[1:, :2500], valid[1:, :2500], one)
    torch.testing.assert_close(batched[1, :2500], alone[0], rtol=1e-4, atol=1e-6)
    assert torch.all(batched[1, 2500:] == 0)


# --- snr_reference, noise_rir_apply_prob, context -----------------------------


def _decaying_rir(rng, n=3000, peak=40, tau=400.0):
    rir = (rng.normal(size=n) * np.exp(-np.arange(n) / tau)).astype(np.float32)
    rir[:peak] *= 0.01
    rir[peak] = 10.0  # the direct path, above every reflection
    return rir


def test_bad_options():
    with pytest.raises(ValueError):
        BatchWavAugment(noise_scp="n.scp", snr_reference="late")
    with pytest.raises(ValueError):
        BatchWavAugment(noise_scp="n.scp", noise_rir_apply_prob=0.5)  # no rir_scp
    with pytest.raises(ValueError):
        BatchWavAugment(noise_scp="n.scp", context=-1)


def test_early_snr_reference_measures_the_early_reverberation(tmp_path):
    """snr_reference=early: the noise is at the SNR of the early reverberation.

    The early part is the impulse response from 1 ms before its peak to 50 ms
    after it (Kaldi's ComputeEarlyReverbEnergy); the speech convolved with it,
    with the same gain as the reverberated speech, gives the reference power.
    """
    rng = np.random.default_rng(13)
    rir = _decaying_rir(rng, n=8000, tau=2000.0)  # a long tail beyond 50 ms
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", rir))
    noise_scp = write_noises(tmp_path, [(40000, 1)])
    t = 16000
    speech = torch.from_numpy(rng.normal(size=(1, t)).astype(np.float32)) * 0.05
    valid = torch.ones(1, t, dtype=torch.bool)
    draws = dict(
        rir_apply=torch.tensor([True]),
        rir_which=torch.tensor([0]),
        noise_apply=torch.tensor([True]),
        noise_which=torch.tensor([0]),
        noise_offset=torch.tensor([0]),
        snr=torch.tensor([10.0]),
    )
    outs = {}
    for ref in ("dry", "early"):
        aug = BatchWavAugment(
            noise_scp=noise_scp,
            rir_scp=rir_scp,
            noise_db_range="10",
            snr_reference=ref,
        )
        no_noise = dict(draws, noise_apply=torch.tensor([False]))
        reverberant = aug.apply(speech, valid, no_noise)
        noisy = aug.apply(speech, valid, draws)
        outs[ref] = (reverberant, noisy - reverberant)
        # the reverberated speech does not depend on the reference
    torch.testing.assert_close(outs["dry"][0], outs["early"][0])
    aug = BatchWavAugment(noise_scp=noise_scp, rir_scp=rir_scp)
    # the reference powers: the reverberant speech (dry) and its early part
    x = speech.double().numpy()[0]
    early = np.zeros_like(rir)
    early[40 - 16 : 40 + 800] = rir[40 - 16 : 40 + 800]
    full = np.convolve(x, rir.astype(np.float64))[40 : 40 + t]
    part = np.convolve(x, early.astype(np.float64))[40 : 40 + t]
    gain = np.linalg.norm(outs["dry"][0][0].double().numpy()) / np.linalg.norm(full)
    p_full = float(
        aug._non_silent_power(torch.from_numpy(full * gain)[None].float(), valid)
    )
    p_early = float(
        aug._non_silent_power(torch.from_numpy(part * gain)[None].float(), valid)
    )
    assert p_early < 0.9 * p_full  # the tail carries power
    for ref, p in (("dry", p_full), ("early", p_early)):
        noise_power = float(outs[ref][1].pow(2).mean())
        snr = 10 * np.log10(p / noise_power)
        assert abs(snr - 10.0) < 0.05, (ref, snr)


def test_early_reference_without_reverberation_is_the_dry_power(tmp_path):
    rng = np.random.default_rng(14)
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", _decaying_rir(rng)))
    noise_scp = write_noises(tmp_path, [(20000, 1)])
    speech = torch.from_numpy(rng.normal(size=(1, 8000)).astype(np.float32)) * 0.05
    valid = torch.ones(1, 8000, dtype=torch.bool)
    draws = dict(
        rir_apply=torch.tensor([False]),
        rir_which=torch.tensor([0]),
        noise_apply=torch.tensor([True]),
        noise_which=torch.tensor([0]),
        noise_offset=torch.tensor([5]),
        snr=torch.tensor([3.0]),
    )
    outs = [
        BatchWavAugment(noise_scp=noise_scp, rir_scp=rir_scp, snr_reference=r).apply(
            speech, valid, draws
        )
        for r in ("dry", "early")
    ]
    torch.testing.assert_close(outs[0], outs[1])


def test_noise_is_reverberated_by_its_own_rir(tmp_path):
    """noise_rir_apply_prob: the added noise is the noise convolved with its RIR."""
    rng = np.random.default_rng(15)
    rir = _decaying_rir(rng, n=2000)
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", rir))
    noise_scp = write_noises(tmp_path, [(30000, 1)])
    aug = BatchWavAugment(
        noise_scp=noise_scp,
        rir_scp=rir_scp,
        rir_apply_prob=0.0,
        noise_rir_apply_prob=1.0,
        noise_db_range="0",
    ).train()
    t = 6000
    speech = torch.from_numpy(rng.normal(size=(1, t)).astype(np.float32)) * 0.05
    valid = torch.ones(1, t, dtype=torch.bool)
    torch.manual_seed(0)
    draws = aug.draw(torch.tensor([t]))
    assert bool(draws["noise_rir_apply"].all())
    added = (aug.apply(speech, valid, draws) - speech)[0].double().numpy()
    bank, starts, _ = aug.bank("noise", torch.device("cpu"))
    o = int(draws["noise_offset"][0])
    dry = bank[o : o + t].double().numpy() / 32768
    expected = np.convolve(dry, rir.astype(np.float64))[40 : 40 + t]
    expected *= np.dot(added, expected) / np.dot(expected, expected)
    np.testing.assert_allclose(added, expected, atol=1e-4 * np.abs(expected).max())
    # SNR 0 dB against the reverberated noise
    p = float(aug._non_silent_power(speech, valid))
    assert abs(10 * np.log10(p / np.mean(added**2))) < 0.05


def test_noise_rir_probability_zero_keeps_the_noise_dry(tmp_path):
    rng = np.random.default_rng(16)
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", _decaying_rir(rng)))
    noise_scp = write_noises(tmp_path, [(30000, 1)])
    aug = BatchWavAugment(noise_scp=noise_scp, rir_scp=rir_scp, rir_apply_prob=0.0)
    assert "noise_rir_which" not in aug.draw(torch.tensor([4000]))


def test_context_is_removed_and_gives_the_reverberation_its_history(tmp_path):
    """With context c, the row is reverberated whole and its first c dropped.

    The lengths shrink by c, also in eval mode.
    """
    rng = np.random.default_rng(17)
    rir = _decaying_rir(rng, n=1500, peak=0)
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", rir))
    c, t = 1500, 4000
    aug = BatchWavAugment(rir_scp=rir_scp, rir_shift=False, context=c).train()
    speech = torch.from_numpy(rng.normal(size=(2, c + t)).astype(np.float32)) * 0.05
    out, lens = aug(speech, torch.tensor([c + t, c + t]))
    assert out.shape == (2, t) and lens.tolist() == [t, t]
    for b in range(2):
        full = np.convolve(speech[b].double().numpy(), rir.astype(np.float64))
        expected = full[c : c + t]
        got = out[b].double().numpy()
        expected *= np.dot(got, expected) / np.dot(expected, expected)
        np.testing.assert_allclose(got, expected, atol=1e-5 * np.abs(expected).max())
    # the window's first samples carry the tail of the context
    head = np.convolve(speech[0, c:].double().numpy(), rir.astype(np.float64))[:200]
    assert not np.allclose(
        out[0, :200].double().numpy(),
        head * (np.linalg.norm(out[0, :200].double().numpy()) / np.linalg.norm(head)),
        atol=1e-4,
    )
    aug.eval()
    out, lens = aug(speech, torch.tensor([c + t, c + 100]))
    torch.testing.assert_close(out, speech[:, c:])
    assert lens.tolist() == [t, 100]


@pytest.mark.parametrize("snr_reference", ["dry", "early"])
def test_context_equals_the_window_alone(tmp_path, snr_reference):
    """With context, the window gets what it gets alone with the same choices.

    The window starts with silence and then has speech, so the frames of the
    silence detection must start where the window starts; a loud context must
    not change the window's power, its SNR or its peak.
    """
    rng = np.random.default_rng(18)
    noise_scp = write_noises(tmp_path, [(60000, 1)])
    c, t = 2000, 9000
    window = np.zeros(t, dtype=np.float32)
    window[3000:] = rng.normal(size=t - 3000) * 0.05  # silence, then speech
    window = torch.from_numpy(window)[None]
    draws = dict(
        noise_apply=torch.tensor([True]),
        noise_which=torch.tensor([0]),
        noise_offset=torch.tensor([100]),
        snr=torch.tensor([5.0]),
    )
    alone = BatchWavAugment(noise_scp=noise_scp, snr_reference=snr_reference)
    expected = alone.apply(window, torch.ones(1, t, dtype=torch.bool), draws)
    aug = BatchWavAugment(noise_scp=noise_scp, context=c, snr_reference=snr_reference)
    # the noise of the whole row starts c samples earlier
    with_context = dict(draws, noise_offset=torch.tensor([100 - c + 60000]))
    for context in (torch.zeros(1, c), torch.full((1, c), 3.0)):
        row = torch.cat([context, window], 1)
        out = aug.apply(row, torch.ones(1, c + t, dtype=torch.bool), with_context)
        torch.testing.assert_close(out[:, c:], expected, rtol=1e-5, atol=1e-7)
        # the achieved SNR, against the window's power computed independently
        # by CommonPreprocessor's detect_non_silence on the window alone
        w = window.double().numpy()
        speech_power = (w[detect_non_silence(w)] ** 2).mean()
        added = (out[:, c:] - window).double().numpy()
        snr = 10 * np.log10(speech_power / (added**2).mean())
        assert abs(snr - 5.0) < 0.01, snr


def test_remove_context_checks_each_row(tmp_path):
    aug = BatchWavAugment(noise_scp=write_noises(tmp_path, [(4000, 1)]), context=100)
    speech, lengths = aug.remove_context(torch.zeros(2, 300), torch.tensor([300, 150]))
    assert speech.shape == (2, 200) and lengths.tolist() == [200, 50]
    with pytest.raises(RuntimeError):
        aug.remove_context(torch.zeros(2, 300), torch.tensor([300, 100]))
    assert aug.remove_context(torch.zeros(1, 300))[0].shape == (1, 200)
    no_context = BatchWavAugment(noise_scp=write_noises(tmp_path, [(4000, 1)]))
    x = torch.zeros(1, 300)
    assert no_context.remove_context(x)[0] is x


def test_context_longer_than_the_row_raises(tmp_path):
    aug = BatchWavAugment(noise_scp=write_noises(tmp_path, [(4000, 1)]), context=100)
    with pytest.raises(ValueError):
        aug.eval()(torch.zeros(1, 100), None)


@pytest.mark.parametrize("snr_reference", ["dry", "early"])
def test_rows_without_reverberation_are_not_convolved_and_rows_match_alone(
    tmp_path, snr_reference
):
    """Mixed flags: each row equals the row alone with its own flags.

    Only the rows that get reverberation (and the noise rows that get theirs)
    are convolved; the others keep the dry speech or the dry noise.
    """
    rng = np.random.default_rng(19)
    rir_scp = _scp(tmp_path, "rir", _write(tmp_path / "rir.wav", _decaying_rir(rng)))
    noise_scp = write_noises(tmp_path, [(40000, 1)])
    aug = BatchWavAugment(
        noise_scp=noise_scp,
        rir_scp=rir_scp,
        noise_rir_apply_prob=0.5,
        snr_reference=snr_reference,
    )
    t = 5000
    speech = torch.from_numpy(rng.normal(size=(4, t)).astype(np.float32)) * 0.05
    valid = torch.ones(4, t, dtype=torch.bool)
    draws = dict(
        rir_apply=torch.tensor([True, False, True, False]),
        rir_which=torch.zeros(4, dtype=torch.long),
        noise_apply=torch.tensor([True, True, False, True]),
        noise_which=torch.zeros(4, dtype=torch.long),
        noise_offset=torch.tensor([0, 100, 200, 300]),
        snr=torch.tensor([5.0, 10.0, 0.0, 3.0]),
        noise_rir_apply=torch.tensor([True, False, True, True]),
        noise_rir_which=torch.zeros(4, dtype=torch.long),
    )
    batched = aug.apply(speech, valid, draws)
    for b in range(4):
        one = {k: v[b : b + 1] for k, v in draws.items()}
        alone = aug.apply(speech[b : b + 1], valid[b : b + 1], one)
        torch.testing.assert_close(batched[b], alone[0], rtol=1e-4, atol=1e-6)
    # row 3 without its noise: no reverberation and no noise leave it unchanged
    quiet = {k: v[3:] for k, v in draws.items()}
    quiet["noise_apply"] = torch.tensor([False])
    torch.testing.assert_close(aug.apply(speech[3:], valid[3:], quiet), speech[3:])


# --- Other sampling rates -----------------------------------------------------

RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000]


@pytest.mark.parametrize("fs", [8000, 24000, 32000, 48000])
@pytest.mark.parametrize("noise_factor", [2.5, 0.3])  # longer / shorter noise
def test_same_result_as_common_preprocessor_at_other_rates(
    tmp_path, monkeypatch, fs, noise_factor
):
    """At 8, 24, 32 and 48 kHz the module equals CommonPreprocessor.

    The speech, the impulse response and the noise are all at fs (neither
    resamples), with reverberation and noise together and rir_shift=False.
    """
    rng = np.random.default_rng(fs)
    length = fs  # 1 s
    wav = (rng.normal(size=length) * 0.05).astype(np.float32)
    wav[length // 3 : length // 2] *= 0.001
    n_rir = fs // 4
    rir = (rng.normal(size=n_rir) * np.exp(-np.arange(n_rir) / (fs / 25))).astype(
        np.float32
    )
    rir[0] = 1.0
    noise_len = int(length * noise_factor)
    noise = rng.uniform(-0.3, 0.3, size=noise_len).astype(np.float32)
    rir_path = _write(tmp_path / "rir.wav", rir, fs=fs)
    noise_path = _write(tmp_path / "noise.wav", noise, fs=fs)
    snr = 5.0
    cpu_offset = noise_len // 3 if noise_len > length else noise_len // 5
    expected = _cpu_reference(
        tmp_path, monkeypatch, wav, rir_path, noise_path, cpu_offset, snr
    )
    aug = BatchWavAugment(
        noise_scp=_scp(tmp_path, "noise", noise_path),
        rir_scp=_scp(tmp_path, "rir", rir_path),
        noise_db_range=str(snr),
        rir_shift=False,
        fs=fs,
    ).train()
    gpu_offset = cpu_offset if noise_len >= length else (-cpu_offset) % noise_len
    draws = dict(
        rir_apply=torch.tensor([True]),
        rir_which=torch.tensor([0]),
        noise_apply=torch.tensor([True]),
        noise_which=torch.tensor([0]),
        noise_offset=torch.tensor([gpu_offset]),
        snr=torch.tensor([snr]),
    )
    speech = torch.from_numpy(wav)[None]
    out = aug.apply(speech, torch.ones(1, length, dtype=torch.bool), draws)
    out = out[0].double().numpy()
    assert np.abs(expected - wav).max() > 0.01
    err = np.abs(out - expected).max() / np.abs(expected).max()
    assert err < 1e-4, err


@pytest.mark.parametrize("fs", RATES)
def test_early_part_is_kaldis_window_at_each_rate(tmp_path, fs):
    """The early impulse response is Kaldi's window at each rate.

    It runs from int(peak - 0.001 fs) to int(peak + 0.05 fs), exclusive, also
    where 1 ms is not a whole number of samples (22.05 and 44.1 kHz).
    """
    rng = np.random.default_rng(fs + 1)
    n, peak = int(0.2 * fs), int(0.01 * fs) + 3
    rir = (rng.normal(size=n) * 0.1).astype(np.float32)
    rir[peak] = 5.0
    aug = BatchWavAugment(
        rir_scp=_scp(tmp_path, "rir", _write(tmp_path / "rir.wav", rir, fs=fs)),
        fs=fs,
    )
    t = int(0.3 * fs)
    x = rng.normal(size=t).astype(np.float32) * 0.05
    full, early = aug._reverberate(torch.from_numpy(x)[None], torch.tensor([0]), True)
    lo, hi = int(peak - 0.001 * fs), int(peak + 0.05 * fs)
    part = np.zeros(n)
    part[lo:hi] = rir[lo:hi]
    ref_full = np.convolve(x.astype(np.float64), rir.astype(np.float64))
    ref_early = np.convolve(x.astype(np.float64), part)
    scale = np.abs(ref_full).max()
    np.testing.assert_allclose(
        full[0].double().numpy(), ref_full[peak : peak + t], atol=2e-5 * scale
    )
    np.testing.assert_allclose(
        early[0].double().numpy(), ref_early[peak : peak + t], atol=2e-5 * scale
    )


@pytest.mark.parametrize(
    "src, dst", [(8000, 16000), (48000, 16000), (16000, 24000), (44100, 32000)]
)
def test_resampled_banks_equal_librosa(tmp_path, src, dst):
    """Noise and impulse responses at another rate equal librosa.resample."""
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(src + dst)
    rir = (
        rng.normal(size=src // 5) * np.exp(-np.arange(src // 5) / (src / 50))
    ).astype(np.float32)
    rir[0] = 1.0
    noise = rng.uniform(-0.5, 0.5, size=src).astype(np.float32)
    aug = BatchWavAugment(
        noise_scp=_scp(tmp_path, "noise", _write(tmp_path / "n.wav", noise, fs=src)),
        rir_scp=_scp(tmp_path, "rir", _write(tmp_path / "r.wav", rir, fs=src)),
        fs=dst,
    )
    want_rir = librosa.resample(rir, orig_sr=src, target_sr=dst)
    got_rir = aug.bank("rir", torch.device("cpu"))[0].numpy()
    np.testing.assert_allclose(got_rir, want_rir, rtol=1e-5, atol=1e-6)
    want_noise = librosa.resample(noise, orig_sr=src, target_sr=dst)
    want_noise = np.round(want_noise * (32767.0 / np.abs(want_noise).max()))
    got_noise = aug.bank("noise", torch.device("cpu"))[0].numpy().astype(np.float64)
    assert got_noise.shape == want_noise.shape
    assert np.abs(got_noise - want_noise).max() <= 1  # int16 rounding


def test_host_and_device_impulse_responses_are_the_same(tmp_path):
    """The host batch (slice copies) equals the device bank's gather."""
    rng = np.random.default_rng(20)
    rirs = np.zeros((900, 3), dtype=np.float32)
    rirs[:, 0] = rng.normal(size=900)
    rirs[:300, 1] = rng.normal(size=300)  # shorter: zero after its end
    rirs[:600, 2] = rng.normal(size=600)
    path = _write(tmp_path / "rirs.wav", rirs)
    lengths = []
    outs = []
    for keep in (False, True):
        aug = BatchWavAugment(rir_scp=_scp(tmp_path, "rir", path), keep_on_device=keep)
        lengths.append(aug.bank("rir", torch.device("cpu"))[2].tolist())
        outs.append(
            aug._impulse_responses(torch.tensor([2, 0, 1, 2]), torch.device("cpu"))
        )
    torch.testing.assert_close(outs[0], outs[1])
    assert outs[0].shape == (4, max(lengths[0]))
