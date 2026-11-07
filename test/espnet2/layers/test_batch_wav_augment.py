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
    draws = aug.draw(3, 16000, torch.device("cpu"))
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
