import numpy as np
import pytest
import soundfile
import torch

from espnet2.layers.batch_noise_augment import BatchNoiseAugment
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
    aug = BatchNoiseAugment(scp, noise_apply_prob=1.0).eval()
    out, out_lengths = aug(speech, lengths)
    assert out is speech and out_lengths is lengths
    assert aug._cache == {}  # never loaded outside training

    aug = BatchNoiseAugment(scp, noise_apply_prob=0.0).train()
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
    aug = BatchNoiseAugment(scp, noise_apply_prob=1.0, noise_db_range="0_10").train()
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
    aug = BatchNoiseAugment(scp, noise_db_range="0").train()
    speech = torch.full((1, 3500), 0.01)
    out, _ = aug(speech, torch.tensor([3500]))
    added = (out - speech)[0]
    # period 1000: the added noise repeats
    torch.testing.assert_close(added[:1000], added[1000:2000])
    torch.testing.assert_close(added[1000:2000], added[2000:3000])


def test_multichannel_noise_gives_one_noise_per_channel(tmp_path):
    scp = write_noises(tmp_path, [(5000, 3), (4000, 1)])
    aug = BatchNoiseAugment(scp).train()
    aug(torch.randn(1, 2000) * 0.01, None)
    bank, starts, lengths = aug.noise_bank(torch.device("cpu"))
    assert lengths.tolist() == [5000, 5000, 5000, 4000]
    assert starts.tolist() == [0, 5000, 10000, 15000]
    assert bank.dtype == torch.int16 and bank.numel() == 19000


@pytest.mark.parametrize("t", [800, 5000, 16000])
def test_non_silent_power_matches_detect_non_silence(tmp_path, t):
    scp = write_noises(tmp_path, [(1000, 1)])
    aug = BatchNoiseAugment(scp)
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
    aug = BatchNoiseAugment(scp, noise_db_range="-5").train()
    speech = torch.randn(2, 4000) * 0.9
    out, _ = aug(speech, torch.tensor([4000, 4000]))
    assert out.abs().max() <= 1.0 + 1e-6


def test_noise_is_not_in_the_state_dict(tmp_path):
    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchNoiseAugment(scp).train()
    aug(torch.randn(1, 2000) * 0.01, None)
    assert "cpu" in aug._cache
    assert len(aug.state_dict()) == 0
    # not a buffer: DistributedDataParallel does not broadcast it
    assert list(aug.buffers()) == []


def test_bad_db_range(tmp_path):
    with pytest.raises(ValueError):
        BatchNoiseAugment("unused.scp", noise_db_range="10_0")


@pytest.mark.parametrize("lengths", [[16000, 9000, 5000, 1500, 700], [3000, 2999]])
def test_non_silent_power_of_padded_rows(tmp_path, lengths):
    """Each padded row: the power of the utterance alone (detect_non_silence)."""
    scp = write_noises(tmp_path, [(1000, 1)])
    aug = BatchNoiseAugment(scp)
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
    aug = BatchNoiseAugment(str(tmp_path / "noise.scp")).train()
    bank, _, _ = aug.noise_bank(torch.device("cpu"))
    assert int(bank.abs().max()) == 32767  # peak-normalised, not rounded to 0
    corr = np.corrcoef(bank.numpy().astype(np.float64), wav)[0, 1]
    assert corr > 0.9999


def test_replicas_share_the_loaded_noise(tmp_path):
    """A shallow copy (as DataParallel replicates) reuses and fills the cache."""
    import copy

    scp = write_noises(tmp_path, [(8000, 1)])
    aug = BatchNoiseAugment(scp).train()
    replica = copy.copy(aug)
    replica(torch.randn(1, 2000) * 0.01, None)
    assert "cpu" in aug._cache  # the original sees what the replica loaded


def test_long_noise_uses_every_offset(tmp_path):
    """A noise one sample longer than the batch: both offsets 0 and 1 occur."""
    rng = np.random.default_rng(4)
    t = 64
    wav = (rng.uniform(-0.5, 0.5, size=t + 1) * 32767).astype(np.int16)
    soundfile.write(tmp_path / "n.wav", wav, 16000, subtype="PCM_16")
    (tmp_path / "noise.scp").write_text(f"n {tmp_path / 'n.wav'}\n")
    aug = BatchNoiseAugment(str(tmp_path / "noise.scp"), noise_db_range="0").train()
    torch.manual_seed(0)
    speech = torch.full((200, t), 0.01)
    added = aug(speech, None)[0] - speech
    bank, _, _ = aug.noise_bank(torch.device("cpu"))
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
    aug = BatchNoiseAugment(scp).train()
    calls = []
    load = aug._load

    def slow_load():
        calls.append(threading.get_ident())
        time.sleep(0.2)  # long enough for every thread to arrive meanwhile
        return load()

    monkeypatch.setattr(aug, "_load", slow_load)
    replicas = [copy.copy(aug) for _ in range(4)]
    results = [None] * len(replicas)

    def run(i):
        results[i] = replicas[i].noise_bank(torch.device("cpu"))

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
    aug = BatchNoiseAugment(scp).train()
    aug(torch.randn(1, 2000) * 0.01, None)
    clone = copy.deepcopy(aug)
    out, _ = clone(torch.randn(1, 2000) * 0.01, None)
    assert out.shape == (1, 2000)
