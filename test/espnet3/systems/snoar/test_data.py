import json

import numpy as np
import pytest
import soundfile as sf

from espnet3.systems.snoar.src.data.fisher import FisherRemixDataset, read_noise_list
from espnet3.systems.snoar.src.data.manifest import (
    blob_path,
    crossfade_concat,
    describe_recordings,
    list_audio,
    read_audio,
    read_manifest,
    write_manifest,
)
from espnet3.systems.snoar.src.data.noisy import NoisyCropDataset, UtteranceDataset
from espnet3.systems.snoar.src.data.vad import (
    detect_energy_speech,
    find_gaps,
    harvest_noise,
    read_rttm,
)

from .conftest import FS


def _manifest(wav_dir, tmp_path):
    rows = describe_recordings(list_audio(wav_dir))
    path = tmp_path / "rec.jsonl"
    write_manifest(rows, path)
    return rows, path


def test_noisy_crops_with_noise_pool(wav_dir, tmp_path):
    rows, path = _manifest(wav_dir, tmp_path)
    pool = harvest_noise(rows, vad="energy", fs=FS)
    assert pool and all(r["end"] - r["start"] >= int(0.3 * FS) for r in pool)
    write_manifest(pool, tmp_path / "pool.jsonl")
    ds = NoisyCropDataset(
        str(path),
        segment_seconds=1.0,
        noise_pool=str(tmp_path / "pool.jsonl"),
        noise_level="snr",
        noise_snr=[5, 5],
    )
    item = ds[0]
    assert item["speech_mix"].shape == (FS,) and item["noise2"].shape == (FS,)
    snr = 10 * np.log10(np.sum(item["speech_mix"] ** 2) / np.sum(item["noise2"] ** 2))
    assert snr == pytest.approx(5.0, abs=0.01)
    det = NoisyCropDataset(str(path), segment_seconds=1.0, deterministic=True)
    assert np.array_equal(det[1]["speech_mix"], det[1]["speech_mix"])


def test_utterances_and_manifest_errors(wav_dir, tmp_path):
    _, path = _manifest(wav_dir, tmp_path)
    utt = UtteranceDataset(str(path))[0]
    assert utt["utt_id"] == "rec0" and len(utt["speech"]) == int(1.5 * FS)
    with pytest.raises(ValueError, match="no `speech_ref`"):
        NoisyCropDataset(str(path), with_refs=True)
    with pytest.raises(ValueError, match="noise_level"):
        NoisyCropDataset(str(path), noise_level="loud")
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="expects 8000 Hz"):
        UtteranceDataset(str(path), fs=8000)[0]


def test_gaps_and_labels(tmp_path):
    assert find_gaps([(1.0, 2.0)], 4.0) == [(0.0, 0.9), (2.1, 4.0)]
    assert find_gaps([(0.0, 4.0)], 4.0) == []
    rttm = tmp_path / "a.rttm"
    rttm.write_text("SPEAKER rec0 1 0.50 0.40 <NA> <NA> spk0 <NA> <NA>\n")
    assert read_rttm(rttm) == {"rec0": [(0.5, 0.9)]}
    with pytest.raises(ValueError, match="no recording id"):
        harvest_noise([{"id": "other", "audio": "x", "num_samples": 10}], vad=str(rttm))
    t = np.arange(FS) / FS
    x = np.where((t > 0.3) & (t < 0.6), np.sin(2 * np.pi * 200 * t), 0.001 * np.cos(t))
    ((s0, s1),) = detect_energy_speech(x.astype(np.float32), FS)
    assert s0 == pytest.approx(0.3, abs=0.03) and s1 == pytest.approx(0.6, abs=0.04)


def test_crossfade_fills_length():
    out = crossfade_concat([np.ones(300), np.ones(300), np.ones(300)], 800, fade=40)
    assert (
        out.shape == (800,) and np.allclose(out[:300], 1) and np.all(out > 0.99 - 1e-6)
    )


def _fake_cache(tmp_path, n, sup=None, dry=True):
    np.save(tmp_path / "speech.000000.npy", (0.1 * np.ones(n)).astype(np.float16))
    np.save(tmp_path / "noise.000000.npy", (0.01 * np.ones(n)).astype(np.float16))
    row = {"id": "m", "offset": 0, "length": n, "ps": 0.01, "pn": 1e-4}
    if dry:
        np.save(tmp_path / "dry.000000.npy", (-0.2 * np.ones(n)).astype(np.float16))
        row["ps_dry"] = 0.04
    if sup is not None:
        row["sup"] = sup
    (tmp_path / "index.000000.jsonl").write_text(json.dumps(row) + "\n")


def test_fisher_remix(tmp_path):
    _fake_cache(tmp_path, 3 * FS)
    ds = FisherRemixDataset(
        str(tmp_path),
        [0],
        segment_seconds=1.0,
        snr_range=[10, 10],
        gain_range=[0, 0],
        num_items=2,
    )
    it = ds[0]
    assert np.allclose(it["speech_mix"], it["speech_ref"] + it["noise_ref"])
    snr = 10 * np.log10(np.mean(it["speech_ref"] ** 2) / np.mean(it["noise_ref"] ** 2))
    assert snr == pytest.approx(10.0, abs=0.1)
    with pytest.raises(FileNotFoundError, match="build_fisher_cache"):
        FisherRemixDataset(str(tmp_path), [7])


def test_fisher_dry_condition_and_noise_files(tmp_path):
    _fake_cache(tmp_path, 3 * FS)
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    paths = []
    for i in range(40):
        f = noise_dir / f"n{i}.wav"
        sf.write(f, 0.05 * np.random.default_rng(i).standard_normal(2 * FS), FS)
        paths.append(str(f))
    lst = tmp_path / "noise.txt"
    lst.write_text("\n".join(paths) + "\n")
    tr = read_noise_list(str(lst), "train", valid_fraction=0.25)
    va = read_noise_list(str(lst), "valid", valid_fraction=0.25)
    assert tr and va and not set(tr) & set(va) and len(tr) + len(va) == 40
    ds = FisherRemixDataset(
        str(tmp_path),
        [0],
        segment_seconds=1.0,
        conditions={"dry": 1.0},
        noise_list=str(lst),
        noise_split="valid",
        noise_valid_fraction=0.25,
        snr_range=[5, 5],
        gain_range=[0, 0],
        num_items=4,
        deterministic=True,
    )
    it = ds[1]
    assert np.allclose(it["speech_ref"], -0.2, atol=1e-3)
    snr = 10 * np.log10(0.04 / np.mean(it["noise_ref"] ** 2))
    assert snr == pytest.approx(5.0, abs=0.1)
    with pytest.raises(ValueError, match="conditions"):
        FisherRemixDataset(str(tmp_path), [0], conditions={"wet": 1.0})


def test_fisher_crops_land_on_speech(tmp_path):
    _fake_cache(tmp_path, 10 * FS, sup=[[8.0, 10.0]])
    ds = FisherRemixDataset(
        str(tmp_path), [0], segment_seconds=1.0, min_active=0.5, num_items=8
    )
    mask = ds._mask(ds.meetings[0])
    for i in range(8):
        _, st = ds._draw_crop(np.random.default_rng(i))
        assert mask[st : st + FS].mean() >= 0.5


def test_build_fisher_cache_adds_dry(tmp_path, monkeypatch):
    import sys
    import types

    from espnet3.systems.snoar.src.data import fisher

    class Sup:
        def __init__(self, a, b):
            self.start, self.end = a, b

    class Cut:
        def __init__(self, cid, n):
            self.id, self.n = cid, n
            self.supervisions = [Sup(0.0, n / FS / 2)]

        def load_custom(self, name):
            v = {"spk_reverb": 0.3, "spk_dry": 0.1, "noise": 0.01}[name]
            return np.full((2, self.n), v, dtype=np.float32)

    cuts = [Cut("a", FS), Cut("b", 2 * FS)]
    fake = types.SimpleNamespace(
        CutSet=types.SimpleNamespace(from_shar=lambda fields: iter(cuts))
    )
    monkeypatch.setitem(sys.modules, "lhotse", fake)
    fields = {"cuts": ["c"], "spk_reverb": ["r"], "spk_dry": ["d"], "noise": ["n"]}
    out = tmp_path / "cache"
    out.mkdir()
    fisher._cache_shard((0, fields, str(out), ("speech", "noise"), FS))
    first = [json.loads(line) for line in open(out / "index.000000.jsonl")]
    assert "ps_dry" not in first[0] and first[1]["offset"] == FS
    msg = fisher._cache_shard((0, fields, str(out), ("speech", "dry", "noise"), FS))
    assert "['dry']" in msg
    rows = [json.loads(line) for line in open(out / "index.000000.jsonl")]
    assert rows[0]["ps"] == first[0]["ps"]
    assert rows[0]["ps_dry"] == pytest.approx(0.04, rel=1e-3)  # (2 * 0.1) ** 2
    dry = np.load(out / "dry.000000.npy")
    assert dry.shape == (3 * FS,) and np.allclose(dry, 0.2, atol=1e-3)
    assert "complete" in fisher._cache_shard((0, fields, str(out), ("dry",), FS))
    with pytest.raises(ValueError, match="unknown cache kinds"):
        fisher.build_fisher_cache(str(tmp_path), str(out), kinds=("wet",))


def test_read_audio_from_blob_byte_range(tmp_path):
    rng = np.random.default_rng(0)
    clips = [rng.uniform(-0.5, 0.5, n).astype(np.float32) for n in (1600, 2400)]
    blob, ranges = tmp_path / "blob_0.bin", []
    with open(blob, "wb") as f:
        for i, c in enumerate(clips):
            one = tmp_path / f"c{i}.flac"
            sf.write(one, c, FS, subtype="PCM_16")
            data = one.read_bytes()
            ranges.append((f.tell(), f.tell() + len(data)))
            f.write(data)
    for c, (s0, s1) in zip(clips, ranges):
        x = read_audio(blob_path(blob, s0, s1), 100, 900, fs=FS)
        np.testing.assert_allclose(x, c[100:900], atol=1 / 2**14)


def _fake_omniio(root, name, clips):
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = root / name
    d.mkdir(parents=True)
    blob, rows = d / "blob_0.bin", []
    with open(blob, "wb") as f:
        for i, (x, sr) in enumerate(clips):
            buf = __import__("io").BytesIO()
            sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
            s0 = f.tell()
            f.write(buf.getvalue())
            rows.append(
                dict(
                    id=f"{name}{i}",
                    sample_rate=sr,
                    samples=len(x) if x.ndim == 1 else x.shape[0],
                    start_byte=s0,
                    end_byte=f.tell(),
                    path=str(blob),
                )
            )
    pq.write_table(pa.Table.from_pylist(rows), d / "metadata.parquet")


def test_urgent_mix_from_omniio_store(tmp_path):
    from espnet3.systems.snoar.src.data.urgent import UrgentMixDataset

    rng = np.random.default_rng(0)

    def tone(sr, sec):
        return (0.3 * np.sin(np.arange(int(sr * sec)) * 0.05)).astype(np.float32)

    _fake_omniio(
        tmp_path, "speech_a", [(tone(48000, 2.0), 48000), (tone(22050, 1.5), 22050)]
    )
    _fake_omniio(
        tmp_path,
        "noise_a",
        [(rng.uniform(-0.3, 0.3, (32000, 2)).astype(np.float32), 32000)],
    )
    rir = np.zeros(800, dtype=np.float32)
    rir[0], rir[400] = 1.0, 0.5
    _fake_omniio(tmp_path, "rir_a", [(rir, 16000)])
    ds = UrgentMixDataset(
        str(tmp_path),
        split=None,
        speech_subsets=["speech_a"],
        noise_subsets=["noise_a"],
        rir_subsets=["rir_a"],
        segment_seconds=1.0,
        reverb_prob=0.5,
        snr_range=[5.0, 5.0],
        gain_range=[0.0, 0.0],
        num_items=8,
        deterministic=True,
    )
    for i in range(8):
        it = ds[i]
        s, n = it["speech_ref"], it["noise_ref"]
        assert it["speech_mix"].shape == (FS,)
        np.testing.assert_allclose(it["speech_mix"], s + n, atol=1e-6)
        snr = 10 * np.log10(
            np.mean(s.astype(np.float64) ** 2) / np.mean(n.astype(np.float64) ** 2)
        )
        assert abs(snr - 5.0) < 0.05
    np.testing.assert_array_equal(ds[2]["speech_mix"], ds[2]["speech_mix"])
    with pytest.raises(ValueError):
        UrgentMixDataset(str(tmp_path), split="bogus")

    # an allow-list that excludes every noise_a file leaves only the extra file
    allow = tmp_path / "allow.txt"
    allow.write_text("noise_a/not_there\n")
    extra = tmp_path / "extra.wav"
    sf.write(extra, rng.uniform(-0.1, 0.1, 8000).astype(np.float32), 8000)
    (tmp_path / "extra.txt").write_text(f"{extra}\n")
    kw = dict(
        split=None,
        speech_subsets=["speech_a"],
        noise_subsets=["noise_a"],
        rir_subsets=["rir_a"],
        segment_seconds=1.0,
        num_items=2,
        deterministic=True,
    )
    ds2 = UrgentMixDataset(
        str(tmp_path),
        noise_allow_list=str(allow),
        noise_allow_subsets=["noise_a"],
        extra_noise_list=str(tmp_path / "extra.txt"),
        **kw,
    )
    assert len(ds2.noise) == 1 and ds2.noise.blobs == [str(extra)]
    assert ds2[0]["noise_ref"].shape == (FS,)
    with pytest.raises(ValueError):
        UrgentMixDataset(
            str(tmp_path),
            noise_allow_list=str(allow),
            noise_allow_subsets=["noise_a"],
            **kw,
        )
