"""Tests of rotating noise/music banks (espnet2.layers.rotating_bank, BatchWavAugment)."""

import numpy as np
import pytest
import soundfile
import torch

from espnet2.layers.batch_wav_augment import BatchWavAugment
from espnet2.layers.rotating_bank import Pool, parse_caps

HEAD = "id\tpath\tkind\tsource\tlicence\tduration_s\tflagged\n"


def write_pool(tmp_path, rows, name="pool.tsv"):
    p = tmp_path / name
    p.write_text(HEAD + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return str(p)


def wav(tmp_path, name, seconds, fs=16000, seed=0):
    x = (
        np.random.default_rng(seed)
        .standard_normal(int(seconds * fs))
        .astype(np.float32)
        * 0.1
    )
    path = tmp_path / f"{name}.wav"
    soundfile.write(path, x, fs)
    return str(path)


def test_parse_caps():
    assert parse_caps("noise=150,babble=50") == {"noise": 150.0, "babble": 50.0}
    assert parse_caps({"music": 75}) == {"music": 75.0}


def big_pool(tmp_path):
    rows = [(f"a{i}", "x", "noise", "big", "CC0-1.0", 300, 0) for i in range(100)]
    rows += [(f"b{i}", "x", "noise", "small", "CC-BY-4.0", 300, 0) for i in range(3)]
    rows += [(f"c{i}", "x", "babble", "crowd", "CC-BY-4.0", 300, 1) for i in range(20)]
    return write_pool(tmp_path, rows)


def test_subset_respects_caps_and_draws_sources_in_turn(tmp_path):
    pool = Pool(big_pool(tmp_path), "noise=2,babble=0.5")
    sub = pool.draw((0, 0, 0))
    hours = {
        k: sum(r.duration_s for r in sub if r.kind == k) / 3600
        for k in ("noise", "babble")
    }
    # at the cap, overshooting by at most one file
    assert 2 <= hours["noise"] < 2 + 300 / 3600 + 1e-9
    assert 0.5 <= hours["babble"] < 0.5 + 300 / 3600 + 1e-9
    # the small source is drawn in turn, not crowded out by the big one
    assert sum(r.source == "small" for r in sub) == 3


def test_same_seed_same_subset_other_seed_other_subset(tmp_path):
    pool = Pool(big_pool(tmp_path), "noise=2,babble=0.5")
    a = [r.id for r in pool.draw((7, 3, 1))]
    assert a == [
        r.id for r in Pool(big_pool(tmp_path), "noise=2,babble=0.5").draw((7, 3, 1))
    ]
    assert a != [r.id for r in pool.draw((7, 4, 1))]
    assert a != [r.id for r in pool.draw((7, 3, 2))]  # another rank


@pytest.mark.parametrize("kind", ["noise", "music", "human_nonspeech"])
def test_a_flagged_noise_or_music_row_is_refused(tmp_path, kind):
    rows = [("n1", "x", kind, "s", "CC0-1.0", 30, 1)]
    with pytest.raises(ValueError, match="flagged"):
        Pool(write_pool(tmp_path, rows), {kind: 1})


def test_kind_without_cap_and_missing_columns_are_refused(tmp_path):
    rows = [
        ("n1", "x", "noise", "s", "CC0-1.0", 30, 0),
        ("b1", "x", "babble", "s", "CC0-1.0", 30, 1),
    ]
    with pytest.raises(ValueError, match="no cap"):
        Pool(write_pool(tmp_path, rows), "noise=1")
    p = tmp_path / "bad.tsv"
    p.write_text("id\tpath\n")
    with pytest.raises(ValueError, match="no column"):
        Pool(str(p), "noise=1")


def wav_pool(tmp_path, n=12, seconds=2.0):
    rows = []
    for i in range(n):
        src = "s1" if i % 3 else "s2"
        rows.append(
            (
                f"n{i}",
                wav(tmp_path, f"n{i}", seconds, seed=i),
                "noise",
                src,
                "CC0-1.0",
                seconds,
                0,
            )
        )
    return write_pool(tmp_path, rows)


def rotating(tmp_path, **kw):
    opts = dict(
        noise_pool=wav_pool(tmp_path),
        noise_pool_hours="noise=0.002",
        noise_apply_prob=1.0,
        noise_db_range="0_10",
        rotate_every_steps=2,
        rotate_seed=3,
        rotate_log=str(tmp_path / "rot"),
    )
    opts.update(kw)
    return BatchWavAugment(**opts).train()


def test_rotation_swaps_every_n_steps_keeps_shapes_and_logs(tmp_path):
    torch.manual_seed(0)
    aug = rotating(tmp_path)
    speech = torch.randn(2, 8000) * 0.1
    banks = []
    for step in range(5):
        out, _ = aug(speech, None)
        assert out.shape == speech.shape and out.dtype == speech.dtype
        bank, starts, lengths = aug.bank("noise", torch.device("cpu"))
        assert bank.dtype == torch.int16 and bank.dim() == 1
        assert len(starts) == len(lengths) and int(lengths.sum()) == bank.numel()
        banks.append([r for r in aug._seen["noise"]])
    # subsets 0, 1, 2 at steps 0, 2, 4
    assert [s["k"] for s in aug.rotate_stats] == [0, 1, 2]
    assert int(aug.rotate_step) == 5
    # each subset at the cap: 0.002 h = 7.2 s, i.e. 4 files of 2 s
    assert all(abs(s["noise_hours"] - 8 / 3600) < 1e-9 for s in aug.rotate_stats)
    assert (
        aug.rotate_stats[-1]["noise_coverage"] > aug.rotate_stats[0]["noise_coverage"]
    )
    log = (tmp_path / "rot.rank0.tsv").read_text().splitlines()
    assert log[0].startswith("k\tstep\trank") and len(log) > 3
    # only the current subset is cached: no second bank alongside it
    assert sorted(k for k in aug._cache if k[0] == "noise") == [
        ("noise", "cpu"),
        ("noise", "host"),
    ]


def test_rotation_is_deterministic_and_resumes(tmp_path):
    speech = torch.randn(2, 8000) * 0.1

    def bank_at(aug, steps):
        for _ in range(steps):
            aug(speech, None)
        return aug.bank("noise", torch.device("cpu"))[0].clone()

    a = bank_at(rotating(tmp_path), 5)  # subset 2
    b = bank_at(rotating(tmp_path), 5)
    torch.testing.assert_close(a, b)
    # a module resumed at step 4 (the counter is in the state dict) starts with subset 2
    first = rotating(tmp_path)
    for _ in range(4):
        first(speech, None)
    resumed = rotating(tmp_path)
    resumed.load_state_dict(first.state_dict())
    assert int(resumed.rotate_step) == 4
    torch.testing.assert_close(bank_at(resumed, 1), a)
    assert resumed.rotate_stats[0]["k"] == 2


def test_eval_mode_reads_no_pool_and_options_are_checked(tmp_path):
    aug = BatchWavAugment(
        noise_pool=str(tmp_path / "missing.tsv"),
        noise_pool_hours="noise=1",
        rotate_every_steps=10,
    ).eval()
    x = torch.randn(1, 4000)
    torch.testing.assert_close(aug(x, None)[0], x)
    with pytest.raises(ValueError):
        BatchWavAugment(
            noise_pool="p.tsv", noise_pool_hours="noise=1"
        )  # no rotation step
    with pytest.raises(ValueError):
        BatchWavAugment(noise_pool="p.tsv", noise_scp="n.scp", rotate_every_steps=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_swap_keeps_the_device(tmp_path):
    aug = rotating(tmp_path, keep_on_device=True).cuda()
    speech = torch.randn(2, 8000, device="cuda") * 0.1
    for _ in range(3):
        out, _ = aug(speech, None)
        assert out.device.type == "cuda"
    bank = aug.bank("noise", speech.device)
    assert all(t.device.type == "cuda" for t in bank)
    assert sum(k[0] == "noise" and k[1] != "host" for k in aug._cache) == 1
    # with keep_on_device the host keeps only starts and lengths
    host = aug._cache[("noise", "host")]
    assert host[0].numel() == 0 and int(host[2].sum()) == bank[0].numel()
