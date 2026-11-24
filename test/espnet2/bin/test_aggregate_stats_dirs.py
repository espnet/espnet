from argparse import ArgumentParser

import numpy as np
import pytest

from espnet2.bin.aggregate_stats_dirs import get_parser, main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


def _write_stats_dir(path, keys, feats):
    """Stats dir of collect_stats for `keys`; `feats[k]` is the feature of key k."""
    for mode in ["train", "valid"]:
        d = path / mode
        (d / "collect_feats").mkdir(parents=True)
        (d / "batch_keys").write_text("speech\n")
        (d / "stats_keys").write_text("feats\n")
        (d / "speech_shape").write_text("".join(f"{k} {len(feats[k])}\n" for k in keys))
        (d / "collect_feats" / "feats.scp").write_text(
            "".join(f"{k} {path.name}/feats.ark:{i}\n" for i, k in enumerate(keys))
        )
        x = np.concatenate([feats[k] for k in keys])
        np.savez(
            d / "feats_stats.npz", count=len(x), sum=x.sum(0), sum_square=(x**2).sum(0)
        )


@pytest.mark.parametrize(
    "split",
    [[["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]], [["d", "b"], ["c", "a"]]],
)
def test_aggregate_is_independent_of_the_split(tmp_path, split):
    rng = np.random.default_rng(0)
    feats = {k: rng.standard_normal((n, 3)) for k, n in zip("abcd", [5, 7, 7, 2])}
    input_dirs = []
    for i, keys in enumerate(split):
        _write_stats_dir(tmp_path / f"stats.{i}", keys, feats)
        input_dirs.append(tmp_path / f"stats.{i}")
    out = tmp_path / "out"
    main([f"--input_dir={d}" for d in input_dirs] + [f"--output_dir={out}"])

    for mode in ["train", "valid"]:
        assert (out / mode / "speech_shape").read_text() == "a 5\nb 7\nc 7\nd 2\n"
        scp = (out / mode / "collect_feats" / "feats.scp").read_text().splitlines()
        assert [line.split()[0] for line in scp] == ["a", "b", "c", "d"]
        stats = np.load(out / mode / "feats_stats.npz")
        x = np.concatenate([feats[k] for k in "abcd"])
        assert stats["count"] == len(x)
        np.testing.assert_allclose(stats["sum"], x.sum(0))
        np.testing.assert_allclose(stats["sum_square"], (x**2).sum(0))
