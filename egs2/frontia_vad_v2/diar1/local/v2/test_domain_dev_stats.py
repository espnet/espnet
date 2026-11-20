"""Tests of domain_dev_stats.py: full-precision ratios from the reporter state."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

torch = pytest.importorskip("torch")

from domain_dev_stats import domain_table, epoch_stats  # noqa: E402


def test_small_domain_keeps_its_precision(tmp_path):
    # a domain with 0.6237 % of the windows, loss 0.31 and accuracy 0.9
    frac = 0.006237
    valid = {"loss": 0.2}
    for s in ("0", "124"):
        valid[f"valid_frac_animal_la{s}"] = frac
        valid[f"valid_loss_num_animal_la{s}"] = frac * 0.31
        valid[f"valid_acc_num_animal_la{s}"] = frac * 0.9
    states = {
        "reporter": {"stats": {1: {"train": {"loss": 0.5}, "valid": valid}}, "epoch": 1}
    }
    torch.save(states, tmp_path / "checkpoint.pth")
    suffixes, rows = domain_table(epoch_stats(tmp_path)[1])
    assert suffixes == ["0", "124"]
    ((c, f, per),) = rows
    assert c == "animal" and f == pytest.approx(frac)
    for s in suffixes:
        assert per[s][0] == pytest.approx(0.31)
        assert per[s][1] == pytest.approx(0.9)
    # the three decimals of train.log give 0.002 / 0.006 = 1/3, not 0.31
    assert round(frac * 0.31, 3) / round(frac, 3) == pytest.approx(1 / 3)


def test_absent_domain_is_nan(tmp_path):
    valid = {
        "valid_frac_music_la0": 0.0,
        "valid_loss_num_music_la0": 0.0,
        "valid_acc_num_music_la0": 0.0,
    }
    torch.save(
        {"reporter": {"stats": {1: {"valid": valid}}, "epoch": 1}},
        tmp_path / "checkpoint.pth",
    )
    _, rows = domain_table(epoch_stats(tmp_path)[1])
    assert rows[0][2]["0"][0] != rows[0][2]["0"][0]  # nan
