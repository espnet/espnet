"""Tests for the VERSA metric of the ESPnet3 SVS system."""

import json
from pathlib import Path

import pytest

import espnet3.systems.esp2_svs.metrics.versa as versa_module
from espnet3.systems.esp2_svs.metrics.versa import VersaMetric

SCORE_CONFIG = [{"name": "mcd_f0", "dtw": True}]


class FakeScorer:
    """Stand-in for a ``versa.bin.scorer`` process.

    Writes one JSON line per utterance of its ``--pred`` shard, with ``mcd``
    set to the utterance number, and records every command.
    """

    commands = []
    returncode = 0

    def __init__(self, cmd, cwd, stdout, stderr, env):
        FakeScorer.commands.append(cmd)
        pred = Path(cmd[cmd.index("--pred") + 1])
        with open(Path(cwd) / cmd[cmd.index("--output_file") + 1], "w") as f:
            for line in pred.read_text().splitlines():
                key, _ = line.split(maxsplit=1)
                f.write(json.dumps({"key": key, "mcd": float(key[3:])}) + "\n")

    def wait(self):
        return FakeScorer.returncode


@pytest.fixture
def scorer(monkeypatch):
    FakeScorer.commands = []
    FakeScorer.returncode = 0
    monkeypatch.setattr(versa_module, "versa", object())
    monkeypatch.setattr(versa_module.subprocess, "Popen", FakeScorer)
    return FakeScorer


@pytest.fixture
def data(tmp_path):
    for name in ("wav", "ref"):
        lines = [f"utt{i} {name}/utt{i}.wav\n" for i in range(1, 5)]
        (tmp_path / f"{name}.scp").write_text("".join(lines))
    (tmp_path / "text.scp").write_text("utt1 la\n")
    return {k: tmp_path / f"{k}.scp" for k in ("wav", "ref", "text")}


def test_scores_are_averaged_over_shards(tmp_path, data, scorer):
    result = VersaMetric(SCORE_CONFIG, nj=2)(data, "test", tmp_path / "inference")

    assert result == {"mcd": pytest.approx(2.5)}
    # One warm-up utterance, then one process per shard.
    assert len(scorer.commands) == 3
    shards = [cmd[cmd.index("--pred") + 1] for cmd in scorer.commands[1:]]
    assert [len(Path(p).read_text().splitlines()) for p in shards] == [2, 2]
    # VERSA runs in the result directory, so wav paths are made absolute.
    gt = Path(scorer.commands[1][scorer.commands[1].index("--gt") + 1])
    assert all(Path(line.split()[1]).is_absolute() for line in gt.open())


def test_text_and_gpu_are_optional(tmp_path, data, scorer):
    VersaMetric(SCORE_CONFIG, ref_key=None)(data, "test", tmp_path)
    assert len(scorer.commands) == 1
    assert not {"--gt", "--text", "--use_gpu"} & set(scorer.commands[0])

    VersaMetric(SCORE_CONFIG, text_key="text", use_gpu=True)(data, "test", tmp_path)
    assert {"--gt", "--text", "--use_gpu"} <= set(scorer.commands[1])


def test_failed_scorer_raises(tmp_path, data, scorer):
    scorer.returncode = 1
    with pytest.raises(RuntimeError, match="VERSA failed"):
        VersaMetric(SCORE_CONFIG)(data, "test", tmp_path)


def test_missing_versa_raises(tmp_path, data, monkeypatch):
    monkeypatch.setattr(versa_module, "versa", None)
    with pytest.raises(RuntimeError, match="make versa.done"):
        VersaMetric(SCORE_CONFIG)(data, "test", tmp_path)
