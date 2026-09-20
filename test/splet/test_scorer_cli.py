"""Tests for the splet-score command line interface."""

import json

import pytest

from splet.bin.scorer import main

CONFIG = """
normalize:
  - name: lowercase
  - name: remove_punctuation
metrics:
  - name: wer
  - name: cer
"""


@pytest.fixture()
def corpus(tmp_path):
    """Write a three-utterance reference and hypothesis pair."""
    reference = tmp_path / "ref.txt"
    hypothesis = tmp_path / "hyp.txt"
    reference.write_text(
        "utt1 the quick brown fox\nutt2 hello world\nutt3 nothing at all\n",
        encoding="utf-8",
    )
    hypothesis.write_text(
        "utt1 The quick brown fox.\nutt2 hello word\nutt3\n", encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    return reference, hypothesis, config


def test_scores_a_corpus(tmp_path, corpus, capsys):
    reference, hypothesis, config = corpus
    output = tmp_path / "result.jsonl"

    assert (
        main(
            [
                "--hyp",
                str(hypothesis),
                "--ref",
                str(reference),
                "--score_config",
                str(config),
                "--output_file",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["num_utterances"] == 3
    # utt1 is normalized to a perfect match, utt2 has one substitution, and
    # utt3 is empty on the hypothesis side: three words deleted, not an
    # utterance to skip.
    assert summary["wer_ref_len"] == 9
    assert summary["wer_del"] == 3
    assert summary["wer_sub"] == 1
    assert summary["wer"] == pytest.approx(4 / 9)

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert [line["key"] for line in lines] == ["utt1", "utt2", "utt3"]
    assert lines[0]["wer"] == 0.0


def test_pred_and_gt_are_accepted_as_versa_names(tmp_path, corpus, capsys):
    """The VERSA spelling of the two inputs has to work, not only the alias."""
    reference, hypothesis, config = corpus
    assert (
        main(
            [
                "--pred",
                str(hypothesis),
                "--gt",
                str(reference),
                "--score_config",
                str(config),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["num_utterances"] == 3


def test_bare_list_config_is_accepted(tmp_path, corpus, capsys):
    """A config may be a plain list of metrics, exactly as in VERSA."""
    reference, hypothesis, _ = corpus
    config = tmp_path / "list.yaml"
    config.write_text("- name: wer\n", encoding="utf-8")
    assert (
        main(
            [
                "--pred",
                str(hypothesis),
                "--gt",
                str(reference),
                "--score_config",
                str(config),
            ]
        )
        == 0
    )
    assert "wer" in json.loads(capsys.readouterr().out)


def test_list_metrics(capsys):
    assert main(["--list_metrics"]) == 0
    printed = capsys.readouterr().out
    assert "wer\tutterance" in printed
    assert "cer\tutterance" in printed


def test_missing_arguments_exit_nonzero(tmp_path, corpus):
    reference, _, config = corpus
    assert main(["--ref", str(reference), "--score_config", str(config)]) == 2


def test_jsonl_io_reads_turns(tmp_path, capsys):
    """The structured format collapses to plain text for utterance metrics."""
    reference = tmp_path / "ref.jsonl"
    hypothesis = tmp_path / "hyp.jsonl"
    reference.write_text(
        json.dumps(
            {
                "key": "meeting1",
                "turns": [
                    {"speaker": "A", "text": "hello"},
                    {"speaker": "B", "text": "hi there"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hypothesis.write_text(
        json.dumps({"key": "meeting1", "text": "hello hi there"}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("- name: wer\n", encoding="utf-8")

    assert (
        main(
            [
                "--pred",
                str(hypothesis),
                "--gt",
                str(reference),
                "--score_config",
                str(config),
                "--io",
                "jsonl",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["wer"] == 0.0
