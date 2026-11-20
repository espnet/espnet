"""Tests for the splet-measure command line interface."""

import json
import subprocess
import sys

import pytest

from splet.bin.measure import main

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


def test_measures_a_corpus(tmp_path, corpus, capsys):
    reference, hypothesis, config = corpus
    output = tmp_path / "result.jsonl"

    assert (
        main(
            [
                "--hyp",
                str(hypothesis),
                "--ref",
                str(reference),
                "--metrics_config",
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
    # The summary says what produced it: the shared normalization pipeline
    # reached both metrics, and the block is versioned.
    assert summary["metadata"]["format"] == 1
    assert summary["metadata"]["metrics"]["wer"]["config"]["normalize"] == [
        {"name": "lowercase"},
        {"name": "remove_punctuation"},
    ]
    assert summary["metadata"]["metrics"]["cer"]["name"] == "cer"

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
                "--metrics_config",
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
                "--metrics_config",
                str(config),
            ]
        )
        == 0
    )
    assert "wer" in json.loads(capsys.readouterr().out)


def test_list_metrics(capsys):
    assert main(["--list_metrics"]) == 0
    printed = capsys.readouterr().out
    assert "wer\tutterance\treference" in printed
    assert "cer\tutterance\treference" in printed


def test_missing_arguments_exit_nonzero(tmp_path, corpus):
    reference, _, config = corpus
    assert main(["--ref", str(reference), "--metrics_config", str(config)]) == 2


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
                "--metrics_config",
                str(config),
                "--io",
                "jsonl",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["wer"] == 0.0


# Corpus tier (BLEU, chrF, TER) through the command, as st.sh runs it.

REFS = [
    "The cat sat on the mat.",
    "Hello, world!",
    "It is raining again today, isn't it?",
    "We'll meet at the station at 5 p.m.",
]
HYPS = [
    "the cat sat on a mat",
    "Hello world!",
    "It rains again today, isn't it?",
    "We will meet at the station at 5 pm.",
]


def _write(path, lines):
    path.write_text(
        "".join(f"utt{i} {line}\n" for i, line in enumerate(lines)), encoding="utf-8"
    )


def _sacrebleu_cli(tmp_path, *flags):
    """Run the sacrebleu command the way st.sh stage 13 does."""
    ref = tmp_path / "ref.trn.detok"
    hyp = tmp_path / "hyp.trn.detok"
    ref.write_text("\n".join(REFS) + "\n", encoding="utf-8")
    hyp.write_text("\n".join(HYPS) + "\n", encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "sacrebleu", *flags, str(ref), "-i", str(hyp)]
        + ["-m", "bleu", "chrf", "ter", "-w", "6"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {entry["name"]: entry for entry in json.loads(out)}


def _splet_measure(tmp_path, capsys, config):
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    _write(ref, REFS)
    _write(hyp, HYPS)
    conf = tmp_path / "metrics.yaml"
    conf.write_text(config, encoding="utf-8")
    argv = ["--pred", str(hyp), "--gt", str(ref), "--metrics_config", str(conf)]
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_matches_st_sh_case_sensitive_pass(tmp_path, capsys):
    """st.sh: sacrebleu ref -i hyp -m bleu chrf ter."""
    expected = _sacrebleu_cli(tmp_path)
    summary = _splet_measure(
        tmp_path, capsys, "- name: bleu\n- name: chrf\n- name: translation_edit_rate\n"
    )

    for key, name in (
        ("bleu", "BLEU"),
        ("chrf", "chrF2"),
        ("translation_edit_rate", "TER"),
    ):
        assert summary[key] == pytest.approx(expected[name]["score"], abs=1e-6)
        signature = summary["metadata"]["metrics"][key]["signature"]
        assert signature == expected[name]["signature"]
    assert summary["num_utterances"] == 4


def test_matches_st_sh_lowercased_sacrebleu_call(tmp_path, capsys):
    """st.sh: sacrebleu -lc ref -i hyp -m bleu chrf ter.

    -lc lowercases BLEU only, so only the bleu entry takes lowercase: true.
    The remove_punctuation.pl step st.sh runs before this call is not tested.
    """
    expected = _sacrebleu_cli(tmp_path, "-lc")
    summary = _splet_measure(
        tmp_path,
        capsys,
        "- name: bleu\n  lowercase: true\n"
        "- name: chrf\n- name: translation_edit_rate\n",
    )

    for key, name in (
        ("bleu", "BLEU"),
        ("chrf", "chrF2"),
        ("translation_edit_rate", "TER"),
    ):
        assert summary[key] == pytest.approx(expected[name]["score"], abs=1e-6)
        signature = summary["metadata"]["metrics"][key]["signature"]
        assert signature == expected[name]["signature"]
    assert "case:lc" in summary["metadata"]["metrics"]["bleu"]["signature"]


def test_bleu_runs_twice_under_two_ids(tmp_path, capsys):
    """Case-sensitive and lowercased BLEU in one config, as st.sh runs both."""
    summary = _splet_measure(
        tmp_path,
        capsys,
        "- name: bleu\n- name: bleu\n  id: bleu_lc\n  lowercase: true\n",
    )
    assert summary["bleu_lc"] > summary["bleu"]
    assert summary["bleu_lc_ref_len"] == summary["bleu_ref_len"]
    described = summary["metadata"]["metrics"]
    assert described["bleu_lc"]["name"] == "bleu"
    assert "case:lc" in described["bleu_lc"]["signature"]
    assert "case:mixed" in described["bleu"]["signature"]


def test_corpus_and_utterance_metrics_in_one_config(tmp_path, capsys):
    summary = _splet_measure(tmp_path, capsys, "- name: wer\n- name: bleu\n")
    assert "wer" in summary
    assert "bleu" in summary


def test_list_metrics_shows_the_corpus_tier(capsys):
    assert main(["--list_metrics"]) == 0
    printed = capsys.readouterr().out
    for name in ("bleu", "chrf", "translation_edit_rate"):
        assert f"{name}\tcorpus" in printed


def test_an_id_used_in_both_tiers_is_rejected(tmp_path, capsys):
    # BLEU under the id `wer` would replace the WER figure and its metadata.
    with pytest.raises(ValueError, match="'wer'"):
        _splet_measure(tmp_path, capsys, "- name: wer\n- name: bleu\n  id: wer\n")


def test_result_keys_colliding_across_tiers_are_rejected(tmp_path, capsys):
    # BLEU under the id `wer_errors` reports into WER's `wer_errors` count.
    with pytest.raises(ValueError, match="wer_errors"):
        _splet_measure(
            tmp_path, capsys, "- name: wer\n- name: bleu\n  id: wer_errors\n"
        )
