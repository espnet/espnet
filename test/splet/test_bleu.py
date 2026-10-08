"""Tests for the sacrebleu-backed corpus metrics."""

import json
import subprocess
import sys

import pytest

from splet.bin.measure import main
from splet.corpus_metrics import (
    bleu_setup,
    chrf_setup,
    sacrebleu_metric,
    translation_edit_rate_setup,
)
from splet.metric_registry import load_corpus_metrics, measure_corpus

pytest.importorskip("sacrebleu")

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


def test_corpus_bleu_is_not_the_mean_of_sentence_bleus():
    state = bleu_setup()
    corpus = sacrebleu_metric(state, HYPS, REFS)["bleu"]
    sentences = [sacrebleu_metric(state, [h], [r])["bleu"] for h, r in zip(HYPS, REFS)]
    assert corpus != pytest.approx(sum(sentences) / len(sentences), abs=0.1)


def test_bleu_reports_brevity_penalty_and_lengths():
    result = sacrebleu_metric(bleu_setup(), ["a b c"], ["a b c d e f"])
    assert result["bleu_hyp_len"] == 3
    assert result["bleu_ref_len"] == 6
    assert result["bleu_ratio"] == pytest.approx(0.5)
    assert result["bleu_bp"] < 1.0


def test_normalization_is_applied_to_both_sides():
    normalize = [{"name": "lowercase"}, {"name": "remove_punctuation", "keep": "'"}]
    hyps, refs = ["hello WORLD it's fine"], ["Hello, world! It's fine."]
    assert sacrebleu_metric(bleu_setup(), hyps, refs)["bleu"] < 100.0
    for setup, name, perfect in (
        (bleu_setup, "bleu", 100.0),
        (chrf_setup, "chrf", 100.0),
        (translation_edit_rate_setup, "translation_edit_rate", 0.0),
    ):
        result = sacrebleu_metric(setup(normalize=normalize), hyps, refs)
        assert result[name] == pytest.approx(perfect)


def test_utterances_are_matched_by_key_not_by_order():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    pred = {f"utt{i}": text for i, text in reversed(list(enumerate(HYPS)))}
    in_order = sacrebleu_metric(bleu_setup(), HYPS, REFS)["bleu"]
    assert measure_corpus(pred, metrics, gt)["bleu"] == pytest.approx(in_order)


def test_hypothesis_without_reference_is_an_error():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    with pytest.raises(KeyError, match="utt9"):
        measure_corpus({"utt0": "a", "utt9": "b"}, metrics, {"utt0": "a"})


def test_reference_without_hypothesis_is_an_error():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    with pytest.raises(KeyError, match="utt1' and 2 more"):
        measure_corpus({"utt0": HYPS[0]}, metrics, gt)


def test_empty_hypothesis_is_measured_not_dropped():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    pred = {f"utt{i}": text for i, text in enumerate(HYPS)}
    full = measure_corpus(pred, metrics, gt)
    pred["utt3"] = ""
    empty = measure_corpus(pred, metrics, gt)
    assert empty["bleu_ref_len"] == full["bleu_ref_len"]
    assert empty["bleu"] < full["bleu"]


def test_bleu_needs_references():
    with pytest.raises(ValueError, match="references"):
        sacrebleu_metric(bleu_setup(), ["a"], None)


def test_corpus_tier_checks_for_a_reference_before_measuring():
    metrics = load_corpus_metrics([{"name": "chrf"}])
    with pytest.raises(ValueError, match="metric 'chrf' requires a reference"):
        measure_corpus({"utt0": "a"}, metrics, None)


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
