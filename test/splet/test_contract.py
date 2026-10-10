"""Tests for the metric contract: requirements, identity, reduction, provenance.

These are the rules a follow-up metric has to satisfy, so each is pinned by
a test rather than by a sentence in a docstring.
"""

import json

import pytest

from splet.metadata import metadata
from splet.metric_registry import (
    METRIC_CHOICES,
    MetricSpec,
    load_metrics,
    measure_utterances,
    validate_requirements,
)
from splet.summary import error_rate_from_counts, summarize
from splet.utils_shared import text_loader_setup
from splet.utterance_metrics import error_rate_metric, error_rate_setup

# --- zero denominator: one policy, utterance and corpus --------------------


def test_empty_reference_with_insertions_is_one_at_both_levels():
    """ref '' / hyp 'hallucinated words': two insertions, rate 1.0 everywhere."""
    modules = load_metrics([{"name": "wer"}])
    (result,) = measure_utterances({"u": "hallucinated words"}, modules, {"u": ""})
    assert (result["wer"], result["wer_errors"], result["wer_ins"]) == (1.0, 2, 2)
    summary = summarize([result], modules)
    assert summary["wer"] == 1.0
    assert (summary["wer_errors"], summary["wer_ref_len"]) == (2, 0)


def test_empty_reference_and_empty_hypothesis_is_zero_at_both_levels():
    modules = load_metrics([{"name": "wer"}])
    (result,) = measure_utterances({"u": ""}, modules, {"u": ""})
    assert result["wer"] == 0.0
    assert summarize([result], modules)["wer"] == 0.0


def test_error_rate_from_counts_is_the_shared_policy():
    assert error_rate_from_counts(2, 0) == 1.0
    assert error_rate_from_counts(0, 0) == 0.0
    assert error_rate_from_counts(1, 4) == 0.25


def test_insertions_against_an_empty_reference_count_in_the_corpus():
    """Mixed with a normal utterance, the two insertions reach the numerator."""
    modules = load_metrics([{"name": "wer"}])
    results = measure_utterances(
        {"u1": "hallucinated words", "u2": "the cat"},
        modules,
        {"u1": "", "u2": "the cat sat"},
    )
    summary = summarize(results, modules)
    assert (summary["wer_errors"], summary["wer_ref_len"]) == (3, 3)


# --- duplicate ids are errors, not overwrites ------------------------------


def test_kaldi_reader_rejects_a_duplicate_utterance_id(tmp_path):
    path = tmp_path / "hyp"
    path.write_text("u wrong answer\nu correct\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate utterance id 'u'"):
        text_loader_setup(str(path), "kaldi")


def test_jsonl_reader_rejects_a_duplicate_key(tmp_path):
    path = tmp_path / "hyp.jsonl"
    path.write_text(
        '{"key": "u", "text": "wrong answer"}\n{"key": "u", "text": "correct"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key 'u'"):
        text_loader_setup(str(path), "jsonl")


def test_dir_reader_rejects_colliding_stems(tmp_path):
    directory = tmp_path / "hyp"
    directory.mkdir()
    (directory / "u.txt").write_text("wrong answer", encoding="utf-8")
    (directory / "u.json").write_text("correct", encoding="utf-8")
    with pytest.raises(ValueError, match="both name utterance 'u'"):
        text_loader_setup(str(directory), "dir")


# --- instance identity ------------------------------------------------------


def test_one_implementation_runs_twice_under_two_ids():
    """Raw and normalized WER coexist, each under its own keys."""
    modules = load_metrics(
        [
            {"name": "wer", "keep_alignment": True},
            {
                "name": "wer",
                "id": "wer_norm",
                "normalize": [{"name": "lowercase"}],
                "keep_alignment": True,
            },
        ]
    )
    assert list(modules) == ["wer", "wer_norm"]
    (result,) = measure_utterances({"u": "The cat"}, modules, {"u": "the cat"})
    assert result["wer"] == 0.5
    assert result["wer_norm"] == 0.0
    assert "wer_alignment" in result and "wer_norm_alignment" in result
    summary = summarize([result], modules)
    assert (summary["wer"], summary["wer_norm"]) == (0.5, 0.0)


def test_a_repeated_id_is_rejected():
    with pytest.raises(ValueError, match="configured twice"):
        load_metrics([{"name": "wer"}, {"name": "wer"}])


def test_alignment_key_is_namespaced_by_id():
    state = error_rate_setup(metric_id="cer", tokenizer="char", keep_alignment=True)
    assert "cer_alignment" in error_rate_metric(state, "ab", "ab")


# --- declared requirements --------------------------------------------------


def test_a_metric_that_requires_a_reference_fails_up_front_without_one():
    modules = load_metrics([{"name": "wer"}])
    with pytest.raises(ValueError, match="metric 'wer' requires a reference"):
        measure_utterances({"u": "a"}, modules, None)


def test_validate_requirements_passes_when_the_reference_is_there():
    modules = load_metrics([{"name": "wer"}])
    validate_requirements(modules, {"u": "a"})


def test_spec_rejects_unknown_tier_and_requirement():
    with pytest.raises(ValueError, match="unknown tier"):
        MetricSpec(tier="paragraph", setup=None, metric=None, outputs={})
    with pytest.raises(ValueError, match="unknown requirements"):
        MetricSpec(
            tier="utterance", setup=None, metric=None, outputs={}, requires=("mood",)
        )


def test_every_registered_metric_declares_its_outputs_and_a_version():
    for name, spec in METRIC_CHOICES.items():
        assert "" in spec.outputs, f"{name} does not declare its headline key"
        assert spec.version


# --- declared reduction -----------------------------------------------------


def test_an_unknown_summary_rule_is_rejected():
    from splet.summary import declared_rules

    spec = MetricSpec(tier="utterance", setup=None, metric=None, outputs={"": "avg"})
    with pytest.raises(ValueError, match="unknown summary rule 'avg'"):
        declared_rules({"m": {"spec": spec}})


def test_summary_rejects_a_result_key_no_metric_declared():
    modules = load_metrics([{"name": "wer"}])
    (result,) = measure_utterances({"u": "a"}, modules, {"u": "a"})
    result["surprise"] = 3.0
    with pytest.raises(ValueError, match="not declared"):
        summarize([result], modules)


def test_text_outputs_are_carried_but_not_summarized():
    modules = load_metrics([{"name": "wer", "keep_alignment": True}])
    results = measure_utterances({"u": "a b"}, modules, {"u": "a c"})
    assert "wer_alignment" in results[0]
    assert "wer_alignment" not in summarize(results, modules)


# --- provenance -------------------------------------------------------------


def test_metadata_records_what_produced_the_result():
    modules = load_metrics(
        [{"name": "wer", "id": "wer_norm", "backend": "python"}],
        normalize=[{"name": "lowercase"}, {"name": "remove_punctuation", "keep": "'"}],
    )
    block = metadata(modules)
    assert block["format"] == 1
    assert block["splet"]
    entry = block["metrics"]["wer_norm"]
    assert entry["name"] == "wer"
    assert entry["version"] == METRIC_CHOICES["wer"].version
    assert entry["requires"] == ["reference"]
    assert entry["backend"] == "python"
    assert entry["config"]["tokenizer"] == "word"
    assert entry["config"]["normalize"] == [
        {"name": "lowercase"},
        {"name": "remove_punctuation", "keep": "'"},
    ]
    # It is written out, so it must survive a round trip through JSON.
    assert json.loads(json.dumps(block)) == block


# --- batches: the validation-loop path ---------------------------------------


def test_accumulator_matches_summarize_batch_by_batch():
    """Adding a batch at a time gives the corpus figure, not a mean of batches."""
    from splet.metric_registry import measure_batch
    from splet.summary import Accumulator

    modules = load_metrics([{"name": "wer"}, {"name": "cer"}])
    refs = ["a b c d e f g h", "x", "the cat sat", ""]
    hyps = ["a b c d e f g h", "y", "the cat", "boo"]
    acc = Accumulator(modules)
    for start in range(0, len(refs), 2):  # two batches of two
        acc.add_all(
            measure_batch(hyps[start : start + 2], refs[start : start + 2], modules)
        )
    corpus = acc.result()
    full = summarize(
        measure_utterances(
            {str(i): h for i, h in enumerate(hyps)},
            modules,
            {str(i): r for i, r in enumerate(refs)},
        ),
        modules,
    )
    assert corpus == full
    assert corpus["wer"] == pytest.approx(3 / 12)  # 1 sub + 1 del + 1 ins over 12 words
    # The mean of the two batch rates would be a different, wrong, number.
    assert corpus["wer"] != pytest.approx((1 / 9 + 2 / 3) / 2)


def test_accumulator_state_sums_across_workers():
    """Two workers' counts, summed, give the same figure as one worker."""
    from splet.metric_registry import measure_batch
    from splet.summary import Accumulator

    modules = load_metrics([{"name": "wer"}])
    worker1 = Accumulator(modules).add_all(measure_batch(["a b"], ["a c"], modules))
    worker2 = Accumulator(modules).add_all(measure_batch(["x y z"], ["x y"], modules))
    merged = {"num_utterances": 0, "sums": {}}
    for state in (worker1.state(), worker2.state()):
        merged["num_utterances"] += state["num_utterances"]
        for key, value in state["sums"].items():
            merged["sums"][key] = merged["sums"].get(key, 0) + value
    together = Accumulator(modules).load_state(merged).result()
    assert together["wer_errors"] == 2 and together["wer_ref_len"] == 4
    assert together["wer"] == 0.5


def test_measure_batch_rejects_mismatched_lengths_and_missing_reference():
    from splet.metric_registry import measure_batch

    modules = load_metrics([{"name": "wer"}])
    with pytest.raises(ValueError, match="2 hypotheses but 1 references"):
        measure_batch(["a", "b"], ["a"], modules)
    with pytest.raises(ValueError, match="requires a reference"):
        measure_batch(["a"], None, modules)


def test_accumulator_reset_forgets():
    from splet.metric_registry import measure_batch
    from splet.summary import Accumulator

    modules = load_metrics([{"name": "wer"}])
    acc = Accumulator(modules).add_all(measure_batch(["a"], ["b"], modules))
    assert acc.result()["wer"] == 1.0
    acc.reset()
    assert acc.result() == {"num_utterances": 0}
