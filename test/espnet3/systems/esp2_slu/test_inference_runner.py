"""Tests for the esp2_slu inference runners."""

import numpy as np
import pytest

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.esp2_slu.src.inference_runner import (
    FirstPassTranscriptRunner,
    SLUInferenceRunner,
)
from espnet3.systems.esp2_slu.src.slu_output import split_intent


def _result(text):
    """One n-best entry shaped as ESPnet2 returns it."""
    return [(text, ["t"], [1], None)]


def test_split_intent_handles_partial_text():
    assert split_intent("news_query read the news") == ("news_query", "read the news")
    assert split_intent("news_query") == ("news_query", "")
    assert split_intent("   ") == ("", "")


def test_slu_runner_splits_the_hypothesis():
    """The intent and the words are scored separately, so they are split here."""
    out = SLUInferenceRunner.format_output(
        _result("news_query read the news"), batched=False
    )

    assert out == {
        "hyp": "news_query read the news",
        "hyp_intent": "news_query",
        "hyp_transcript": "read the news",
    }


def test_slu_runner_survives_an_intent_only_hypothesis():
    out = SLUInferenceRunner.format_output(_result("news_query"), batched=False)

    assert out["hyp_intent"] == "news_query"
    assert out["hyp_transcript"] == ""


def test_slu_runner_survives_an_empty_hypothesis():
    out = SLUInferenceRunner.format_output([(None, [], [], None)], batched=False)

    assert out == {"hyp": "", "hyp_intent": "", "hyp_transcript": ""}


def test_slu_runner_refuses_a_batch():
    """The backend decodes one utterance at a time; a batch would misalign."""
    with pytest.raises(ValueError, match="one utterance at a time"):
        SLUInferenceRunner.format_output([_result("a")], batched=True)


def test_first_pass_runner_keeps_the_hypothesis_whole():
    """Splitting an ASR hypothesis would discard its first word."""
    out = FirstPassTranscriptRunner.format_output(
        _result("read the news"), batched=False
    )

    assert out["hyp_transcript"] == "read the news"
    assert out["hyp_intent"] == ""


def test_first_pass_runner_handles_a_batch():
    out = FirstPassTranscriptRunner.format_output(
        [_result("one"), _result("two")], batched=True
    )

    assert [item["hyp_transcript"] for item in out] == ["one", "two"]


def test_runners_hand_the_backend_a_plain_array():
    audio = Audio(np.zeros(8000, dtype=np.float32), 16000)

    for runner in (SLUInferenceRunner, FirstPassTranscriptRunner):
        prepared = runner.prepare_model_inputs({"speech": audio, "transcript": "hi"})
        assert isinstance(prepared["speech"], np.ndarray)
        assert prepared["speech"].shape == (8000,)
        assert prepared["transcript"] == "hi"
