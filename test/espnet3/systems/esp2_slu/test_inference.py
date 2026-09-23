"""Tests for the esp2_slu two-pass inference wrapper."""

from pathlib import Path

import pytest
import torch

import espnet3.systems.esp2_slu.inference as slu_inference_module
from espnet3.systems.esp2_slu.inference import SLUInference

_TOKENS = ["<blank>", "<unk>", "news", "read", "the", "<sos/eos>"]


@pytest.fixture()
def token_list_path(tmp_path: Path) -> Path:
    path = tmp_path / "transcript_tokens.txt"
    path.write_text("\n".join(_TOKENS) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def recorded_calls(monkeypatch):
    """Replace `Speech2Understand` with a stub that records what it is given."""
    calls = {}

    class DummySpeech2Understand:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def __call__(self, speech, transcript):
            calls["speech"] = speech
            calls["transcript"] = transcript
            return [("news_query read the news", [], [], None)]

    monkeypatch.setattr(
        slu_inference_module, "Speech2Understand", DummySpeech2Understand
    )
    return calls


def test_transcript_is_tokenized_against_the_token_list(
    token_list_path, recorded_calls
):
    """The string field must reach the model as ids, not as text.

    `SLUPreprocessor` does this during training; the infer stage has no
    preprocessor, so without it every run fails inside `Speech2Understand`.
    """
    model = SLUInference(transcript_token_list=token_list_path)

    model(speech=torch.zeros(16000), transcript="read the news")

    transcript = recorded_calls["transcript"]
    assert isinstance(transcript, torch.Tensor)
    assert transcript.dtype == torch.long
    assert transcript.tolist() == [3, 4, 2]


def test_unknown_words_become_unk(token_list_path, recorded_calls):
    """A word the first pass invented must not break the lookup."""
    model = SLUInference(transcript_token_list=token_list_path)

    assert model.transcript_to_ids("read gibberish") == [3, 1]


def test_empty_transcript_yields_one_token(token_list_path, recorded_calls):
    """A blank first-pass hypothesis must not produce an empty sequence."""
    model = SLUInference(transcript_token_list=token_list_path)

    assert model.transcript_to_ids("   ") == [1]


def test_remaining_arguments_reach_speech2understand(token_list_path, recorded_calls):
    """Decoding parameters and the device are passed straight through."""
    SLUInference(
        transcript_token_list=token_list_path,
        beam_size=20,
        ctc_weight=0.5,
        device="cpu",
    )

    assert recorded_calls["init_kwargs"] == {
        "beam_size": 20,
        "ctc_weight": 0.5,
        "device": "cpu",
    }
