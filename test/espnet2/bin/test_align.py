"""The forced aligner, on emissions a test can write out by hand.

No model is loaded here: alignment needs the CTC posteriors and the token
ids, and a stand-in can provide both. What is checked is the arithmetic
between them - which frames a token is given, and what comes back as a
second - because that is what a caller reads.
"""

import types

import numpy as np
import pytest
import torch

from espnet2.bin.align import ForcedAligner, Segment, Token

TOKENS = ["<blank>", "a", "b", "c"]


def _emissions(path):
    """Log probabilities that make `path` the obvious alignment.

    `path` is one token id a frame; each frame is near-certain about its own.
    """
    probs = np.full((len(path), len(TOKENS)), 1e-6, dtype=np.float32)
    for frame, token in enumerate(path):
        probs[frame, token] = 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


class _Stub:
    """Enough of a Speech2Text for the aligner: a tokenizer and a CTC head."""

    def __init__(self, emissions, sample_rate=16000):
        self.emissions = emissions
        self.sample_rate = sample_rate
        self.dtype = "float32"
        self.device = "cpu"
        self.asr_model = types.SimpleNamespace(
            ctc=object(), blank_id=0, token_list=TOKENS
        )
        self.tokenizer = types.SimpleNamespace(text2tokens=list)
        self.converter = types.SimpleNamespace(
            tokens2ids=lambda tokens: [TOKENS.index(t) for t in tokens]
        )

    def read_audio(self, speech):
        return np.asarray(speech, dtype=np.float32)


def _aligner(path):
    aligner = ForcedAligner(_Stub(_emissions(path)))
    aligner.log_probs = lambda speech: aligner.model.emissions
    return aligner


def test_each_utterance_gets_the_frames_it_was_said_in():
    # "a" in the first second, silence, "bc" in the third
    frames = [1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 3, 3]
    aligner = _aligner(frames)
    # twelve frames over a second of audio: twelve frames a second
    audio = np.zeros(16000, dtype=np.float32)

    segments = aligner(audio, ["a", "bc"])

    assert [s.text for s in segments] == ["a", "bc"]
    assert segments[0].start == pytest.approx(0.0, abs=0.09)
    assert segments[0].end == pytest.approx(2 / 12, abs=0.09)
    assert segments[1].start == pytest.approx(8 / 12, abs=0.09)
    assert segments[1].end == pytest.approx(1.0, abs=0.09)
    assert segments[0].score > 0.9 and segments[1].score > 0.9


def test_a_segment_carries_the_tokens_it_was_made_of():
    aligner = _aligner([1, 0, 2, 3])

    tokens = aligner(np.zeros(16000, dtype=np.float32), ["abc"])[0].tokens

    assert [t.text for t in tokens] == ["a", "b", "c"]
    assert tokens[0].start < tokens[1].start < tokens[2].start
    assert all(isinstance(t, Token) for t in tokens)


def test_text_that_cannot_fit_is_refused_by_name():
    aligner = _aligner([1, 2])

    with pytest.raises(ValueError, match="cannot fit"):
        aligner(np.zeros(16000, dtype=np.float32), ["abcabcabc"])


def test_nothing_to_align_is_refused():
    aligner = _aligner([1, 2, 3])
    audio = np.zeros(16000, dtype=np.float32)

    with pytest.raises(ValueError, match="give the utterances"):
        aligner(audio, [])
    with pytest.raises(ValueError, match="nothing to align"):
        aligner(audio, [""])


def test_a_model_without_a_ctc_head_says_so():
    stub = _Stub(_emissions([1]))
    stub.asr_model.ctc = None

    with pytest.raises(ValueError, match="no CTC head"):
        ForcedAligner(stub)


def test_the_segment_type_is_what_it_says():
    segment = Segment(text="a", start=0.0, end=1.0, score=0.5, tokens=[])
    assert (segment.text, segment.start, segment.end, segment.score) == (
        "a",
        0.0,
        1.0,
        0.5,
    )
    assert torch  # the module under test needs it, and so does this file
