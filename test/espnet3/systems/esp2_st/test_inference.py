"""The ST system's Inference, with a stand-in for Speech2Text."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)
from espnet3.systems.esp2_st.inference import TOO_SHORT_TEXT, Inference


class FakeSpeech2Text:
    """Returns the n-best list st_inference.Speech2Text does.

    ``multidecoder`` returns its ``(translations, transcripts)`` pair
    instead, and ``too_short`` the error an unsubsamplable utterance
    raises.
    """

    def __init__(self, fs="16k", multidecoder=False, too_short=False, **kwargs):
        self.st_train_args = SimpleNamespace(frontend_conf={"fs": fs} if fs else {})
        self.multidecoder = multidecoder
        self.too_short = too_short
        self.kwargs = kwargs
        self.seen = None

    def __call__(self, speech):
        if self.too_short:
            raise TooShortUttError("has 1 frames, but needs more than 7", 1, 7)
        self.seen = speech
        nbest = [
            ("hallo welt", ["hallo", "welt"], [1, 2], None),
            ("hallo", ["hallo"], [1], None),
        ]
        if self.multidecoder:
            return (nbest, [("hello world", ["hello", "world"], [3, 4], None, None)])
        return nbest


def audio(seconds=1.0, rate=16000):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def test_translates_the_best_hypothesis():
    backend = FakeSpeech2Text()
    assert Inference(backend)(audio()) == {"text": "hallo welt"}
    assert isinstance(backend.seen, np.ndarray)


def test_a_multidecoder_model_gives_its_translation_not_its_transcript():
    """Speech2Text returns (translations, transcripts) there; only the first."""
    assert Inference(FakeSpeech2Text(multidecoder=True))(audio()) == {
        "text": "hallo welt"
    }


def test_an_utterance_too_short_to_subsample_is_scored_not_raised():
    """egs2 keeps the row so its reference still counts against BLEU."""
    assert Inference(FakeSpeech2Text(too_short=True))(audio(0.05)) == {
        "text": TOO_SHORT_TEXT
    }


def test_a_too_short_utterance_does_not_stop_the_rest_of_the_batch():
    class OnlyFirstIsShort(FakeSpeech2Text):
        def __call__(self, speech):
            if len(speech) < 1000:
                raise TooShortUttError("too short", 1, 7)
            return super().__call__(speech)

    out = Inference(OnlyFirstIsShort()).batch(
        [{"speech": audio(0.05)}, {"speech": audio(1.0)}]
    )
    assert out == [{"text": TOO_SHORT_TEXT}, {"text": "hallo welt"}]


def test_sample_rate_comes_from_the_frontend_config():
    assert Inference(FakeSpeech2Text(fs="16k")).sample_rate == 16000
    assert Inference(FakeSpeech2Text(fs=8000)).sample_rate == 8000


def test_an_empty_frontend_config_is_espnet2s_own_default():
    """mini_an4 leaves frontend_conf empty; DefaultFrontend then runs at 16 kHz."""
    assert Inference(FakeSpeech2Text(fs=None)).sample_rate == 16000


def test_the_declaration_is_what_the_infer_stage_reads():
    assert [f.name for f in Inference.inputs] == ["speech"]
    assert [f.name for f in Inference.outputs] == ["text"]


def test_a_hypothesis_with_no_text_is_an_empty_string():
    """Speech2Text leaves `text` None when it has no tokenizer."""

    class NoTokenizer(FakeSpeech2Text):
        def __call__(self, speech):
            return [(None, [], [], None)]

    assert Inference(NoTokenizer())(audio()) == {"text": ""}


@pytest.mark.parametrize("multidecoder", [False, True])
def test_the_backend_sees_mono_samples(multidecoder):
    backend = FakeSpeech2Text(multidecoder=multidecoder)
    Inference(backend)(np.zeros((16000, 2), dtype=np.float32))
    assert backend.seen.ndim == 1
