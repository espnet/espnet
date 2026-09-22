"""The forced aligner, on emissions a test can write out by hand.

No model is loaded here: alignment needs the CTC posteriors and the token
ids, and a stand-in can provide both. What is checked is the arithmetic
between them - which frames a token is given, and what comes back as a
second - because that is what a caller reads.
"""

import sys
import types
import warnings

import numpy as np
import pytest
import torch

from espnet2.bin.align import ForcedAligner, Segment, Token
from espnet2.utils.pretrained import ModelTagError

TOKENS = ["<blank>", "a", "b", "c"]


def _emissions(path, tokens=None):
    """Log probabilities that make `path` the obvious alignment.

    `path` is one token id a frame; each frame is near-certain about its own.
    `tokens` is the vocabulary the frames are over, when it is not the small
    one this file mostly uses.
    """
    probs = np.full((len(path), len(tokens or TOKENS)), 1e-6, dtype=np.float32)
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


def test_a_repeated_token_needs_the_blank_between_it():
    """Three frames hold three tokens, unless two of them are the same.

    CTC reads "aa" as one "a" without a blank in between, so the text has to
    fit the blanks too. Counting only the tokens let a text through to
    torchaudio, which refused it in its own words.
    """
    aligner = _aligner([1, 1, 2])

    with pytest.raises(ValueError, match="blank"):
        aligner(np.zeros(16000, dtype=np.float32), ["aab"])

    # the same three frames take three different tokens
    segments = aligner(np.zeros(16000, dtype=np.float32), ["ab"])
    assert [t.text for t in segments[0].tokens] == ["a", "b"]


BPE = ["<blank>", "a", "b", "c", "A", "B", "C", "abc"]


class _BPEStub(_Stub):
    """A vocabulary with one word in it, and the letters of that word.

    Like a real BPE model: the word it was trained on is one token, and the
    same word in another case is spelled out.
    """

    def __init__(self, emissions):
        super().__init__(emissions)
        self.asr_model.token_list = BPE
        self.tokenizer = types.SimpleNamespace(
            text2tokens=lambda text: [text] if text in BPE else list(text)
        )
        self.converter = types.SimpleNamespace(
            tokens2ids=lambda tokens: [BPE.index(t) for t in tokens]
        )


def test_text_spelled_a_way_the_vocabulary_lacks_is_pointed_out():
    """The times survive the wrong case; the score does not.

    Measured on test_utils/ctc_align_test.wav with espnet/owsm_ctc_v4_1B:
    "THE SALE OF THE HOTELS" is 18 tokens and scores 0.0000 where "The sale
    of the hotels" is 6 and scores 0.99. Nothing in the result says why, so
    the aligner says it.
    """
    aligner = ForcedAligner(_BPEStub(_emissions([7, 0, 4, 5, 6], BPE)))
    aligner.log_probs = lambda speech: aligner.model.emissions
    audio = np.zeros(16000, dtype=np.float32)

    with pytest.warns(UserWarning, match="vocabulary does not have"):
        aligner(audio, ["ABC"])

    # the spelling the vocabulary has is not remarked on
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aligner(audio, ["abc"])


def test_the_sample_rate_is_the_models_own():
    """A caller that reads its own audio has to resample to the same rate.

    The Space draws a waveform and so loads the audio itself; hard-coding
    16000 there makes a checkpoint trained at another rate quietly wrong.
    """
    aligner = _aligner([1, 2, 3])
    assert aligner.sample_rate == 16000

    aligner.model.sample_rate = 8000
    assert aligner.sample_rate == 8000


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


def test_from_pretrained_loads_the_class_the_model_was_published_for(monkeypatch):
    """The tag says which: an S2T config or an ASR one, and nothing else."""
    import espnet2.bin.align as align

    built = {}

    class _S2T:
        def __init__(self, **kwargs):
            built.update(kwargs, which="s2t")
            self.s2t_model = types.SimpleNamespace(ctc=object(), blank_id=0)

    class _ASR:
        def __init__(self, **kwargs):
            built.update(kwargs, which="asr")
            self.asr_model = types.SimpleNamespace(ctc=object(), blank_id=0)

    modules = {
        "espnet2.bin.s2t_inference": types.SimpleNamespace(Speech2Text=_S2T),
        "espnet2.bin.asr_inference": types.SimpleNamespace(Speech2Text=_ASR),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        align, "download_pretrained", lambda tag: {"s2t_train_config": "c"}
    )
    align.ForcedAligner.from_pretrained("espnet/a-model")
    assert built["which"] == "s2t" and built["s2t_train_config"] == "c"

    monkeypatch.setattr(
        align, "download_pretrained", lambda tag: {"asr_train_config": "c"}
    )
    align.ForcedAligner.from_pretrained("espnet/another", device="mps")
    assert built["which"] == "asr" and built["device"] == "mps"


def test_from_pretrained_names_a_tag_it_cannot_read(monkeypatch):
    import espnet2.bin.align as align

    monkeypatch.setattr(
        align, "download_pretrained", lambda tag: {"tts_train_config": "c"}
    )

    with pytest.raises(ModelTagError, match="neither an S2T nor an ASR"):
        align.ForcedAligner.from_pretrained("espnet/a-tts-model")


def test_an_s2t_model_is_read_through_its_own_buffering():
    """Speech2Text.ctc_log_probs knows the window; the aligner does not."""
    asked = []

    class _S2T:
        s2t_model = types.SimpleNamespace(ctc=object(), blank_id=0, token_list=TOKENS)
        sample_rate = 16000

        def ctc_log_probs(self, speech):
            asked.append(len(speech))
            return _emissions([1, 2, 3])

        def read_audio(self, speech):
            return np.asarray(speech, dtype=np.float32)

    aligner = ForcedAligner(_S2T())
    assert aligner.s2t

    probs = aligner.log_probs(np.zeros(1600, dtype=np.float32))

    assert probs.shape == (3, len(TOKENS)) and asked == [1600]


def test_the_segment_type_is_what_it_says():
    segment = Segment(text="a", start=0.0, end=1.0, score=0.5, tokens=[])
    assert (segment.text, segment.start, segment.end, segment.score) == (
        "a",
        0.0,
        1.0,
        0.5,
    )
    assert torch  # the module under test needs it, and so does this file
