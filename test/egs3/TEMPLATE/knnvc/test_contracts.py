"""The recipe-local kinds of the kNN-VC TEMPLATE: reference_audio and features."""

import numpy as np
import pytest
import torch

import egs3.TEMPLATE.knnvc.src.contracts as contracts
from espnet3.contract import KINDS, Field
from espnet3.contract.kinds.audio import Audio

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_kinds_are_registered_once              | Both names resolve; a second |
# |                                             | import does not raise.       |
# | test_reference_audio_checks_each_utterance  | Every element is an Audio at |
# |                          | the field's rate and channels; empty/ scalar/   |
# |                          | stereo/ NaN elements are refused.               |
# | test_reference_audio_tells_batch_from_set   | A list of arrays is one set, |
# |                          | a list of lists a batch, a (rate, samples) pair  |
# |                          | one utterance.                                  |
# | test_reference_audio_accepts_without_model  | Dataset items are checked    |
# |                                             | with no model at hand.       |
# | test_reference_audio_normalizers_apply_per_utterance |                   |
# | test_features_is_a_finite_real_matrix       | 2-D float only; tensors are  |
# |                          | converted; NaN, ints and 1-D are refused.       |

SR = 16000
FIELD = Field("reference_speech", "reference_audio", channels=1)


def _wav(seconds, seed=0, rate=SR):
    return (np.random.RandomState(seed).randn(int(seconds * rate)) * 0.1).astype(
        np.float32
    )


def test_kinds_are_registered_once():
    assert isinstance(KINDS["reference_audio"], contracts.ReferenceAudioKind)
    assert isinstance(KINDS["features"], contracts.FeaturesKind)
    import importlib

    importlib.reload(contracts)  # a second registration is a no-op
    assert isinstance(KINDS["reference_audio"], contracts.ReferenceAudioKind)


def test_reference_audio_checks_each_utterance():
    kind = KINDS["reference_audio"]
    options = {"sample_rate": SR}
    refs = kind.apply(
        [_wav(0.3), (8000, _wav(0.4, rate=8000)), Audio(_wav(0.2), SR)],
        FIELD,
        None,
        options,
        output=False,
    )
    assert [type(r) for r in refs] == [Audio] * 3
    assert [r.rate for r in refs] == [SR] * 3
    assert [r.channels for r in refs] == [1] * 3
    assert refs[1].array.shape == (int(0.4 * SR),)

    with pytest.raises(TypeError, match="at least one"):
        kind.apply([], FIELD, None, options, output=False)
    with pytest.raises(TypeError, match="list of reference"):
        kind.apply(_wav(0.3), FIELD, None, options, output=False)
    with pytest.raises(TypeError, match="channel"):
        kind.apply([np.zeros((2, SR), np.float32)], FIELD, None, options, output=False)
    with pytest.raises(ValueError, match="finite"):
        kind.apply(
            [np.full(SR, np.nan, np.float32)], FIELD, None, options, output=False
        )
    with pytest.raises(ValueError, match="sample rate"):
        kind.apply(
            [(8000, _wav(0.4, rate=8000))],
            FIELD,
            None,
            {"sample_rate": SR, "rate_policy": "reject"},
            output=False,
        )
    with pytest.raises(ValueError, match="Unknown"):
        Field("reference_speech", "reference_audio", options={"bogus": 1})


def test_reference_audio_tells_batch_from_set():
    kind = KINDS["reference_audio"]
    one_set = [_wav(0.3), _wav(0.4)]
    assert not kind.is_batch(one_set, FIELD, None)
    assert kind.is_batch([one_set, one_set], FIELD, None)
    # A (rate, samples) pair is one utterance, so a list of them is one set.
    assert not kind.is_batch([(SR, _wav(0.3)), (SR, _wav(0.4))], FIELD, None)
    assert not kind.is_batch((SR, _wav(0.3)), FIELD, None)


def test_reference_audio_accepts_without_model():
    kind = KINDS["reference_audio"]
    assert kind.accepts([_wav(0.3), torch.zeros(SR)], FIELD)
    assert kind.accepts([Audio(_wav(0.3), SR), "ref.wav"], FIELD)
    assert not kind.accepts([], FIELD)
    assert not kind.accepts(_wav(0.3), FIELD)
    assert not kind.accepts((SR, _wav(0.3)), FIELD)
    assert not kind.accepts(["ref.wav", 3], FIELD)


def test_reference_audio_normalizers_apply_per_utterance():
    field = Field(
        "reference_speech",
        "reference_audio",
        channels=1,
        options={"sample_rate": SR, "normalize": ["first_channel"]},
    )
    refs = KINDS["reference_audio"].apply(
        [np.zeros((2, SR), np.float32), _wav(0.3)],
        field,
        None,
        field.options,
        output=False,
    )
    assert [r.channels for r in refs] == [1, 1]


def test_features_is_a_finite_real_matrix():
    kind = KINDS["features"]
    field = Field("feats", "features")
    matrix = np.zeros((5, 8), np.float32)
    assert kind.check(matrix, field, None, output=False) is matrix
    assert kind.check(torch.zeros(5, 8), field, None, output=False).shape == (5, 8)
    assert kind.accepts(np.zeros((5, 8), np.float16), field)
    for bad in (np.zeros(5, np.float32), np.zeros((5, 8), np.int64), "feats.npy"):
        assert not kind.accepts(bad, field)
        with pytest.raises(TypeError):
            kind.check(bad, field, None, output=False)
    nan = np.full((5, 8), np.nan, np.float32)
    assert not kind.accepts(nan, field)
    with pytest.raises(ValueError, match="NaN"):
        kind.check(nan, field, None, output=False)
    with pytest.raises(NotImplementedError):
        kind.join(matrix, matrix)
