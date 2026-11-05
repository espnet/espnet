import logging

import numpy as np
import pytest

from espnet2.train.preprocessor import (
    CommonPreprocessor,
    Qwen2AudioPreprocessor,
    S2TCTCPreprocessor,
    S2TPreprocessor,
)


def _build_preprocessor(**kwargs):
    return CommonPreprocessor(
        train=False,
        token_type="word",
        token_list=["<unk>", "hello"],
        **kwargs,
    )


@pytest.mark.execution_timeout(300)
def test_qwen2audio_preprocessor():
    preprocessor = Qwen2AudioPreprocessor(sampling_rate=16000)
    assert preprocessor.sampling_rate == 16000

    data = {
        "text": "hello world",
        "speech": np.random.randn(16000).astype(np.float32),
    }

    out = preprocessor("utt1", data)

    assert isinstance(out["input_ids"], np.ndarray)
    assert isinstance(out["attention_mask"], np.ndarray)
    assert isinstance(out["input_features"], np.ndarray)
    assert isinstance(out["feature_attention_mask"], np.ndarray)
    assert out["input_ids"].ndim == 1
    assert out["input_features"].ndim == 2


@pytest.mark.execution_timeout(10)
def test_qwen2audio_preprocessor_invalid_sampling_rate():
    with pytest.raises(NotImplementedError, match="16kHz only"):
        Qwen2AudioPreprocessor(sampling_rate=8000)


@pytest.mark.execution_timeout(10)
def test_long_text_warns_only_once(caplog):
    preprocessor = _build_preprocessor()
    long_text = " ".join(["hello"] * 600)

    with caplog.at_level(logging.WARNING):
        for i in range(5):
            preprocessor(f"utt{i}", {"text": long_text})

    warnings = [r for r in caplog.records if "exceeds" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.execution_timeout(10)
def test_short_text_does_not_warn(caplog):
    preprocessor = _build_preprocessor()

    with caplog.at_level(logging.WARNING):
        preprocessor("utt", {"text": "hello hello"})

    assert not [r for r in caplog.records if "exceeds" in r.getMessage()]


@pytest.mark.execution_timeout(10)
def test_threshold_is_configurable(caplog):
    preprocessor = _build_preprocessor(text_length_warning_thres=10)

    with caplog.at_level(logging.WARNING):
        preprocessor("utt", {"text": " ".join(["hello"] * 20)})

    warnings = [r for r in caplog.records if "exceeds" in r.getMessage()]
    assert len(warnings) == 1
    assert "10" in warnings[0].getMessage()


@pytest.mark.execution_timeout(10)
def test_warning_can_be_disabled(caplog):
    preprocessor = _build_preprocessor(text_length_warning_thres=0)

    with caplog.at_level(logging.WARNING):
        preprocessor("utt", {"text": " ".join(["hello"] * 600)})

    assert not [r for r in caplog.records if "exceeds" in r.getMessage()]


@pytest.mark.execution_timeout(10)
def test_pretokenized_text_is_passed_through(caplog):
    preprocessor = _build_preprocessor()

    with caplog.at_level(logging.WARNING):
        out = preprocessor("utt", {"text": np.arange(600)})

    assert out["text"].shape == (600,)
    assert not [r for r in caplog.records if "exceeds" in r.getMessage()]


@pytest.mark.parametrize(
    "preprocessor_class", [S2TPreprocessor, S2TCTCPreprocessor], ids=["s2t", "s2t_ctc"]
)
def test_the_s2t_preprocessors_pass_the_augmentation_through(preprocessor_class):
    # both inherit _speech_process, and neither forwarded the arguments that
    # turn it on; the CTC one is what OWSM-CTC trains with
    effects = [[1.0, "speed_perturb", {"factor": 1.1}]]
    preprocessor = preprocessor_class(
        train=True,
        token_type="word",
        # the symbols these preprocessors look up when they are built
        token_list=[
            "<unk>",
            "hello",
            "<notimestamps>",
            "<0.00>",
            "<30.00>",
            "<nolang>",
        ],
        data_aug_effects=effects,
        data_aug_num=[1, 1],
        data_aug_prob=1.0,
    )
    assert preprocessor.data_aug is not None
    assert preprocessor.data_aug_prob == 1.0


def _write_delayed_impulse_rir(tmp_path, delay):
    import soundfile

    rir = np.zeros(64)
    rir[delay] = 1.0
    wav = tmp_path / "rir.wav"
    soundfile.write(wav, rir, 16000, subtype="FLOAT")
    scp = tmp_path / "rir.scp"
    scp.write_text(f"rir1 {wav}\n")
    return str(scp)


@pytest.mark.parametrize("remove_delay", [False, True])
def test_rir_remove_delay_keeps_speech_aligned(tmp_path, remove_delay):
    # an RIR that is a pure impulse after `delay` samples only shifts the signal,
    # so removing the direct-path delay must give back the input unchanged
    delay = 20
    preprocessor = CommonPreprocessor(
        train=True,
        rir_scp=_write_delayed_impulse_rir(tmp_path, delay),
        rir_apply_prob=1.0,
        rir_remove_delay=remove_delay,
    )
    rng = np.random.default_rng(0)
    speech = (0.1 * rng.standard_normal(1600)).astype(np.float64)
    out = preprocessor("utt1", {"speech": speech.copy()})["speech"].reshape(-1)
    assert out.shape == speech.shape
    if remove_delay:
        np.testing.assert_allclose(out, speech, rtol=1e-5, atol=1e-6)
    else:
        # the tail falls off the end and the power is renormalized, so compare
        # the shifted signal up to a scale factor
        np.testing.assert_allclose(out[:delay], 0.0, atol=1e-12)
        scale = np.dot(out[delay:], speech[:-delay]) / np.dot(
            speech[:-delay], speech[:-delay]
        )
        np.testing.assert_allclose(out[delay:], scale * speech[:-delay], atol=1e-6)
