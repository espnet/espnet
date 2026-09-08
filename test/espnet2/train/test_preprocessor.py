import logging

import numpy as np
import pytest

from espnet2.train.preprocessor import CommonPreprocessor, Qwen2AudioPreprocessor


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
