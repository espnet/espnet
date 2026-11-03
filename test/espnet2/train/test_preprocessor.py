import logging

import numpy as np
import pytest

from espnet2.train.preprocessor import (
    CommonPreprocessor,
    LIDMultiLabelPreprocessor,
    LIDSoftLabelPreprocessor,
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


@pytest.fixture()
def lang2utt(tmp_path):
    """Write a three-language lang2utt file."""
    path = tmp_path / "lang2utt"
    path.write_text("ara u1\neng u2\njpn u3\n", encoding="utf-8")
    return path


def _label_set_preprocessor(preprocessor_class, lang2utt, **kwargs):
    """Build a LID label-set preprocessor without data augmentation."""
    return preprocessor_class(
        train=False,
        lang2utt=str(lang2utt),
        fix_duration=False,
        noise_apply_prob=0.0,
        rir_apply_prob=0.0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "preprocessor_class,raw,expected",
    [
        (LIDSoftLabelPreprocessor, "eng", [0.0, 1.0, 0.0]),
        (LIDSoftLabelPreprocessor, "ara eng", [0.5, 0.5, 0.0]),
        (LIDSoftLabelPreprocessor, "jpn ara eng", [1 / 3, 1 / 3, 1 / 3]),
        (LIDMultiLabelPreprocessor, "eng", [0.0, 1.0, 0.0]),
        (LIDMultiLabelPreprocessor, "ara eng", [1.0, 1.0, 0.0]),
        (LIDMultiLabelPreprocessor, "jpn ara eng", [1.0, 1.0, 1.0]),
    ],
)
def test_lid_target_preprocessor_label_sets(
    lang2utt, preprocessor_class, raw, expected
):
    """Soft labels split the mass over the languages; multi-labels set each to 1."""
    preprocessor = _label_set_preprocessor(preprocessor_class, lang2utt)
    speech = np.zeros(1600, dtype=np.float32)
    data = preprocessor("u1", {"speech": speech, "lid_labels": raw})
    assert data["lid_labels"].dtype == np.float32
    np.testing.assert_allclose(data["lid_labels"], expected)


@pytest.mark.parametrize(
    "preprocessor_class", [LIDSoftLabelPreprocessor, LIDMultiLabelPreprocessor]
)
@pytest.mark.parametrize(
    "raw,error,message",
    [
        ("", ValueError, "empty"),
        ("eng eng", ValueError, "duplicate"),
        ("missing", KeyError, "unknown"),
    ],
)
def test_lid_target_preprocessor_rejects_invalid_labels(
    lang2utt, preprocessor_class, raw, error, message
):
    """Empty, duplicate and unknown labels raise errors."""
    preprocessor = _label_set_preprocessor(preprocessor_class, lang2utt)
    with pytest.raises(error, match=message):
        preprocessor("u1", {"speech": np.zeros(1600, np.float32), "lid_labels": raw})


@pytest.mark.parametrize(
    "preprocessor_class", [LIDSoftLabelPreprocessor, LIDMultiLabelPreprocessor]
)
def test_lid_target_preprocessor_rejects_lang2vec_table(
    tmp_path, lang2utt, preprocessor_class
):
    """lang2vec targets are not defined for label sets and are rejected."""
    table = tmp_path / "lang2vec"
    table.write_text("ara 0.1 0.2\neng 0.3 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lang2vec_table"):
        _label_set_preprocessor(preprocessor_class, lang2utt, lang2vec_table=str(table))


@pytest.mark.parametrize(
    "preprocessor_class", [LIDSoftLabelPreprocessor, LIDMultiLabelPreprocessor]
)
def test_lid_target_preprocessor_passes_speech_only_data(lang2utt, preprocessor_class):
    """Data without lid_labels (inference, statistics) is processed unchanged."""
    preprocessor = _label_set_preprocessor(preprocessor_class, lang2utt)
    speech = np.zeros(1600, dtype=np.float32)
    data = preprocessor("u1", {"speech": speech})
    assert list(data) == ["speech"]
    np.testing.assert_array_equal(data["speech"], speech)
