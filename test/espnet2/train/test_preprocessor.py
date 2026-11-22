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


def test_rir_delay_arguments_are_last():
    """Positional arguments of older callers keep their meaning."""
    import inspect

    params = list(inspect.signature(CommonPreprocessor.__init__).parameters)
    assert params[-2:] == ["rir_remove_delay", "rir_delay_per_channel"]
    # the arguments before it keep the order they had without it
    assert params[params.index("rir_apply_prob") + 1] == "noise_scp"


@pytest.mark.parametrize("fs, expected_delay", [(16000, 40), (0, 20)])
def test_rir_at_another_rate_is_resampled_to_fs(tmp_path, fs, expected_delay):
    """An 8 kHz RIR on 16 kHz speech: with fs, its 20-sample delay becomes 40."""
    import soundfile

    if fs:
        pytest.importorskip("resampy")  # librosa's kaiser_fast (espnet[egs2])

    rir = np.zeros(64)
    rir[20] = 1.0
    wav = tmp_path / "rir8k.wav"
    soundfile.write(wav, rir, 8000, subtype="FLOAT")
    (tmp_path / "rir.scp").write_text(f"rir1 {wav}\n")
    preprocessor = CommonPreprocessor(
        train=True, rir_scp=str(tmp_path / "rir.scp"), rir_apply_prob=1.0, fs=fs
    )
    rng = np.random.default_rng(0)
    speech = 0.1 * rng.standard_normal(1600)
    out = preprocessor("utt1", {"speech": speech.copy()})["speech"].reshape(-1)
    corr = [np.dot(out[k:], speech[: len(speech) - k]) for k in range(64)]
    assert int(np.argmax(corr)) == expected_delay


@pytest.mark.parametrize("per_channel, expected", [(False, [0, 20]), (True, [0, 0])])
def test_rir_delay_of_a_multichannel_rir(tmp_path, per_channel, expected):
    """A two-channel RIR with peaks at samples 10 and 30.

    One shift (the overall peak) keeps the 20-sample difference between the
    channels; per channel, both channels are aligned.
    """
    import soundfile

    rir = np.zeros((64, 2))
    rir[10, 0] = 1.0
    rir[30, 1] = 0.9
    wav = tmp_path / "rir2ch.wav"
    soundfile.write(wav, rir, 16000, subtype="FLOAT")
    (tmp_path / "rir.scp").write_text(f"rir1 {wav}\n")
    preprocessor = CommonPreprocessor(
        train=True,
        rir_scp=str(tmp_path / "rir.scp"),
        rir_apply_prob=1.0,
        rir_remove_delay=True,
        rir_delay_per_channel=per_channel,
    )
    rng = np.random.default_rng(0)
    speech = 0.1 * rng.standard_normal(1600)
    out = preprocessor("utt1", {"speech": speech.copy()})["speech"]
    assert out.shape == (1600, 2)
    shifts = []
    for ch in range(2):
        corr = [np.dot(out[k:, ch], speech[: 1600 - k]) for k in range(64)]
        shifts.append(int(np.argmax(corr)))
    assert shifts == expected
