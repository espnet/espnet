import numpy as np

from espnet3.systems.esp2_enh.inference import Inference


class FakeSeparateSpeech:
    """Returns twice the input as speaker 1 and silence as speaker 2."""

    def __init__(self):
        self.calls = []

    def __call__(self, speech_mix, fs):
        self.calls.append((speech_mix.copy(), fs))
        return [speech_mix * 2.0, np.zeros_like(speech_mix)]


def test_batch_is_one_padded_call_and_each_output_keeps_its_length():
    backend = FakeSeparateSpeech()
    model = Inference(backend)

    out = model.batch(
        [
            {"speech_mix": (16000, np.full(800, 0.1, dtype=np.float32))},
            {"speech_mix": (16000, np.full(400, 0.2, dtype=np.float32))},
        ]
    )

    assert len(backend.calls) == 1
    batch, fs = backend.calls[0]
    assert batch.shape == (2, 800) and fs == 16000
    assert not batch[1, 400:].any()
    assert [o["enhanced"].array.shape for o in out] == [(800,), (400,)]
    np.testing.assert_allclose(out[0]["enhanced"].array, 0.2)
    np.testing.assert_allclose(out[1]["enhanced"].array, 0.4)
    assert out[0]["enhanced"].rate == 16000


def test_output_that_would_clip_is_scaled_to_a_0_9_peak():
    model = Inference(FakeSeparateSpeech())

    wave = np.array([0.1, -0.75, 0.5], dtype=np.float32)
    enhanced = model((16000, wave))["enhanced"].array

    np.testing.assert_allclose(enhanced, [0.12, -0.9, 0.6], rtol=1e-6)


def test_input_is_resampled_to_the_model_rate():
    backend = FakeSeparateSpeech()
    model = Inference(backend, sample_rate=8000)

    enhanced = model((16000, np.zeros(1600, dtype=np.float32)))["enhanced"]

    assert backend.calls[0][0].shape == (1, 800)
    assert backend.calls[0][1] == 8000
    assert enhanced.rate == 8000
