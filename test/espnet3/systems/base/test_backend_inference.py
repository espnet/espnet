"""BackendInference: the base a wrapped-backend system stands on."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from espnet3.api.inference import Field
from espnet3.systems.base.backend_inference import BackendInference


class Wrapped(BackendInference):
    inputs = (Field("speech", "audio"),)
    outputs = (Field("text", "text"),)

    def run(self, speech):
        return {"text": str(len(speech.array))}


def test_a_built_backend_is_kept_and_no_class_is_needed():
    assert Wrapped(SimpleNamespace())(np.zeros(8, dtype=np.float32))["text"] == "8"
    with pytest.raises(TypeError, match="declares no backend_class"):
        Wrapped(device="cpu")


def test_the_rate_is_read_off_the_backend_in_order():
    assert Wrapped(SimpleNamespace(sample_rate=8000)).sample_rate == 8000
    assert Wrapped(SimpleNamespace(fs="22.05k")).sample_rate == 22050
    args = SimpleNamespace(frontend_conf={"fs": "16k"})
    assert Wrapped(SimpleNamespace(tts_train_args=args)).sample_rate == 16000
    assert Wrapped(SimpleNamespace()).sample_rate == 16000
    assert Wrapped(object()).sample_rate == 16000
