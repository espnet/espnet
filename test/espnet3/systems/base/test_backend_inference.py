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
    backend = SimpleNamespace(sample_rate=8000)
    assert Wrapped(backend)(np.zeros(8, dtype=np.float32))["text"] == "8"
    with pytest.raises(TypeError, match="declares no backend_class"):
        Wrapped(device="cpu")


def test_the_rate_is_read_off_the_backend_in_order():
    assert Wrapped(SimpleNamespace(sample_rate=8000)).sample_rate == 8000
    assert Wrapped(SimpleNamespace(fs="22.05k")).sample_rate == 22050
    # a toolkit's config layout is the system's knowledge, not the base's
    args = SimpleNamespace(frontend_conf={"fs": "16k"})
    for silent in (SimpleNamespace(), object(), SimpleNamespace(tts_train_args=args)):
        with pytest.raises(TypeError, match="cannot tell the rate"):
            Wrapped(silent).sample_rate


@pytest.mark.parametrize(
    "path, why",
    [
        ("this.s", "this.s"),  # importing it alone runs code
        ("subprocess.Popen", "subprocess.Popen"),
        ("espnet3.systems.base.inference_provider.os.system", "resolves to"),
        ("espnet2.bin.launch.main", "not a class"),
    ],
)
def test_a_backend_class_argument_must_be_an_espnet_class(path, why, capsys):
    """A bundle's inference.yaml can set it: it is checked before any import."""
    from espnet3.systems.esp2_asr.inference import Inference

    with pytest.raises(ValueError, match=f"must be an ESPnet2 or ESPnet3 class.*{why}"):
        Inference(backend_class=path)
    assert "Zen of Python" not in capsys.readouterr().out
