"""CPU-only tests for how MiniOmniE2EModel passes its device to OmniInference.

espnet2.sds.end_to_end.mini_omni.inference needs snac, litgpt and
openai-whisper, none of which ci/install.sh installs, and importing it would
also pull a 2.9 GB checkpoint. __init__ imports it lazily, so a stub in
sys.modules is enough to exercise the wiring without any of that.
"""

import sys
import types

import pytest

from espnet2.sds.end_to_end import mini_omni_e2e as mod

pytest.importorskip("huggingface_hub")


@pytest.fixture
def recorded(monkeypatch):
    """Stub out the checkpoint download and OmniInference, recording its args."""
    seen = {}

    class _RecordingOmniInference:
        def __init__(self, ckpt_dir="./checkpoint", device="cuda:0"):
            seen["ckpt_dir"] = ckpt_dir
            seen["device"] = device

    stub = types.ModuleType("espnet2.sds.end_to_end.mini_omni.inference")
    stub.OmniInference = _RecordingOmniInference
    monkeypatch.setitem(sys.modules, "espnet2.sds.end_to_end.mini_omni.inference", stub)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *args, **kwargs: None
    )
    # pydub is only needed for the audio path, which these tests do not reach.
    monkeypatch.setattr(mod, "is_pydub_available", True, raising=False)
    return seen


def test_device_argument_reaches_omni_inference(recorded):
    """A caller asking for CPU must not get a client built on CUDA.

    The device was previously hardcoded, so `device="cpu"` still initialised on
    CUDA and raised "Torch not compiled with CUDA enabled" on a CPU-only host.
    """
    model = mod.MiniOmniE2EModel(device="cpu", dtype="float32")

    assert recorded["device"] == "cpu"
    assert model.device == "cpu"


def test_device_still_defaults_to_cuda(recorded):
    """The default must not change, so existing GPU callers are unaffected."""
    model = mod.MiniOmniE2EModel()

    assert recorded["device"] == "cuda"
    assert model.device == "cuda"
