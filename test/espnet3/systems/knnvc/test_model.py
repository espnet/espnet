"""Tests for the kNN-VC inference model (with a stub encoder)."""

from test.espnet3.systems.knnvc import fixtures
from test.espnet3.systems.knnvc.tiny import TINY_GENERATOR, TINY_HOP

import numpy as np
import pytest
import torch

import espnet3.systems.knnvc.model as model_module
from espnet3.systems.knnvc.model import KNNVCModel
from espnet3.systems.knnvc.vocoder import KNNVCGenerator

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_call_returns_waveform                  | Output is float32 (samples,) |
# |                          | with frames * hop samples; loudness applied.    |
# | test_matching_set_cached_by_key             | Same target_speaker reuses   |
# |                          | the encoded references; the cache stays bounded. |
# | test_reference_speech_validation            | Empty references raise.      |
# | test_loudness_disabled                      | tgt_loudness_db=None leaves  |
# |                                             | the vocoder output as is.    |
# | test_trim_silence_removes_padding           | VAD never crashes/empties a  |
# |                          | reference; level <= 0 disables it; source kept. |
# | test_build_encoder_accepts_instance_or_config | An `encoder:` block already  |
# |                          | instantiated by Hydra is used as-is; a mapping  |
# |                          | is instantiated; anything else raises TypeError.|
# | test_build_encoder_needs_a_checkpoint_without_an_encoder                 |
# |                          | No encoder and no checkpoint raises clearly.    |

# TINY_GENERATOR consumes 6-dim features; make the stub encoder emit 6 dims.
_DIM = TINY_GENERATOR["in_channels"]


class StubEncoder(fixtures.DummyEncoder):
    """DummyEncoder variant with the generator's input dimension."""

    def __init__(self, checkpoint=None, layer=6, device="cpu"):
        super().__init__(device=device)
        generator = torch.Generator().manual_seed(1)
        self.projection = torch.nn.Parameter(
            torch.randn(fixtures.HOP_LENGTH, _DIM, generator=generator),
            requires_grad=False,
        )

    @property
    def output_dim(self):
        return _DIM


@pytest.fixture
def vocoder_checkpoint(tmp_path):
    torch.manual_seed(0)
    generator = KNNVCGenerator(**TINY_GENERATOR)
    path = tmp_path / "generator.pth"
    torch.save({f"generator.{k}": v for k, v in generator.state_dict().items()}, path)
    return path


@pytest.fixture
def model(monkeypatch, vocoder_checkpoint):
    monkeypatch.setattr(model_module, "WavLMEncoder", StubEncoder)
    return KNNVCModel(
        vocoder_checkpoint=vocoder_checkpoint,
        wavlm_checkpoint="unused.pt",
        generator=TINY_GENERATOR,
        topk=2,
        tgt_loudness_db=-16.0,
        device="cpu",
    )


def _wav(seconds, seed):
    rng = np.random.RandomState(seed)
    return (rng.randn(int(seconds * fixtures.SAMPLE_RATE)) * 0.1).astype(np.float32)


def test_call_returns_waveform(model):
    source = _wav(1.0, 0)
    references = [_wav(0.5, 1), _wav(0.7, 2)]
    out = model(source, references, target_speaker="spkX")

    frames = len(source) // fixtures.HOP_LENGTH
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.shape == (frames * TINY_HOP,)
    assert np.isfinite(out).all()


def test_matching_set_cached_by_key(model):
    references = [_wav(0.5, 1)]
    first = model.get_matching_set(references, cache_key="spk")
    second = model.get_matching_set([_wav(0.9, 5)], cache_key="spk")
    assert first is second
    uncached = model.get_matching_set([_wav(0.9, 5)])
    assert uncached.shape[0] != first.shape[0]

    # The cache holds tensors on the compute device, so it must stay bounded.
    assert model.max_cached_speakers == 1
    model.get_matching_set([_wav(0.6, 6)], cache_key="other")
    assert list(model._matching_set_cache) == ["other"]
    model.max_cached_speakers = 0
    model.get_matching_set(references, cache_key="nocache")
    assert "nocache" not in model._matching_set_cache


def test_reference_speech_validation(model):
    with pytest.raises(ValueError, match="at least one"):
        model.get_matching_set([])
    single = model.get_matching_set(_wav(0.5, 3))
    assert single.shape == (len(_wav(0.5, 3)) // fixtures.HOP_LENGTH, _DIM)


def test_loudness_disabled(model):
    model.tgt_loudness_db = None
    source = _wav(1.0, 0)
    references = [_wav(0.5, 1)]
    out = model(source, references)
    feats = model.get_features(source)
    matched = model_module.match_features(
        feats, model.get_matching_set(references), topk=2
    )
    expected = model.vocode(matched).numpy()
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_trim_silence_removes_padding(model):
    rng = np.random.RandomState(11)
    # Speech-like burst (noise with a slowly varying envelope) between silences.
    burst = (rng.randn(16000) * np.linspace(0.05, 0.5, 16000)).astype(np.float32)
    silence = np.zeros(8000, np.float32)
    padded = np.concatenate([silence, burst, silence])

    trimmed = model.trim_silence(padded)
    assert trimmed.dtype == torch.float32 and trimmed.dim() == 1
    # Silence must actually go: `<=` would also pass if nothing were trimmed.
    assert trimmed.numel() < len(padded)
    # ... but the speech must survive it.
    assert trimmed.numel() >= len(burst) // 2

    # An all-silent reference is kept untouched instead of vanishing.
    assert model.trim_silence(silence).numel() == len(silence)
    # A non-positive trigger level disables trimming entirely.
    model.vad_trigger_level = 0.0
    assert model.trim_silence(padded).numel() == len(padded)
    # References are trimmed before encoding; the source never is.
    model.vad_trigger_level = 7.0
    matching = model.get_matching_set([padded])
    assert matching.shape[0] < len(padded) // fixtures.HOP_LENGTH
    assert model.get_features(padded).shape[0] == len(padded) // fixtures.HOP_LENGTH


def test_build_encoder_accepts_instance_or_config():
    """Hydra instantiates a nested `encoder:` block before KNNVCModel runs."""
    instance = StubEncoder()
    assert (
        KNNVCModel.build_encoder(instance, "unused.pt", 6, torch.device("cpu"))
        is instance
    )

    from_config = KNNVCModel.build_encoder(
        {"_target_": f"{StubEncoder.__module__}.{StubEncoder.__qualname__}"},
        "unused.pt",
        6,
        torch.device("cpu"),
    )
    assert isinstance(from_config, StubEncoder)

    with pytest.raises(TypeError, match="`encoder` must be None"):
        KNNVCModel.build_encoder(object(), "unused.pt", 6, torch.device("cpu"))


def test_build_encoder_needs_a_checkpoint_without_an_encoder():
    """With no encoder and no WavLM checkpoint there is nothing to build.

    The checkpoint has no package default: the recipes supply it through
    ``wavlm_checkpoint`` in ``egs3/TEMPLATE/knnvc/conf``.
    """
    with pytest.raises(ValueError, match="wavlm_checkpoint"):
        KNNVCModel.build_encoder(None, None, 6, torch.device("cpu"))
