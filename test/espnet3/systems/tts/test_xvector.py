"""Tests for the x-vector provider and runner used by compute_xvectors."""

import builtins
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from espnet3.systems.tts.xvector_provider import XVectorProvider
from espnet3.systems.tts.xvector_runner import XVectorRunner

# ===============================================================
# Test Case Summary
# ===============================================================
#
# manifest loading
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_load_manifest_groups_by_speaker        | TSV rows become utterances   |
# |                                             | plus a speaker mapping.      |
# | test_load_manifest_skips_blank_lines        | Blank lines are ignored.     |
#
# provider environments
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_env_local_returns_worker_env     | The driver env carries the   |
# |                          | model, manifest, and an existing output_dir.     |
# | test_build_env_local_requires_config        | Missing xvector/manifest/    |
# |                                             | output_dir all raise.        |
# | test_build_env_local_rejects_empty_manifest | An empty manifest raises.    |
# | test_worker_setup_fn_builds_same_env        | The worker setup function    |
# |                                             | produces the same env.       |
# | test_worker_setup_fn_requires_config        | Same validation in a worker. |
# | test_has_cuda_reports_torch_availability    | _has_cuda follows torch.     |
#
# model construction
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_model_speechbrain                | Delegates to speechbrain's   |
# |                                             | EncoderClassifier.           |
# | test_build_model_espnet_model_tag_vs_file   | A .pth path becomes          |
# |                                             | model_file, a tag model_tag. |
# | test_build_model_rejects_unknown_toolkit    | Unknown toolkit raises.      |
#
# extraction and persistence
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_writes_embedding               | forward() saves <utt>.pt and |
# |                                             | reports status 'ok'.         |
# | test_forward_skips_existing                 | An existing .pt is skipped   |
# |                                             | without re-reading audio.    |
# | test_forward_accepts_iterable               | An iterable index yields a   |
# |                                             | list of status dicts.        |
# | test_forward_converts_embedding_types       | ndarray/tensor/list are all  |
# |                                             | stored as float32 tensors.   |
# | test_load_audio_mixes_stereo_to_mono        | Multi-channel input is mixed |
# |                                             | down to mono float32.        |
# | test_load_audio_keeps_mono_untouched        | Mono input keeps its shape,  |
# |                                             | dtype and native rate.       |
# | test_extract_embedding_dispatches_toolkit   | Each toolkit reaches its own |
# |                                             | extractor; unknown raises.   |
# | test_extract_speechbrain_uses_encode_batch  | The speechbrain path calls   |
# |                                             | encode_batch and unwraps it. |
# | test_extract_espnet_resamples_and_mixes     | The espnet path resamples    |
# |                                             | and mixes down to mono.      |
# | test_extract_rawnet_pads_short_audio        | RawNet pads short clips into |
# |                                             | ten 3-second segments.       |


def _write_manifest(tmp_path, rows):
    path = tmp_path / "train.tsv"
    path.write_text("".join(rows), encoding="utf-8")
    return path


def _wav(tmp_path, name="a.wav", seconds=1.0, sr=16000):
    path = tmp_path / name
    sf.write(path, np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return path


@pytest.fixture
def manifest(tmp_path):
    """A two-speaker, three-utterance manifest with real wav files."""
    rows = []
    for utt_id, spk in (("u1", "0"), ("u2", "0"), ("u3", "1")):
        wav_path = _wav(tmp_path, f"{utt_id}.wav")
        rows.append(f"{utt_id}\t{wav_path}\thello\t{spk}\n")
    return _write_manifest(tmp_path, rows)


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the network-bound model build and the extractor."""
    monkeypatch.setattr(
        XVectorProvider, "_build_model", staticmethod(lambda *a, **k: "MODEL")
    )
    monkeypatch.setattr(
        XVectorRunner,
        "_extract_embedding",
        staticmethod(
            lambda wav, sr, model, toolkit, device: np.zeros(192, dtype=np.float32)
        ),
    )


@pytest.fixture
def fake_speechbrain(monkeypatch):
    """Install a stub ``speechbrain`` so its code paths run without the dep.

    ``speechbrain`` is an optional extra, so without this the two branches
    that import it would only ever be skipped, never covered.
    """
    created = []
    for name in (
        "speechbrain",
        "speechbrain.dataio",
        "speechbrain.dataio.preprocess",
        "speechbrain.inference",
        "speechbrain.inference.classifiers",
    ):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
            created.append(name)

    preprocess = sys.modules["speechbrain.dataio.preprocess"]
    if not hasattr(preprocess, "AudioNormalizer"):

        class _AudioNormalizer:
            def __call__(self, wav, sample_rate):
                return wav

        monkeypatch.setattr(
            preprocess, "AudioNormalizer", _AudioNormalizer, raising=False
        )
    return sys.modules["speechbrain.inference.classifiers"]


def _config(**xvector):
    return OmegaConf.create({"xvector": xvector or {"toolkit": "speechbrain"}})


def _provider(manifest, tmp_path, **xvector):
    return XVectorProvider(
        _config(**xvector),
        params={
            "manifest_path": str(manifest),
            "output_dir": str(tmp_path / "xvec"),
        },
    )


# ---------------------------------------------------------------
# manifest loading
# ---------------------------------------------------------------


def test_load_manifest_groups_by_speaker(manifest):
    utterances, speaker_to_utterances = XVectorProvider._load_manifest(manifest)

    assert [utt_id for utt_id, _ in utterances] == ["u1", "u2", "u3"]
    assert speaker_to_utterances == {"0": ["u1", "u2"], "1": ["u3"]}


def test_load_manifest_skips_blank_lines(tmp_path):
    path = _write_manifest(tmp_path, ["u1\t/a.wav\thello\t0\n", "\n", "\n"])

    utterances, _ = XVectorProvider._load_manifest(path)

    assert utterances == [("u1", "/a.wav")]


# ---------------------------------------------------------------
# provider environments
# ---------------------------------------------------------------


def test_build_env_local_returns_worker_env(manifest, tmp_path, stub_model):
    env = _provider(manifest, tmp_path).build_env_local()

    assert sorted(env) == [
        "config",
        "device",
        "model",
        "output_dir",
        "speaker_to_utterances",
        "toolkit",
        "utterances",
    ]
    assert env["model"] == "MODEL"
    assert env["toolkit"] == "speechbrain"
    assert len(env["utterances"]) == 3
    # The stage writes straight into this directory, so it must already exist.
    assert env["output_dir"].is_dir()


def test_build_env_local_requires_config(manifest, tmp_path, stub_model):
    no_xvector = XVectorProvider(OmegaConf.create({}), params={})
    with pytest.raises(RuntimeError, match="xvector configuration not found"):
        no_xvector.build_env_local()

    no_manifest = XVectorProvider(_config(), params={})
    with pytest.raises(RuntimeError, match="provide manifest_path"):
        no_manifest.build_env_local()

    no_output = XVectorProvider(_config(), params={"manifest_path": str(manifest)})
    with pytest.raises(RuntimeError, match="output_dir must be provided"):
        no_output.build_env_local()


def test_build_env_local_rejects_empty_manifest(tmp_path, stub_model):
    empty = _write_manifest(tmp_path, [])
    provider = _provider(empty, tmp_path)

    with pytest.raises(RuntimeError, match="No utterances found"):
        provider.build_env_local()


def test_worker_setup_fn_builds_same_env(manifest, tmp_path, stub_model):
    provider = _provider(manifest, tmp_path)

    setup = provider.build_worker_setup_fn()
    env = setup()

    assert sorted(env) == sorted(provider.build_env_local())
    assert env["model"] == "MODEL"
    assert len(env["utterances"]) == 3


def test_worker_setup_fn_requires_config(manifest, tmp_path, stub_model):
    """The worker repeats the driver's validation, since it re-reads config."""
    with pytest.raises(RuntimeError, match="xvector configuration not found"):
        XVectorProvider(OmegaConf.create({}), params={}).build_worker_setup_fn()()

    with pytest.raises(RuntimeError, match="provide manifest_path"):
        XVectorProvider(_config(), params={}).build_worker_setup_fn()()

    with pytest.raises(RuntimeError, match="output_dir must be provided"):
        XVectorProvider(
            _config(), params={"manifest_path": str(manifest)}
        ).build_worker_setup_fn()()

    empty = _write_manifest(tmp_path, [])
    with pytest.raises(RuntimeError, match="No utterances found"):
        XVectorProvider(
            _config(),
            params={
                "manifest_path": str(empty),
                "output_dir": str(tmp_path / "xvec"),
            },
        ).build_worker_setup_fn()()


def test_has_cuda_reports_torch_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert XVectorProvider._has_cuda() is True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert XVectorProvider._has_cuda() is False


# ---------------------------------------------------------------
# model construction
# ---------------------------------------------------------------


def test_build_model_speechbrain(monkeypatch, fake_speechbrain):
    calls = {}

    class _EncoderClassifier:
        @staticmethod
        def from_hparams(source, run_opts):
            calls["source"] = source
            calls["run_opts"] = run_opts
            return "SB_MODEL"

    monkeypatch.setattr(
        fake_speechbrain, "EncoderClassifier", _EncoderClassifier, raising=False
    )

    model = XVectorProvider._build_model(
        "speechbrain", "speechbrain/spkrec-ecapa-voxceleb", "cpu"
    )

    assert model == "SB_MODEL"
    assert calls == {
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        "run_opts": {"device": "cpu"},
    }


def test_build_model_espnet_model_tag_vs_file(monkeypatch):
    """A local .pth goes to model_file; anything else is treated as a tag."""
    seen = []

    class _Speech2Embedding:
        @staticmethod
        def from_pretrained(**kwargs):
            seen.append(kwargs)
            return "ESPNET_MODEL"

    import espnet2.bin.spk_inference as spk_inference

    monkeypatch.setattr(spk_inference, "Speech2Embedding", _Speech2Embedding)

    XVectorProvider._build_model("espnet", "/models/spk.pth", "cpu")
    assert seen[-1]["model_file"] == "/models/spk.pth"
    assert seen[-1]["model_tag"] is None

    XVectorProvider._build_model("espnet", "espnet/some_model", "cpu")
    assert seen[-1]["model_tag"] == "espnet/some_model"
    assert seen[-1]["model_file"] is None


def test_build_model_rejects_unknown_toolkit():
    with pytest.raises(ValueError, match="Unknown toolkit: nope"):
        XVectorProvider._build_model("nope", "model", "cpu")


# ---------------------------------------------------------------
# extraction and persistence
# ---------------------------------------------------------------


def test_forward_writes_embedding(manifest, tmp_path, stub_model):
    env = _provider(manifest, tmp_path).build_env_local()

    result = XVectorRunner.forward(0, **env)

    assert result == {"utt_id": "u1", "status": "ok"}
    saved = torch.load(str(env["output_dir"] / "u1.pt"))
    assert saved.dtype == torch.float32
    assert saved.shape == (192,)


def test_forward_skips_existing(manifest, tmp_path, stub_model, monkeypatch):
    env = _provider(manifest, tmp_path).build_env_local()
    XVectorRunner.forward(0, **env)

    def _fail(*args, **kwargs):
        raise AssertionError("audio must not be re-read for an existing .pt")

    monkeypatch.setattr(XVectorRunner, "_load_audio", staticmethod(_fail))

    assert XVectorRunner.forward(0, **env) == {"utt_id": "u1", "status": "skipped"}


def test_forward_accepts_iterable(manifest, tmp_path, stub_model):
    env = _provider(manifest, tmp_path).build_env_local()

    results = XVectorRunner.forward(range(3), **env)

    assert [r["utt_id"] for r in results] == ["u1", "u2", "u3"]
    assert {r["status"] for r in results} == {"ok"}


@pytest.mark.parametrize(
    "embedding",
    [
        np.zeros(4, dtype=np.float64),
        torch.zeros(4, dtype=torch.float64),
        [0.0, 0.0, 0.0, 0.0],
    ],
)
def test_forward_converts_embedding_types(manifest, tmp_path, monkeypatch, embedding):
    monkeypatch.setattr(
        XVectorProvider, "_build_model", staticmethod(lambda *a, **k: "MODEL")
    )
    monkeypatch.setattr(
        XVectorRunner,
        "_extract_embedding",
        staticmethod(lambda wav, sr, model, toolkit, device: embedding),
    )
    env = _provider(manifest, tmp_path).build_env_local()

    XVectorRunner.forward(0, **env)

    saved = torch.load(str(env["output_dir"] / "u1.pt"))
    assert saved.dtype == torch.float32
    assert saved.shape == (4,)


def test_load_audio_mixes_stereo_to_mono(tmp_path):
    """_load_audio owns the mono mix-down that librosa.load used to provide."""
    path = tmp_path / "stereo.wav"
    stereo = np.stack(
        [np.full(16, 1.0, dtype=np.float32), np.full(16, -0.5, dtype=np.float32)],
        axis=-1,
    )
    sf.write(path, stereo, 16000)

    wav, in_sr = XVectorRunner._load_audio(path)

    assert in_sr == 16000
    assert wav.ndim == 1
    assert wav.shape == (16,)
    assert wav.dtype == np.float32
    np.testing.assert_allclose(wav, 0.25, atol=1e-4)


def test_load_audio_keeps_mono_untouched(tmp_path):
    wav, in_sr = XVectorRunner._load_audio(_wav(tmp_path, seconds=0.5, sr=8000))

    assert (wav.ndim, in_sr, wav.shape) == (1, 8000, (4000,))
    assert wav.dtype == np.float32


def test_extract_embedding_dispatches_toolkit(monkeypatch):
    seen = []
    for name in ("espnet", "speechbrain", "rawnet"):
        monkeypatch.setattr(
            XVectorRunner,
            f"_extract_{name}",
            staticmethod(lambda wav, sr, model, device, n=name: seen.append(n)),
        )

    wav = np.zeros(16, dtype=np.float32)
    for name in ("espnet", "speechbrain", "rawnet"):
        XVectorRunner._extract_embedding(wav, 16000, "MODEL", name, "cpu")
    assert seen == ["espnet", "speechbrain", "rawnet"]

    with pytest.raises(ValueError, match="Unknown toolkit: nope"):
        XVectorRunner._extract_embedding(wav, 16000, "MODEL", "nope", "cpu")


def test_extract_speechbrain_uses_encode_batch(fake_speechbrain):
    class _Model:
        def __init__(self):
            self.calls = 0

        def encode_batch(self, wav_tensor):
            self.calls += 1
            # speechbrain returns (batch, 1, emb); the runner unwraps [0].
            return torch.zeros(1, 1, 192)

    model = _Model()
    out = XVectorRunner._extract_speechbrain(
        np.zeros(16000, dtype=np.float32), 16000, model, "cpu"
    )

    assert model.calls == 1
    assert out.shape == (1, 192)


def test_extract_espnet_resamples_and_mixes():
    seen = {}

    def _model(wav_tensor):
        seen["shape"] = tuple(wav_tensor.shape)
        return torch.zeros(192)

    # 8 kHz stereo input must arrive as mono at the espnet default of 16 kHz.
    out = XVectorRunner._extract_espnet(
        np.zeros((2, 8000), dtype=np.float32), 8000, _model, "cpu"
    )

    assert len(seen["shape"]) == 1
    assert seen["shape"][0] == 16000
    assert out.shape == (192,)


def test_extract_rawnet_pads_short_audio():
    seen = {}

    def _model(audios):
        seen["shape"] = tuple(audios.shape)
        return torch.zeros(10, 256)

    # A 0.5 s clip is shorter than RawNet3's 3 s window and must be padded.
    out = XVectorRunner._extract_rawnet(
        np.zeros(8000, dtype=np.float32), 16000, _model, "cpu"
    )

    assert seen["shape"] == (10, 48000)
    assert out.shape == (256,)


# ---------------------------------------------------------------
# Optional-dependency and fallback branches
# ---------------------------------------------------------------
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_has_cuda_without_torch                 | A missing torch reports no   |
# |                                             | CUDA instead of raising.     |
# | test_build_model_rawnet                     | The RawNet3 branch builds,   |
# |                                             | loads weights and evals.     |
# | test_extract_rawnet_resamples               | Non-16 kHz input is          |
# |                                             | resampled before segmenting. |


def test_has_cuda_without_torch(monkeypatch):
    """_has_cuda must degrade to False when torch is not installed."""
    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)

    assert XVectorProvider._has_cuda() is False


def test_build_model_rawnet(monkeypatch, tmp_path):
    """RawNet3 is a vendored third-party module, so stub it to cover the branch."""
    built = {}

    class _RawNet3(torch.nn.Module):
        def __init__(self, block, **kwargs):
            super().__init__()
            built["block"] = block
            built["kwargs"] = kwargs

        def load_state_dict(self, state_dict, *a, **k):
            built["loaded"] = state_dict

        def to(self, device):
            built["device"] = device
            return self

        def eval(self):
            built["eval"] = True
            return self

    rawnet_mod = types.ModuleType("RawNet3")
    rawnet_mod.RawNet3 = _RawNet3
    block_mod = types.ModuleType("RawNetBasicBlock")
    block_mod.Bottle2neck = "BOTTLE2NECK"
    monkeypatch.setitem(sys.modules, "RawNet3", rawnet_mod)
    monkeypatch.setitem(sys.modules, "RawNetBasicBlock", block_mod)

    ckpt = tmp_path / "rawnet.pth"
    torch.save({"model": {"w": torch.zeros(1)}}, str(ckpt))

    model = XVectorProvider._build_model("rawnet", str(ckpt), "cpu")

    assert isinstance(model, _RawNet3)
    assert built["block"] == "BOTTLE2NECK"
    assert built["kwargs"]["nOut"] == 256
    assert built["device"] == "cpu"
    assert built["eval"] is True
    assert "w" in built["loaded"]


def test_extract_rawnet_resamples():
    seen = {}

    def _model(audios):
        seen["shape"] = tuple(audios.shape)
        return torch.zeros(10, 256)

    # 8 kHz in must be resampled to 16 kHz before the 3 s windows are cut.
    out = XVectorRunner._extract_rawnet(
        np.zeros(80000, dtype=np.float32), 8000, _model, "cpu"
    )

    assert seen["shape"] == (10, 48000)
    assert out.shape == (256,)
