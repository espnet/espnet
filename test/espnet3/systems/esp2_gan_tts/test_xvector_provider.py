"""Tests for the x-vector provider used by compute_xvectors."""

import builtins
import sys
import types
from test.espnet3.systems.esp2_gan_tts._xvector_helpers import (
    make_config,
    make_provider,
    write_manifest,
)

import pytest
import torch
from omegaconf import OmegaConf

from espnet3.systems.esp2_gan_tts.xvector_provider import (
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_TOOLKIT,
    XVectorProvider,
)

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
# | test_load_manifest_rejects_short_rows       | A row with < 4 columns names |
# |                                             | the file and line.           |
# |
# provider environments
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_env_local_returns_worker_env     | The driver env carries the   |
# |                          | model, manifest, and an existing output_dir.     |
# | test_build_env_defaults_to_espnet_toolkit   | No toolkit/model in config   |
# |                          | means espnet's tts.sh defaults, not speechbrain. |
# | test_build_env_null_device_picks_automatically | device: null behaves like |
# |                                             | a missing key.               |
# | test_build_env_local_requires_config        | Missing xvector/manifest/    |
# |                                             | output_dir all raise.        |
# | test_build_env_local_rejects_empty_manifest | An empty manifest raises.    |
# | test_worker_setup_fn_builds_same_env        | The worker setup function    |
# |                                             | produces the same env.       |
# | test_worker_setup_fn_requires_config        | Same validation in a worker. |
# | test_has_cuda_reports_torch_availability    | _has_cuda follows torch.     |
# | test_has_cuda_without_torch                 | A missing torch reports no   |
# |                                             | CUDA instead of raising.     |
#
# model construction
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_model_speechbrain                | Delegates to speechbrain's   |
# |                                             | EncoderClassifier.           |
# | test_build_model_speechbrain_missing_dep    | Without speechbrain the      |
# |                          | ImportError names the package and the fallback.  |
# | test_build_model_espnet_model_tag_vs_file   | A .pth path becomes          |
# |                                             | model_file, a tag model_tag. |
# | test_build_model_rawnet                     | The RawNet3 branch builds,   |
# |                          | loads weights (weights_only) and evals.          |
# | test_build_model_rawnet_missing_dep         | Without RawNet3 the          |
# |                                             | ImportError says so.         |
# | test_build_model_rejects_unknown_toolkit    | Unknown toolkit raises.      |


# ---------------------------------------------------------------
# manifest loading
# ---------------------------------------------------------------


def test_load_manifest_groups_by_speaker(manifest):
    utterances, speaker_to_utterances = XVectorProvider._load_manifest(manifest)

    assert [utt_id for utt_id, _ in utterances] == ["u1", "u2", "u3"]
    assert speaker_to_utterances == {"0": ["u1", "u2"], "1": ["u3"]}


def test_load_manifest_skips_blank_lines(tmp_path):
    path = write_manifest(tmp_path, ["u1\t/a.wav\thello\t0\n", "\n", "\n"])

    utterances, _ = XVectorProvider._load_manifest(path)

    assert utterances == [("u1", "/a.wav")]


def test_load_manifest_rejects_short_rows(tmp_path):
    """A malformed row must not surface as a bare IndexError in a worker."""
    path = write_manifest(tmp_path, ["u1\t/a.wav\thello\t0\n", "u2\t/b.wav\n"])

    with pytest.raises(RuntimeError, match="Malformed manifest line") as excinfo:
        XVectorProvider._load_manifest(path)

    assert str(path) in str(excinfo.value)
    assert "u2" in str(excinfo.value)


# ---------------------------------------------------------------
# provider environments
# ---------------------------------------------------------------


def test_build_env_local_returns_worker_env(manifest, tmp_path, stub_model):
    env = make_provider(manifest, tmp_path).build_env_local()

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


def test_build_env_defaults_to_espnet_toolkit(manifest, tmp_path, monkeypatch):
    """The defaults are espnet2's tts.sh defaults, which need no extra package."""
    seen = []
    monkeypatch.setattr(
        XVectorProvider,
        "_build_model",
        staticmethod(lambda *args: seen.append(args) or "MODEL"),
    )
    provider = XVectorProvider(
        OmegaConf.create({"xvector": {"device": "cpu"}}),
        params={"manifest_path": str(manifest), "output_dir": str(tmp_path / "x")},
    )

    env = provider.build_env_local()

    assert DEFAULT_TOOLKIT == "espnet"
    assert DEFAULT_PRETRAINED_MODEL == "espnet/voxcelebs12_rawnet3"
    assert seen == [("espnet", "espnet/voxcelebs12_rawnet3", "cpu")]
    assert env["toolkit"] == "espnet"


def test_build_env_null_device_picks_automatically(manifest, tmp_path, monkeypatch):
    """The template ships ``device: null``; it must resolve like a missing key."""
    monkeypatch.setattr(XVectorProvider, "_has_cuda", staticmethod(lambda: False))
    monkeypatch.setattr(
        XVectorProvider, "_build_model", staticmethod(lambda *a, **k: "MODEL")
    )
    provider = XVectorProvider(
        OmegaConf.create({"xvector": {"toolkit": "espnet", "device": None}}),
        params={"manifest_path": str(manifest), "output_dir": str(tmp_path / "x")},
    )

    assert provider.build_env_local()["device"] == "cpu"


def test_build_env_local_requires_config(manifest, tmp_path, stub_model):
    no_xvector = XVectorProvider(OmegaConf.create({}), params={})
    with pytest.raises(RuntimeError, match="xvector configuration not found"):
        no_xvector.build_env_local()

    no_manifest = XVectorProvider(make_config(), params={})
    with pytest.raises(RuntimeError, match="provide manifest_path"):
        no_manifest.build_env_local()

    no_output = XVectorProvider(make_config(), params={"manifest_path": str(manifest)})
    with pytest.raises(RuntimeError, match="output_dir must be provided"):
        no_output.build_env_local()


def test_build_env_local_rejects_empty_manifest(tmp_path, stub_model):
    empty = write_manifest(tmp_path, [])
    provider = make_provider(empty, tmp_path)

    with pytest.raises(RuntimeError, match="No utterances found"):
        provider.build_env_local()


def test_worker_setup_fn_builds_same_env(manifest, tmp_path, stub_model):
    provider = make_provider(manifest, tmp_path)

    setup = provider.build_worker_setup_fn()
    env = setup()

    assert sorted(env) == sorted(provider.build_env_local())
    assert env["model"] == "MODEL"
    assert len(env["utterances"]) == 3


def test_worker_setup_fn_requires_config(manifest, tmp_path, stub_model):
    """The worker shares the driver's validation through ``_build_env``."""
    with pytest.raises(RuntimeError, match="xvector configuration not found"):
        XVectorProvider(OmegaConf.create({}), params={}).build_worker_setup_fn()()

    with pytest.raises(RuntimeError, match="provide manifest_path"):
        XVectorProvider(make_config(), params={}).build_worker_setup_fn()()

    with pytest.raises(RuntimeError, match="output_dir must be provided"):
        XVectorProvider(
            make_config(), params={"manifest_path": str(manifest)}
        ).build_worker_setup_fn()()

    empty = write_manifest(tmp_path, [])
    with pytest.raises(RuntimeError, match="No utterances found"):
        XVectorProvider(
            make_config(),
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


def test_has_cuda_without_torch(monkeypatch):
    """_has_cuda must degrade to False when torch is not installed."""
    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)

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


def _hide_modules(monkeypatch, *names):
    """Make ``import <name>`` fail for each of ``names`` (and their submodules)."""
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name in names or any(name.startswith(n + ".") for n in names):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name in names or any(name.startswith(n + ".") for n in names):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", _blocked)


def test_build_model_speechbrain_missing_dep(monkeypatch):
    _hide_modules(monkeypatch, "speechbrain")

    with pytest.raises(ImportError, match="pip install speechbrain"):
        XVectorProvider._build_model("speechbrain", "speechbrain/x", "cpu")


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

    load_kwargs = {}
    real_load = torch.load

    def _spy_load(*args, **kwargs):
        load_kwargs.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_load)

    ckpt = tmp_path / "rawnet.pth"
    torch.save({"model": {"w": torch.zeros(1)}}, str(ckpt))

    model = XVectorProvider._build_model("rawnet", str(ckpt), "cpu")

    assert isinstance(model, _RawNet3)
    assert built["block"] == "BOTTLE2NECK"
    assert built["kwargs"]["nOut"] == 256
    assert built["device"] == "cpu"
    assert built["eval"] is True
    assert "w" in built["loaded"]
    # The checkpoint is untrusted input: only tensors may be unpickled.
    assert load_kwargs["weights_only"] is True


def test_build_model_rawnet_missing_dep(monkeypatch):
    _hide_modules(monkeypatch, "RawNet3", "RawNetBasicBlock")

    with pytest.raises(ImportError, match="RawNet3 modules are not importable"):
        XVectorProvider._build_model("rawnet", "/models/rawnet.pth", "cpu")


def test_build_model_rejects_unknown_toolkit():
    with pytest.raises(ValueError, match="Unknown toolkit: nope"):
        XVectorProvider._build_model("nope", "model", "cpu")
