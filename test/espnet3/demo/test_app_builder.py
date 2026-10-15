"""The shipped ASR app runs without inference declarations or generated sessions."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from egs3.TEMPLATE.esp2_asr.src import app as recipe_app
from espnet3.publication.demo.packing import pack_demo
from espnet3.utils.config_utils import load_and_merge_config


def _archive(tmp_path, target="espnet2.bin.asr_inference.Speech2Text"):
    pack = tmp_path / "model_pack"
    (pack / "conf").mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "recipe_dir": ".",
                "model": {
                    "_target_": target,
                    "_convert_": "all",
                    "_recursive_": False,
                    "asr_train_config": "${recipe_dir}/config.yaml",
                    "asr_model_file": "${recipe_dir}/weights.pth",
                },
            }
        ),
        pack / "conf" / "inference.yaml",
    )
    return pack


@pytest.mark.parametrize("transducer", [False, True])
def test_demo_builds_native_backend_from_archive(tmp_path, monkeypatch, transducer):
    target = (
        "espnet2.bin.asr_transducer_inference.Speech2Text"
        if transducer
        else "espnet2.bin.asr_inference.Speech2Text"
    )
    pack = _archive(tmp_path, target)
    import importlib

    backend = importlib.import_module(target.rsplit(".", 1)[0])
    monkeypatch.setattr(backend, "Speech2Text", lambda **kwargs: kwargs)
    result = recipe_app.build_model(
        tmp_path, OmegaConf.create({"dir_or_tag": "model_pack", "device": "cpu"})
    )
    expected = {
        "asr_train_config": str(pack / "config.yaml"),
        "asr_model_file": str(pack / "weights.pth"),
        "device": "cpu",
    }
    if transducer:
        expected["return_decoded_hyp"] = True
    assert result == expected


def test_demo_resolves_remote_archive(tmp_path, monkeypatch):
    pack = _archive(tmp_path)
    calls = []
    monkeypatch.setattr(
        recipe_app,
        "snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(pack),
    )
    from espnet2.bin import asr_inference

    monkeypatch.setattr(asr_inference, "Speech2Text", lambda **kwargs: kwargs)
    result = recipe_app.build_model(
        tmp_path, OmegaConf.create({"dir_or_tag": "owner/model", "device": "cpu"})
    )
    assert calls == [{"repo_id": "owner/model"}]
    assert result["asr_model_file"] == str(pack / "weights.pth")


@pytest.mark.parametrize("nested", [False, True])
def test_demo_resamples_audio_and_formats_native_result(tmp_path, nested):
    path = tmp_path / "audio.wav"
    sf.write(path, np.ones((800, 2), dtype=np.float32) * 0.25, 8000)
    seen = []

    def model(speech):
        seen.append(speech)
        best = ("decoded text", [], [], None)
        return [[best]] if nested else [best]

    predict = recipe_app.create_predict(model, 16000)
    assert predict(str(path)) == "decoded text"
    assert seen[0].shape == (1600,)
    assert seen[0].dtype == np.float32
    with pytest.raises(recipe_app.gr.Error, match="Record or upload"):
        predict(None)


@pytest.mark.parametrize("recipe", [None, "librispeech_100", "mini_an4"])
def test_packed_default_app_builds_and_decodes(tmp_path, monkeypatch, recipe):
    pack = _archive(tmp_path)
    user_config = tmp_path / "demo.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "exp_tag": "test",
                "upload_demo": {"hf_repo": "owner/demo"},
                "model": {"dir_or_tag": str(pack), "device": "cpu"},
                "ui": {"description": None, "title": "Test ASR"},
                "pack": {"out_dir": str(tmp_path / "demo")},
            }
        ),
        user_config,
    )
    config = load_and_merge_config(
        user_config,
        "demo.yaml",
        default_package="egs3.TEMPLATE.esp2_asr",
        resolve=False,
    )
    if recipe is not None:
        recipe_config = (
            Path(__file__).resolve().parents[3]
            / "egs3"
            / recipe
            / "esp2_asr"
            / "conf"
            / "demo.yaml"
        )
        config = OmegaConf.merge(
            load_and_merge_config(
                recipe_config,
                "demo.yaml",
                default_package="egs3.TEMPLATE.esp2_asr",
                resolve=False,
            ),
            OmegaConf.load(user_config),
        )
    config.ui.inputs = [{"key": "speech", "type": "audio", "label": "Recording"}]
    config.ui.outputs = [{"key": "text", "type": "text", "label": "Decoded text"}]
    out = pack_demo(SimpleNamespace(demo_config=config, exp_dir=None))
    spec = importlib.util.spec_from_file_location("packed_asr_app", out / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from espnet2.bin import asr_inference

    monkeypatch.setattr(
        asr_inference,
        "Speech2Text",
        lambda **kwargs: lambda speech: [("hello", [], [], None)],
    )
    ui = module.build_demo(out)
    assert ui.title == "Test ASR"
    assert any(
        getattr(block, "label", None) == "Recording" for block in ui.blocks.values()
    )
    assert any(
        getattr(block, "label", None) == "Decoded text" for block in ui.blocks.values()
    )
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(1600), 16000)
    predict = next(iter(ui.fns.values())).fn
    assert predict(str(audio)) == "hello"
