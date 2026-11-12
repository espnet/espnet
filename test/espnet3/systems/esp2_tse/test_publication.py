"""A shared two-audio contract for recipe inference, packed calls and demos."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from espnet3.publication import InferenceModel
from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS
from espnet3.publication.demo.session import DemoSession
from espnet3.systems.base.inference import infer
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.utils.config_utils import load_and_merge_config


def test_packed_demo_and_recipe_wavs(tmp_path, monkeypatch):
    config = load_and_merge_config(
        Path(__file__).resolve().parents[4]
        / "egs3/TEMPLATE/esp2_tse/conf/inference.yaml",
        config_name="inference.yaml",
        default_package="egs3.TEMPLATE.esp2_tse",
        resolve=False,
    )
    wave = np.random.default_rng(0).normal(0, 0.1, 128).astype(np.float32)
    samples = [
        {"utt_id": "target1", "speech_mix": (16000, wave), "enroll_ref1": (16000, wave)}
    ]

    def backend(speech_mix, enroll_ref1, fs):
        assert fs == 16000
        return [speech_mix / 2]

    monkeypatch.setattr(
        InferenceProvider, "build_model", staticmethod(lambda cfg: backend)
    )
    monkeypatch.setattr(
        InferenceProvider, "build_dataset", staticmethod(lambda cfg: samples)
    )
    import espnet3.parallel.parallel as parallel_module

    monkeypatch.setattr(parallel_module, "parallel_config", None)
    config.exp_tag = "test"
    config.exp_dir = str(tmp_path / "exp")
    config.inference_dir = str(tmp_path / "infer")
    config.dataset.test = [{"name": "test"}]
    config.parallel.options = {"processes": False, "dashboard_address": None}
    pack = tmp_path / "pack"
    (pack / "conf").mkdir(parents=True)
    OmegaConf.save(config, pack / "conf/inference.yaml", resolve=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "schema_version": 1,
                "yaml_files": {"inference_config": "conf/inference.yaml"},
            }
        ),
        pack / "meta.yaml",
    )
    monkeypatch.chdir(tmp_path)  # Outside the recipe tree.
    model = InferenceModel.from_packed(pack)
    np.testing.assert_allclose(model(samples[0])["inf"].array, wave / 2)
    session = DemoSession(
        tmp_path,
        OmegaConf.create(
            {"model": {"call_args": {}}, "ui": {"title": None, "description": None}}
        ),
        model,
        DEFAULT_UI_ASSETS.clone(),
    )
    callback = session.create_inference_fn(session.input_specs, session.output_specs)
    rate, output = callback((16000, wave), (16000, wave))
    assert rate == 16000
    np.testing.assert_allclose(output, wave / 2)
    with pytest.raises((ValueError, TypeError)):
        callback((16000, np.full(128, np.nan)), (16000, wave))
    infer(config)
    uid, path = (tmp_path / "infer/test/inf.scp").read_text().strip().split(maxsplit=1)
    saved, rate = sf.read(path)
    assert uid == "target1" and rate == 16000
    np.testing.assert_allclose(saved, wave / 2, atol=1e-4)
