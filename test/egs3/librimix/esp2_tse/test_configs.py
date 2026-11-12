"""Resolve all TSE role defaults and validate inference/scoring bindings."""

import logging
from importlib import import_module
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.contract.inference import load_inference_contract
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
)

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "egs3/librimix/esp2_tse"


def test_all_role_defaults_and_targets(monkeypatch):
    monkeypatch.chdir(RECIPE)
    configs = {
        role: load_and_merge_config(
            RECIPE / "conf" / f"{role}.yaml",
            config_name=f"{role}.yaml",
            default_package="egs3.TEMPLATE.esp2_tse",
            resolve=False,
        )
        for role in ("training", "inference", "metrics", "publication", "demo")
    }
    apply_training_experiment_context(
        log=logging.getLogger(__name__),
        **{f"{k}_config": v for k, v in configs.items()},
    )
    resolve_loaded_configs(*configs.values())
    for cfg in configs.values():
        OmegaConf.to_container(cfg, resolve=True)

        def check(value):
            if isinstance(value, dict):
                if "_target_" in value:
                    mod, name = value["_target_"].rsplit(".", 1)
                    assert hasattr(import_module(mod), name)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        check(OmegaConf.to_container(cfg, resolve=True))
    infer = configs["inference"]
    load_inference_contract(infer.contract).validate_input_keys(infer.input_key)
    assert "speech_ref1" not in infer.input_key
    assert "ref" not in infer.output_artifacts
    assert "output_fn" not in infer
    assert configs["training"].model.model_conf.num_spk == 2
    for entry in configs["metrics"].metrics:
        assert entry.inputs.hyp == "inf"
        if not entry.metric._target_.endswith("DNSMOS"):
            assert entry.inputs.ref == "dataset:speech_ref1"
    assert configs["metrics"].dataset_artifacts.speech_ref1.type == "wav"


def test_contract_input_names_must_match():
    with pytest.raises(ValueError, match="input_key"):
        load_inference_contract(
            {
                "inputs": {"speech_mix": {"kind": "audio", "sample_rate": 16000}},
                "outputs": {"inf": {"kind": "audio", "sample_rate": 16000}},
            }
        ).validate_input_keys(["speech_ref1"])


def test_external_librimix_enrollment_root(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBRIMIX", str(tmp_path))
    cfg = load_and_merge_config(
        RECIPE / "conf/training.yaml",
        config_name="training.yaml",
        default_package="egs3.TEMPLATE.esp2_tse",
        resolve=False,
    )
    assert cfg.dataset.train[0].data_src_args.data_dir == str(tmp_path)
    assert Path(cfg.dataset.preprocessor.train_spk2enroll).parent.parent == tmp_path
