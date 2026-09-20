"""Verify the VOiCES ASR system's statistics adapter."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("fail", [False, True])
def test_statistics_preserve_normalization_and_numel_shapes(
    tmp_path, monkeypatch, fail
):
    """Keep GlobalMVN and the text vocabulary dimension required by numel."""
    from egs3.voices.asr.src import system as system_module

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blank>\n<unk>\nA\nB\n<sos/eos>\n")
    config = OmegaConf.create(
        {
            "model": {
                "normalize": "global_mvn",
                "normalize_conf": {"stats_file": "stats"},
                "token_list": str(tokens),
            },
            "stats_dir": str(tmp_path),
            "dataset": {},
        }
    )
    system = object.__new__(system_module.VoicesSystem)
    system.training_config = config
    monkeypatch.setattr(system, "train_tokenizer", lambda: None)

    def collect(self):
        self.training_config.model.pop("normalize")
        self.training_config.model.pop("normalize_conf")
        if fail:
            raise RuntimeError("Interrupted statistics")

    monkeypatch.setattr(system_module.ASRSystem, "collect_stats", collect)
    samples = [{"speech": np.zeros(16000), "text": np.arange(7)}]
    monkeypatch.setattr(
        system_module,
        "instantiate",
        lambda _: SimpleNamespace(train=samples, valid=samples),
    )
    if fail:
        with pytest.raises(RuntimeError, match="Interrupted statistics"):
            system.collect_stats()
    else:
        system.collect_stats()
        assert (tmp_path / "train/speech_shape").read_text() == "0 16000\n"
        assert (tmp_path / "valid/text_shape").read_text() == "0 7,5\n"
    assert system.training_config is config
    assert config.model.normalize == "global_mvn"
    assert config.model.normalize_conf.stats_file == "stats"
