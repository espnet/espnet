import sys
import types

import pytest
import torch
import yaml

from espnet2.tts.utils.parallel_wavegan_pretrained_vocoder import (
    ParallelWaveGANPretrainedVocoder,
)


def _stub_parallel_wavegan(monkeypatch):
    """Stand in for the optional ``parallel_wavegan`` dependency.

    The loader imports it inside ``__init__``, so that import has to succeed
    before the config read -- the thing under test -- is reached at all.
    """
    captured = {}

    def load_model(model_file, config):
        captured["config"] = config
        return torch.nn.Module()

    utils = types.ModuleType("parallel_wavegan.utils")
    utils.load_model = load_model
    package = types.ModuleType("parallel_wavegan")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "parallel_wavegan", package)
    monkeypatch.setitem(sys.modules, "parallel_wavegan.utils", utils)
    return captured


def test_config_with_python_tag_is_refused(tmp_path, monkeypatch):
    """A vocoder bundle must not execute code through its config.

    ``yaml.Loader`` honours tags such as ``!!python/object/apply``, so a
    downloaded vocoder could run arbitrary code at load time. ``safe_load``
    refuses them.
    """
    _stub_parallel_wavegan(monkeypatch)
    (tmp_path / "config.yml").write_text(
        "sampling_rate: 24000\n"
        "payload: !!python/object/apply:os.system ['echo pwned']\n"
    )

    with pytest.raises(yaml.YAMLError):
        ParallelWaveGANPretrainedVocoder(tmp_path / "checkpoint.pkl")


def test_plain_config_still_loads(tmp_path, monkeypatch):
    """The configs real vocoder bundles ship are unaffected by the change."""
    captured = _stub_parallel_wavegan(monkeypatch)
    (tmp_path / "config.yml").write_text("sampling_rate: 24000\nformat: hdf5\n")

    vocoder = ParallelWaveGANPretrainedVocoder(tmp_path / "checkpoint.pkl")

    assert vocoder.fs == 24000
    assert captured["config"] == {"sampling_rate": 24000, "format": "hdf5"}
