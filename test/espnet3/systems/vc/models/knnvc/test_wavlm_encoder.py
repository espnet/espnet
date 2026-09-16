"""Tests for the WavLM encoder wrapper."""

import inspect
from pathlib import Path

import pytest
import torch

from espnet3.systems.vc.models.knnvc.wavlm_encoder import WAVLM_LARGE_URL, WavLMEncoder

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_paper_defaults_are_the_constructor_defaults | layer 6 and the kNN-VC   |
# |                                             | release URL are the defaults.|
# | test_default_checkpoint_is_the_knn_vc_release | The default URL is the     |
# |                                             | authors' WavLM-Large.        |
# | test_rejects_non_positive_layer             | layer < 1 raises (the index  |
# |                                             | is 1-based).                 |
# | test_rejects_checkpoint_without_cfg_and_model | A foreign checkpoint fails |
# |                                             | with a clear message.        |
# | test_rejects_layer_beyond_checkpoint_depth  | layer > encoder_layers       |
# |                                             | raises before building WavLM.|
# | test_encoder_module_path_is_stable          | The public import path of    |
# |                                             | WavLMEncoder does not move.  |
#
# Building a real WavLMEncoder needs the 1.2 GB WavLM-Large checkpoint, so the
# forward path is covered by the integration script and by
# `scripts/verify_knn_vc_port.py` (which compares features against the
# original kNN-VC implementation) rather than here.

# A minimal WavLM config: two transformer layers is enough for the guards
# below, which all fail before any module is constructed.
_TINY_CFG = {"encoder_layers": 2, "encoder_embed_dim": 8}


def test_default_checkpoint_is_the_knn_vc_release():
    assert WAVLM_LARGE_URL.startswith("https://github.com/bshall/knn-vc/releases/")
    assert WAVLM_LARGE_URL.endswith("WavLM-Large.pt")


@pytest.mark.parametrize("layer", [0, -1])
def test_rejects_non_positive_layer(tmp_path, layer):
    with pytest.raises(ValueError, match="1-based"):
        WavLMEncoder(checkpoint=tmp_path / "unused.pt", layer=layer)


def test_rejects_checkpoint_without_cfg_and_model(tmp_path):
    path = tmp_path / "foreign.pt"
    torch.save({"state_dict": {}}, path)

    with pytest.raises(ValueError, match="'cfg' and 'model'"):
        WavLMEncoder(checkpoint=path)


def test_rejects_layer_beyond_checkpoint_depth(tmp_path):
    path = tmp_path / "shallow.pt"
    torch.save({"cfg": _TINY_CFG, "model": {}}, path)

    with pytest.raises(ValueError, match="exceeds encoder_layers"):
        WavLMEncoder(checkpoint=path, layer=6)


def test_encoder_module_path_is_stable():
    """The recipe configs name this class by dotted path; keep it importable."""
    assert WavLMEncoder.__module__ == "espnet3.systems.vc.models.knnvc.wavlm_encoder"
    assert Path(WavLMEncoder.__module__.replace(".", "/") + ".py").name
    assert WavLMEncoder.sample_rate == 16000
    assert WavLMEncoder.hop_length == 320


def test_paper_defaults_are_the_constructor_defaults():
    """``layer=6`` is not merely a recipe value -- it is the class default.

    The released kNN-VC vocoders were trained on WavLM-Large layer 6; a
    different default would silently produce features they cannot vocode.
    """
    params = inspect.signature(WavLMEncoder.__init__).parameters
    assert params["layer"].default == 6
    assert params["checkpoint"].default == WAVLM_LARGE_URL
