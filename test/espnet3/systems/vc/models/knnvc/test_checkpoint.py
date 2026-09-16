"""Tests for kNN-VC generator checkpoint conversion and loading."""

import re
from test.espnet3.systems.vc.models.knnvc.tiny import TINY_GENERATOR

import pytest
import torch

from espnet3.systems.vc.models.knnvc.checkpoint import (
    convert_knn_vc_generator_state_dict,
    is_knn_vc_generator_state_dict,
    load_generator_state_dict,
    load_state_dict_from_path_or_url,
)
from espnet3.systems.vc.models.knnvc.vocoder import KNNVCGenerator

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_convert_official_layout_round_trip     | Official key layout converts |
# |                          | to a strict-loadable KNNVCGenerator state dict. |
# | test_convert_rejects_unknown_keys           | Foreign keys raise KeyError. |
# | test_load_official_release_file             | {"generator": official} file |
# | test_load_lightning_checkpoint              | {"state_dict": generator.* + |
# |                                             | discriminator.*} file        |
# | test_load_prefixed_and_plain_state_dicts    | generator.* / plain layouts  |
# | test_load_rejects_unrelated_file            | No generator params raises.  |
# | test_load_state_dict_reads_local_file       | A torch.save'd object round- |
# |                                             | trips from a local path.     |
# | test_load_state_dict_missing_file           | A missing path raises        |
# |                                             | FileNotFoundError.           |

# Inverse of the rules in checkpoint.py, used to synthesize official files.
_TO_OFFICIAL = (
    (re.compile(r"^input_projection\.(.+)$"), r"lin_pre.\1"),
    (re.compile(r"^hifigan\.input_conv\.(.+)$"), r"conv_pre.\1"),
    (re.compile(r"^hifigan\.upsamples\.(\d+)\.1\.(.+)$"), r"ups.\1.\2"),
    (
        re.compile(r"^hifigan\.blocks\.(\d+)\.(convs[12])\.(\d+)\.1\.(.+)$"),
        r"resblocks.\1.\2.\3.\4",
    ),
    (re.compile(r"^hifigan\.output_conv\.1\.(.+)$"), r"conv_post.\1"),
)


def _official_state_dict(generator):
    official = {}
    for key, value in generator.state_dict().items():
        for pattern, replacement in _TO_OFFICIAL:
            if pattern.match(key):
                official[pattern.sub(replacement, key)] = value.clone()
                break
        else:
            raise AssertionError(f"unmapped key {key}")
    return official


@pytest.fixture
def generator():
    torch.manual_seed(0)
    return KNNVCGenerator(**TINY_GENERATOR)


def test_convert_official_layout_round_trip(generator):
    official = _official_state_dict(generator)
    assert is_knn_vc_generator_state_dict(official)
    assert any(k.startswith("lin_pre.") for k in official)
    assert any(k.startswith("conv_pre.weight_g") for k in official)

    converted = convert_knn_vc_generator_state_dict(official)
    assert converted.keys() == generator.state_dict().keys()

    torch.manual_seed(1)
    other = KNNVCGenerator(**TINY_GENERATOR)
    other.load_state_dict(converted, strict=True)
    feats = torch.randn(1, 7, TINY_GENERATOR["in_channels"])
    torch.testing.assert_close(other(feats), generator(feats))


def test_convert_rejects_unknown_keys():
    with pytest.raises(KeyError, match="Unexpected key"):
        convert_knn_vc_generator_state_dict({"conv_pre.bias": 0, "foo.bar": 1})


def test_load_official_release_file(tmp_path, generator):
    path = tmp_path / "prematch_g_tiny.pt"
    torch.save({"generator": _official_state_dict(generator)}, path)
    loaded = load_generator_state_dict(path)
    assert loaded.keys() == generator.state_dict().keys()


def test_load_lightning_checkpoint(tmp_path, generator):
    state_dict = {f"generator.{k}": v for k, v in generator.state_dict().items()}
    state_dict["discriminator.msd.weight"] = torch.zeros(1)
    path = tmp_path / "epoch0.ckpt"
    torch.save({"state_dict": state_dict, "epoch": 0}, path)
    loaded = load_generator_state_dict(path)
    assert loaded.keys() == generator.state_dict().keys()


def test_load_prefixed_and_plain_state_dicts(tmp_path, generator):
    plain = generator.state_dict()
    prefixed = {f"generator.{k}": v for k, v in plain.items()}
    torch.save(prefixed, tmp_path / "ave.pth")
    torch.save(plain, tmp_path / "plain.pth")
    assert load_generator_state_dict(tmp_path / "ave.pth").keys() == plain.keys()
    assert load_generator_state_dict(tmp_path / "plain.pth").keys() == plain.keys()


def test_load_rejects_unrelated_file(tmp_path):
    torch.save({"state_dict": {"encoder.weight": torch.zeros(1)}}, tmp_path / "x.ckpt")
    with pytest.raises(ValueError, match="No HiFi-GAN generator parameters"):
        load_generator_state_dict(tmp_path / "x.ckpt")
    torch.save([1, 2, 3], tmp_path / "list.pt")
    with pytest.raises(ValueError, match="does not contain a dict"):
        load_generator_state_dict(tmp_path / "list.pt")


def test_load_state_dict_reads_local_file(tmp_path):
    """Any ``torch.save``-d object round-trips; the loader is format-agnostic."""
    path = tmp_path / "ckpt.pt"
    payload = {"cfg": {"a": 1}, "model": {"w": torch.zeros(2)}}
    torch.save(payload, path)

    state = load_state_dict_from_path_or_url(path)

    assert state["cfg"] == payload["cfg"]
    assert torch.equal(state["model"]["w"], torch.zeros(2))


def test_load_state_dict_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_state_dict_from_path_or_url(tmp_path / "absent.pt")
