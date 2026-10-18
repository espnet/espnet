"""Check original checkpoint renaming independently of model downloads."""

import importlib.util
from pathlib import Path

import pytest
import torch

PATH = (
    Path(__file__).resolve().parents[3] / "egs2/geolid/lid1/local/convert_checkpoint.py"
)
spec = importlib.util.spec_from_file_location("convert_checkpoint", PATH)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_conversion_strict_load(tiny_geolid):
    _, model, _ = tiny_geolid
    source = {}
    prefix = converter.PREFIX
    for key, value in model.state_dict().items():
        old = key
        for i in (0, 1):
            layer = prefix + f"layers.{i}."
            if key.startswith(layer):
                suffix = key.removeprefix(layer)
                module, parameter = suffix.split(".", 1)
                if module == "layer":
                    old = layer + parameter
                elif module == "conditioning_layer":
                    old = prefix + "lang2vec_conditioning_projs." + parameter
                else:
                    group = {
                        "encoder": "ecapa_encoder",
                        "prediction": "lang2vec_head",
                    }.get(module, module)
                    old = (
                        prefix
                        + f"{group}.{i}."
                        + ("0." if module == "prediction" else "")
                        + parameter
                    )
        source[old] = value
    source[prefix + "aamsoftmax_loss.weight"] = source["loss.weight"]
    converted = converter.convert_state_dict(source, [0, 1])
    model.load_state_dict(converted, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, converted[key])


def test_conflicting_alias_is_rejected():
    source = {
        "loss.weight": torch.ones(2),
        converter.PREFIX + "aamsoftmax_loss.weight": torch.zeros(2),
    }
    with pytest.raises(ValueError, match="Conflicting checkpoint aliases"):
        converter.convert_state_dict(source, [32])


def test_early_independent_projection_names():
    weight = torch.randn(4, 3)
    converted = converter.convert_state_dict(
        {converter.PREFIX + "conditioning_projs.32.weight": weight}, [32]
    )
    assert torch.equal(
        converted[converter.PREFIX + "layers.32.conditioning_layer.weight"], weight
    )
