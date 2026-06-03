from pathlib import Path

# NOTE: these fixture imports are necessary to have a config file in place
from test.espnet2.bin.test_s2t_inference import (  # noqa: F401
    s2t_config_file,
    token_list,
)

import pytest
import torch

from espnet2.bin.s2t_inference import Speech2Text


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="This test is specific to macOS with Metal Performance Shaders enabled",
)
def test_macos_mps_model(s2t_config_file, tmp_path: Path):  # noqa: F811
    # create a float64 model using the CPU since it supports it
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(80, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 5),
    ).to(dtype=torch.float64)
    torch.save(dummy_model.state_dict(), tmp_path / "s2t" / "model.pth")

    # load it with mps and make sure it gets loaded
    # as float32 since mps doesn't support float64
    s2t = Speech2Text.from_pretrained(
        s2t_model_file=str(tmp_path / "s2t" / "model.pth"),
        device="mps",
    )
    assert s2t.device == "mps"
    assert s2t.dtype == "float32"
    model_params = next(s2t.s2t_model.parameters())
    assert model_params.device.type == "mps"
    assert model_params.dtype == torch.float32
