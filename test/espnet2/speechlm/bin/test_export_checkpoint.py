"""Verify that DCP export preserves weights without exporting optimizer state."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

from espnet2.speechlm.bin.export_checkpoint import export_checkpoint


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_export_weights_only(tmp_path, dtype):
    model = torch.nn.Linear(3, 4)
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    state = model.state_dict()
    directory = tmp_path / "dcp"
    dcp.save(
        {"model": state, "optimizer": {"unused": torch.ones(2)}},
        checkpoint_id=directory,
        no_dist=True,
    )
    output = tmp_path / "model.pt"
    export_checkpoint(directory, output, dtype)
    result = torch.load(output, weights_only=True)
    assert set(result) == {"module"}
    assert set(result["module"]) == set(state)
    for name, value in state.items():
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype)
        torch.testing.assert_close(value, result["module"][name])
    with pytest.raises(FileExistsError):
        export_checkpoint(directory, output)
