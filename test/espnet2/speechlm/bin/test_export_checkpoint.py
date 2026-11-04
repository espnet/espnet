"""Verify that DCP export preserves weights without exporting optimizer state."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

pytest.importorskip("liger_kernel.ops.fused_linear_cross_entropy")

from espnet2.speechlm.bin.export_checkpoint import export_checkpoint  # noqa: E402
from espnet2.speechlm.bin.inference import load_checkpoint  # noqa: E402


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.bfloat16, torch.float16])
def test_export_weights_only(tmp_path, dtype):
    model = torch.nn.Linear(3, 4)
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    state = model.state_dict()
    directory = tmp_path / "dcp"
    dcp.save(
        {"model": state, "optimizer": {"unused": torch.ones(2)}},
        storage_writer=dcp.FileSystemWriter(directory, single_file_per_rank=False),
        no_dist=True,
    )
    reader = dcp.FileSystemReader(directory)
    metadata = reader.read_metadata()
    optimizer_files = {
        item.relative_path
        for index, item in metadata.storage_data.items()
        if index.fqn.startswith("optimizer.")
    }
    for filename in optimizer_files:
        (directory / filename).unlink()
    output = tmp_path / "model.pt"
    export_checkpoint(directory, output, dtype)
    result = torch.load(output, weights_only=True)
    assert set(result) == {"module"}
    assert set(result["module"]) == set(state)
    for name, value in state.items():
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype)
        torch.testing.assert_close(value, result["module"][name])
    target = torch.nn.Linear(3, 4).to(dtype=dtype or torch.float32)
    target.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
    load_checkpoint(target, output)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, result["module"][name])
    with pytest.raises(FileExistsError):
        export_checkpoint(directory, output)


def test_export_rejects_checkpoint_without_model(tmp_path):
    directory = tmp_path / "dcp"
    dcp.save({"optimizer": {"unused": torch.ones(2)}}, checkpoint_id=directory)
    output = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="No model tensors"):
        export_checkpoint(directory, output)
    assert not output.exists()
