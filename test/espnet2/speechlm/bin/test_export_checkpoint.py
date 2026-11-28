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


def _adapter_state():
    return {
        "proj.weight": torch.tensor([[1.003]]),
        "proj.lora_A": torch.tensor([[0.25]]),
        "proj.lora_B": torch.tensor([[0.01]]),
        "proj.lora_scale": torch.tensor(1.5),
    }


def test_export_casts_after_merging_lora(tmp_path):
    """Keep an update that would disappear if the base were rounded first."""
    directory = tmp_path / "dcp"
    dcp.save({"model": _adapter_state()}, checkpoint_id=directory)
    output = tmp_path / "weights" / "model.pt"
    export_checkpoint(directory, output, torch.bfloat16)
    state = torch.load(output, weights_only=True)["module"]
    assert set(state) == {"proj.weight"}
    expected = torch.tensor([[1.003 + 1.5 * 0.25 * 0.01]], dtype=torch.bfloat16)
    torch.testing.assert_close(state["proj.weight"], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad_tensor, error",
    [(None, "Incomplete LoRA"), (torch.zeros(1, 2), "Invalid LoRA")],
)
def test_export_rejects_invalid_lora_state(tmp_path, bad_tensor, error):
    """Reject incomplete adapters and mismatched matrix dimensions."""
    state = _adapter_state()
    del state["proj.lora_B"]
    if bad_tensor is not None:
        state["proj.lora_B"] = bad_tensor
    directory = tmp_path / "dcp"
    dcp.save({"model": state}, checkpoint_id=directory)
    output = tmp_path / "model.pt"
    with pytest.raises(ValueError, match=error):
        export_checkpoint(directory, output)
    assert not output.exists()


def test_failed_save_leaves_no_partial_output(tmp_path, monkeypatch):
    """Leave the destination available for a retry after a write failure."""
    directory = tmp_path / "dcp"
    dcp.save({"model": _adapter_state()}, checkpoint_id=directory)
    output = tmp_path / "result" / "model.pt"

    def fail_save(weights, path):
        path.write_bytes(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        export_checkpoint(directory, output)
    assert not list(output.parent.iterdir())


def test_export_preserves_output_published_during_serialization(tmp_path, monkeypatch):
    directory = tmp_path / "dcp"
    dcp.save({"model": _adapter_state()}, checkpoint_id=directory)
    output = tmp_path / "model.pt"
    save = torch.save

    def publish_competing_output(weights, path):
        save(weights, path)
        output.write_bytes(b"already published")

    monkeypatch.setattr(torch, "save", publish_competing_output)
    with pytest.raises(FileExistsError, match="already exists"):
        export_checkpoint(directory, output)
    assert output.read_bytes() == b"already published"
