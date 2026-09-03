import pytest
import torch

from espnet2.beats.generate_beats_checkpoint import (
    average_checkpoints,
    handle_finetuned_checkpoint,
)
from espnet2.torch_utils.safe_torch_load import _ENV_VAR, UnsafeLoadRefusedError


class UnsafeConfig(dict):
    pass


def _save_unsafe_checkpoint(path):
    torch.save(
        {
            "encoder.linear.weight": torch.tensor([1.0, 2.0]),
            "cfg": UnsafeConfig({"encoder_layers": 1}),
        },
        path,
    )


def test_average_checkpoints_requires_explicit_opt_in(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "beats.ckpt"
    _save_unsafe_checkpoint(checkpoint_path)
    monkeypatch.delenv(_ENV_VAR, raising=False)

    with pytest.raises(UnsafeLoadRefusedError):
        average_checkpoints([str(checkpoint_path)])


def test_average_checkpoints_accepts_env_opt_in(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "beats.ckpt"
    _save_unsafe_checkpoint(checkpoint_path)
    monkeypatch.setenv(_ENV_VAR, "1")

    averaged = average_checkpoints([str(checkpoint_path)])

    assert torch.equal(averaged["linear.weight"], torch.tensor([1.0, 2.0]))


def test_handle_finetuned_checkpoint_requires_explicit_opt_in(tmp_path, monkeypatch):
    pretrained_path = tmp_path / "beats_pretrained.ckpt"
    _save_unsafe_checkpoint(pretrained_path)
    monkeypatch.delenv(_ENV_VAR, raising=False)
    config = {"encoder_conf": {"beats_ckpt_path": str(pretrained_path)}}

    with pytest.raises(UnsafeLoadRefusedError):
        handle_finetuned_checkpoint({}, config)
