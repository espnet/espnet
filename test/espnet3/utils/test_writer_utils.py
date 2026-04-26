from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from espnet3.utils.writer_utils import infer_artifact_type, write_artifact


def custom_npy_writer(*, value, output_path: Path, scale: float = 1.0) -> Path:
    target = output_path.with_suffix(".custom.npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(value) * scale)
    return target


def invalid_writer_returns_int(*, value, output_path: Path) -> int:
    return 123


def invalid_writer_returns_missing_path(*, value, output_path: Path) -> Path:
    return output_path.with_suffix(".missing.npy")


def test_write_artifact_with_custom_writer(tmp_path: Path):
    path = write_artifact(
        np.array([1.0, 2.0], dtype=np.float32),
        tmp_path / "artifact",
        field_config={
            "writer": {
                "_target_": "test.espnet3.utils.test_writer_utils.custom_npy_writer",
                "scale": 2.0,
            }
        },
    )
    assert path.suffixes[-2:] == [".custom", ".npy"]
    assert np.allclose(np.load(path), np.array([2.0, 4.0], dtype=np.float32))


def test_write_artifact_writes_wav(tmp_path: Path):
    audio = np.array([0.0, 0.1, -0.1], dtype=np.float32)
    path = write_artifact(
        audio,
        tmp_path / "audio",
        field_config={"type": "wav", "sample_rate": 16000},
    )
    data, rate = sf.read(path, dtype="float32")
    assert path.suffix == ".wav"
    assert rate == 16000
    assert len(data) == len(audio)


def test_write_artifact_writes_tensor_as_npy(tmp_path: Path):
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    path = write_artifact(tensor, tmp_path / "tensor")
    assert path.suffix == ".npy"
    assert np.array_equal(np.load(path), tensor.numpy())


def test_write_artifact_rejects_cuda_tensor(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in test env.")
    tensor = torch.tensor([1.0], device="cuda")
    with pytest.raises(ValueError, match="Move the tensor to CPU"):
        write_artifact(tensor, tmp_path / "tensor")


def test_infer_artifact_type_defaults():
    assert infer_artifact_type(np.array([1])) == "npy"
    assert infer_artifact_type(torch.tensor([1])) == "npy"
    assert infer_artifact_type({"a": [1, 2]}) == "json"
    assert infer_artifact_type(object()) == "pickle"


def test_write_artifact_rejects_wav_without_sample_rate(tmp_path: Path):
    with pytest.raises(ValueError, match="sample_rate"):
        write_artifact(
            np.array([0.0], dtype=np.float32), tmp_path / "audio", {"type": "wav"}
        )


def test_write_artifact_writes_pickle(tmp_path: Path):
    path = write_artifact(object(), tmp_path / "artifact")
    assert path.suffix == ".pkl"
    assert path.exists()


def test_write_artifact_with_dictconfig_field_config(tmp_path: Path):
    field_config = OmegaConf.create(
        {
            "writer": {
                "_target_": "test.espnet3.utils.test_writer_utils.custom_npy_writer",
                "scale": 3.0,
            }
        }
    )
    path = write_artifact(
        np.array([1.0, 2.0], dtype=np.float32),
        tmp_path / "artifact",
        field_config=field_config,
    )
    assert path.suffixes[-2:] == [".custom", ".npy"]
    assert np.allclose(np.load(path), np.array([3.0, 6.0], dtype=np.float32))


def test_write_artifact_rejects_unsupported_type(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported artifact type"):
        write_artifact(np.array([1.0]), tmp_path / "artifact", {"type": "yaml"})


def test_write_artifact_rejects_custom_writer_non_path_return(tmp_path: Path):
    with pytest.raises(TypeError, match="path-like"):
        write_artifact(
            np.array([1.0]),
            tmp_path / "artifact",
            {
                "writer": {
                    "_target_": (
                        "test.espnet3.utils.test_writer_utils."
                        "invalid_writer_returns_int"
                    ),
                }
            },
        )


def test_write_artifact_rejects_custom_writer_missing_output(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        write_artifact(
            np.array([1.0]),
            tmp_path / "artifact",
            {
                "writer": {
                    "_target_": (
                        "test.espnet3.utils.test_writer_utils."
                        "invalid_writer_returns_missing_path"
                    ),
                }
            },
        )
