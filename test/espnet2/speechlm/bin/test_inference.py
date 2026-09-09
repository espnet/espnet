"""Regression checks for the standalone SpeechLM inference command."""

from unittest.mock import patch

import pytest
import torch

from espnet2.speechlm.bin.inference import get_parser, load_checkpoint, main


def test_single_gpu_defaults():
    args = get_parser().parse_args(
        [
            "--train-config",
            "train.yaml",
            "--inference-config",
            "inference.yaml",
            "--model-checkpoint",
            "model.pt",
        ]
    )
    assert (args.rank, args.world_size, args.num_workers) == (1, 1, 1)


def test_native_checkpoint_loads_strictly(tmp_path):
    source = torch.nn.Linear(3, 4)
    target = torch.nn.Linear(3, 4)
    checkpoint = tmp_path / "model.pt"
    torch.save({"module": source.state_dict()}, checkpoint)
    load_checkpoint(target, checkpoint)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name])
    torch.save({"module": {"weight": source.weight}}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(target, checkpoint)


def test_failed_worker_fails_command(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference",
            "--train-config",
            "train.yaml",
            "--inference-config",
            "infer.yaml",
            "--model-checkpoint",
            "model.pt",
            "--test-unregistered-specifier",
            "audio_to_text:clean:/absolute/input.json",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.multiprocessing.Process") as process,
    ):
        process.return_value.exitcode = 1
        with pytest.raises(RuntimeError, match="workers failed"):
            main()
    assert (tmp_path / "audio_to_text_clean").is_dir()
    assert not (tmp_path / "audio_to_text_clean_" / "absolute").exists()
