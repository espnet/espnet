"""Regression checks for the standalone SpeechLM inference command."""

import json
from unittest.mock import MagicMock, patch

import pytest
import soundfile as sf
import torch

pytest.importorskip("liger_kernel.ops.fused_linear_cross_entropy")

from espnet2.speechlm.bin.inference import (  # noqa: E402
    get_parser,
    inference_worker,
    load_checkpoint,
    main,
)


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
        patch("torch.multiprocessing.set_start_method"),
        patch("torch.multiprocessing.Process") as process,
    ):
        process.return_value.exitcode = 1
        with pytest.raises(RuntimeError, match="workers failed"):
            main()
    assert (tmp_path / "audio_to_text_clean").is_dir()
    assert not (tmp_path / "audio_to_text_clean_" / "absolute").exists()


def test_worker_writes_complete_text_and_audio(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("job_type: speechlm\n")
    model = MagicMock()
    model.to.return_value.eval.return_value = model
    waveform = torch.linspace(-0.5, 0.5, 32).reshape(1, 1, -1)
    model.inference.return_value = (
        [
            ["assistant", "text", ["A complete caption."]],
            ["assistant", "audio", (waveform, torch.tensor([32]), 16000)],
        ],
        None,
    )
    job = MagicMock()
    job.build_model.return_value = model
    with (
        patch(
            "espnet2.speechlm.bin.inference._all_job_types",
            {"speechlm": lambda *a, **k: job},
        ),
        patch("espnet2.speechlm.bin.inference.load_checkpoint", return_value=model),
        patch(
            "espnet2.speechlm.bin.inference.to_device", side_effect=lambda x, *a, **k: x
        ),
        patch("espnet2.speechlm.bin.inference.DataIteratorFactory") as factory,
        patch("torch.cuda.set_device"),
        patch("torch.cuda.manual_seed"),
        patch("torch.cuda.manual_seed_all"),
        patch("espnet2.speechlm.bin.inference.setup_worker_logger"),
    ):
        factory.return_value.build_iter.return_value = [
            {"keys": [("text_to_audio", "smoke", "example")]}
        ]
        inference_worker(
            0, 1, config, config, tmp_path / "model.pt", "", "", tmp_path, 0
        )
    result = json.loads((tmp_path / "inference_rank0" / "results.json").read_text())
    messages = result["example"]
    assert messages[0][2] == "A complete caption."
    audio, rate = sf.read(messages[1][2])
    assert rate == 16000
    torch.testing.assert_close(
        torch.from_numpy(audio).float(), waveform.flatten(), rtol=0, atol=1 / 32768
    )
