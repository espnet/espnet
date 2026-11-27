import subprocess
import sys
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import stage_names
from espnet3.systems.speechlm.system import SpeechLMSystem


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("train one.json", "train-two.json"):
        (tmp_path / name).touch()
    for name in ("a", "b", "valid"):
        (tmp_path / f"stats_dialogue_{name}.jsonl").write_text('{"utt": 12}\n')
    speechlm_config = tmp_path / "model config.yaml"
    speechlm_config.write_text("job_type: speechlm\n")
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path),
            "speechlm_config": str(speechlm_config),
            "num_device": 2,
            "stats_num_workers": 3,
            "train_unregistered_specifier": [
                "dialogue:a:train one.json",
                "dialogue:b:train-two.json:0.5",
            ],
            "valid_registered_specifier": ["dialogue:valid"],
            "wandb_mode": "disabled",
            "save_loader_state": True,
        }
    )


def test_stage_order():
    assert stage_names(SpeechLMSystem) == ["collect_stats", "train", "infer"]


def test_collect_stats(config):
    with patch("subprocess.run", autospec=True) as run:
        SpeechLMSystem(training_config=config).collect_stats()
    run.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "espnet2.speechlm.bin.prepare_length_stats",
            "--output-dir",
            config.stats_dir,
            "--num-workers",
            "3",
            "--train-config",
            config.speechlm_config,
            "--train-unregistered-specifier",
            "dialogue:a:train one.json",
            "--train-unregistered-specifier",
            "dialogue:b:train-two.json:0.5",
            "--valid-registered-specifier",
            "dialogue:valid",
        ],
        check=True,
    )


@pytest.mark.parametrize("num_device,save_state", [(1, False), (2, True)])
def test_train(config, tmp_path, monkeypatch, num_device, save_state):
    config.num_device = num_device
    config.save_loader_state = save_state
    if save_state:
        weights = tmp_path / "base.pt"
        weights.touch()
        config.resume_path = str(weights)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    with patch("subprocess.run", autospec=True) as run:
        SpeechLMSystem(training_config=config).train()
    run.assert_called_once()
    args = run.call_args.args[0]
    assert args[:8] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(num_device),
        "--module",
        "espnet2.speechlm.bin.train",
    ]
    assert args[8:] == [
        "--output-dir",
        config.exp_dir,
        "--stats-dir",
        config.stats_dir,
        "--wandb-mode",
        "disabled",
        "--train-config",
        config.speechlm_config,
        "--train-unregistered-specifier",
        "dialogue:a:train one.json",
        "--train-unregistered-specifier",
        "dialogue:b:train-two.json:0.5",
        "--valid-registered-specifier",
        "dialogue:valid",
    ] + (["--resume-path", str(weights), "--save-loader-state"] if save_state else [])
    assert run.call_args.kwargs["env"]["PYTHONHASHSEED"] == "0"
    assert run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3,5"
    assert run.call_args.kwargs["check"] is True


@pytest.mark.parametrize(
    "stage,key,value,message",
    [
        ("train", "valid_registered_specifier", [], "valid data specifier"),
        ("collect_stats", "stats_num_workers", 0, "stats_num_workers"),
        ("train", "num_device", 0, "num_device"),
        ("train", "train_unregistered_specifier", "data", "list of nonempty"),
        ("train", "valid_registered_specifier", ["dialogue:valid:nan"], "factor"),
        ("collect_stats", "valid_registered_specifier", ["dialogue:a/b"], "name"),
    ],
)
def test_invalid_config_fails_before_launch(config, stage, key, value, message):
    config[key] = value
    with patch("subprocess.run", autospec=True) as run:
        with pytest.raises(ValueError, match=message):
            getattr(SpeechLMSystem(training_config=config), stage)()
        run.assert_not_called()


@pytest.mark.parametrize("stage", ["train", "collect_stats"])
def test_child_failure_propagates(config, stage):
    failure = subprocess.CalledProcessError(1, "speechlm")
    with patch("subprocess.run", side_effect=failure):
        with pytest.raises(subprocess.CalledProcessError) as error:
            getattr(SpeechLMSystem(training_config=config), stage)()
    assert error.value is failure


def test_training_requires_usable_stats(config, tmp_path):
    stats = tmp_path / "stats_dialogue_b.jsonl"
    stats.write_text("")
    with patch("subprocess.run", autospec=True) as run:
        for remove in (False, True):
            if remove:
                stats.unlink()
            with pytest.raises(FileNotFoundError, match=stats.name):
                SpeechLMSystem(training_config=config).train()
        run.assert_not_called()


def test_shared_stats_require_the_same_source(config):
    config.valid_unregistered_specifier = ["dialogue:a:train one.json:2"]
    with patch("subprocess.run", autospec=True) as run:
        SpeechLMSystem(training_config=config).collect_stats()
        run.assert_called_once()
        config.valid_unregistered_specifier = ["dialogue:a:train-two.json"]
        with pytest.raises(ValueError, match="Conflicting SpeechLM datasets"):
            SpeechLMSystem(training_config=config).collect_stats()
        run.assert_called_once()


def test_missing_manifest_fails_before_launch(config):
    config.train_unregistered_specifier = ["dialogue:a:missing.json"]
    with patch("subprocess.run", autospec=True) as run:
        with pytest.raises(FileNotFoundError, match="missing.json"):
            SpeechLMSystem(training_config=config).train()
        run.assert_not_called()


def test_fine_tuning_requires_weights_or_a_completed_checkpoint(config, tmp_path):
    config.require_checkpoint = True
    incomplete = tmp_path / "exp/checkpoints/step_10"
    incomplete.mkdir(parents=True)
    system = SpeechLMSystem(training_config=config)
    with patch("subprocess.run", autospec=True) as run:
        with pytest.raises(ValueError, match="Set resume_path"):
            system.train()
        run.assert_not_called()
        config.resume_path = str(incomplete)
        with pytest.raises(ValueError, match="Incomplete distributed checkpoint"):
            system.train()
        run.assert_not_called()
        config.resume_path = str(tmp_path / "initial.pt")
        (tmp_path / "initial.pt").touch()
        system.train()
        assert "--resume-path" in run.call_args.args[0]
        (incomplete / ".metadata").touch()
        system.train()
        assert "--resume-path" not in run.call_args.args[0]
