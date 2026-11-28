import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from espnet3.systems.speechlm.system import SpeechLMSystem
from espnet3.utils.config_utils import load_and_merge_config


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[4] / "egs3/bagpiper/speechlm"
    clone = tmp_path / "standalone"
    shutil.copytree(source / "conf", clone / "conf")
    monkeypatch.chdir(clone)
    for key in (
        "BAGPIPER_NGPU",
        "BAGPIPER_LORA_NGPU",
        "BAGPIPER_TRAIN_CONFIG",
        "BAGPIPER_LORA_CONFIG",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BAGPIPER_PREPARED", str(tmp_path / "data"))
    for split in ("train", "valid"):
        for partition in ("generation", "understanding"):
            manifest = tmp_path / "data" / partition / f"{split}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}")
            stats = (
                clone / "exp/sft_stats" / f"stats_dialogue_{partition}_{split}.jsonl"
            )
            stats.parent.mkdir(parents=True, exist_ok=True)
            stats.write_text('{"example": 128}\n')
    return clone


def _load(recipe, name):
    return load_and_merge_config(
        recipe / "conf" / name,
        config_name="training.yaml",
        default_package="egs3.TEMPLATE.speechlm",
    )


def test_full_and_lora_use_the_same_checkpoint_and_sft_data(recipe, monkeypatch):
    weights = recipe / "base.pt"
    weights.touch()
    monkeypatch.setenv("BAGPIPER_INIT", str(weights))
    full = _load(recipe, "training_full.yaml")
    lora = _load(recipe, "training_lora.yaml")
    assert full.stats_dir == lora.stats_dir
    assert full.train_unregistered_specifier == lora.train_unregistered_specifier
    assert full.valid_unregistered_specifier == lora.valid_unregistered_specifier
    assert full.exp_dir != lora.exp_dir
    for config, workers in [(full, "8"), (lora, "1")]:
        with patch("subprocess.run", autospec=True) as run:
            SpeechLMSystem(training_config=config).train()
        args = run.call_args.args[0]
        assert args[args.index("--nproc_per_node") + 1] == workers
        assert args[args.index("--train-config") + 1] == config.speechlm_config
        assert args[args.index("--resume-path") + 1] == str(weights)
    native = yaml.safe_load(Path(full.speechlm_config).read_text())
    adapted = yaml.safe_load(Path(lora.speechlm_config).read_text())
    # Token ranges depend on the insertion order as well as the IO settings.
    assert list(native["multimodal_io"].items()) == list(
        adapted["multimodal_io"].items()
    )
    for key in ("model", "preprocessor"):
        assert native[key] == adapted[key]
    assert "lora" not in native["trainer"]
    assert native["trainer"]["freeze_param"] == ["multimodal_io_dict.discrete_audio"]
    for settings, config in [(native, full), (adapted, lora)]:
        assert settings["data_loading"]["batch_size"] == 4096
        accumulation = settings["trainer"]["gradient_accumulation_steps"]
        assert config.num_device * accumulation == 8
    legacy = _load(recipe, "training.yaml")
    assert legacy.num_device == 2
    assert legacy.exp_dir not in (lora.exp_dir, full.exp_dir)
    legacy_trainer = yaml.safe_load(Path(legacy.speechlm_config).read_text())["trainer"]
    assert "multimodal_io_dict.continuous_audio" in legacy_trainer["freeze_param"]
    assert adapted["trainer"]["lora"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_lora_can_use_two_gpus_and_requires_initial_weights(recipe, monkeypatch):
    monkeypatch.setenv("BAGPIPER_NGPU", "8")
    monkeypatch.setenv("BAGPIPER_LORA_NGPU", "2")
    monkeypatch.delenv("BAGPIPER_INIT", raising=False)
    config = _load(recipe, "training_lora.yaml")
    assert config.num_device == 2
    with patch("subprocess.run", autospec=True) as run:
        with pytest.raises(ValueError, match="initial weights"):
            SpeechLMSystem(training_config=config).train()
        run.assert_not_called()
