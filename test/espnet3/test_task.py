# tests/test_task_wrapper.py
import yaml
import pytest
from omegaconf import OmegaConf

# Replace with your actual module path
# Example: from espneteztask import get_task_class, save_espnet_config, get_espnet_model
from espnet3.task import get_task_class, save_espnet_config


# ===============================================================
# Test Case Summary for Task Wrapper (espnet3.task)
# ===============================================================
#
# Tests for `get_task_class(task_name)`
# | Test Name                               | Description                      |
# |----------------------------------------|-----------------------------------|
# | test_get_task_class_returns_correct_class  | Maps "asr" to ASRTask         |
#


@pytest.mark.parametrize(
    "task_name, expected_cls_name",
    [
        ("asr", "ASRTask"),
        ("asr_transducer", "ASRTransducerTask"),
        ("asvspoof", "ASVSpoofTask"),
        ("diar", "DiarizationTask"),
        ("enh", "EnhancementTask"),
        ("enh_s2t", "EnhS2TTask"),
        ("enh_tse", "TargetSpeakerExtractionTask"),
        ("gan_svs", "GANSVSTask"),
        ("gan_tts", "GANTTSTask"),
        ("hubert", "HubertTask"),
        ("lm", "LMTask"),
        ("mt", "MTTask"),
        ("s2st", "S2STTask"),
        ("s2t", "S2TTask"),
        ("slu", "SLUTask"),
        ("spk", "SpeakerTask"),
        ("st", "STTask"),
        ("svs", "SVSTask"),
        ("tts", "TTSTask"),
        ("uasr", "UASRTask"),
    ],
)
def test_get_task_class_returns_correct_class(task_name, expected_cls_name):
    cls = get_task_class(task_name)
    assert cls.__name__ == expected_cls_name


def test_get_espnet_model():
    from espnet2.tasks.asr import ASRTask

    task = ASRTask()
    model = task.get_espnet_model()
    assert model is task.model


def test_save_espnet_config(tmp_path):
    from espnet2.tasks.asr import ASRTask

    task = ASRTask()
    config_path = tmp_path / "config.yaml"
    save_espnet_config(task, config_path)

    # Load the saved config to verify its contents
    with open(config_path, "r") as f:
        loaded_config = yaml.safe_load(f)

    # Convert task's config to a dictionary for comparison
    task_config_dict = OmegaConf.to_container(task.config, resolve=True)

    assert loaded_config == task_config_dict

