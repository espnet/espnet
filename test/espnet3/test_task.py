# tests/test_task_wrapper.py
from argparse import Namespace
from pathlib import Path

import pytest

# Replace with your actual module path
# Example: from espnet3task import get_task_class, save_espnet_config, get_espnet_model
from espnet3.task import get_espnet_model, get_task_class, save_espnet_config
from espnet3.utils.config import load_config_with_defaults

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
    "task_path, expected_cls_name",
    [
        ("espnet3.wrapper.tasks.ASRTask", "ASRTask"),
        ("espnet3.wrapper.tasks.ASRTransducerTask", "ASRTransducerTask"),
        ("espnet3.wrapper.tasks.ASVSpoofTask", "ASVSpoofTask"),
        ("espnet3.wrapper.tasks.DiarizationTask", "DiarizationTask"),
        ("espnet3.wrapper.tasks.EnhancementTask", "EnhancementTask"),
        ("espnet3.wrapper.tasks.EnhS2TTask", "EnhS2TTask"),
        (
            "espnet3.wrapper.tasks.TargetSpeakerExtractionTask",
            "TargetSpeakerExtractionTask",
        ),
        ("espnet3.wrapper.tasks.GANSVSTask", "GANSVSTask"),
        ("espnet3.wrapper.tasks.GANTTSTask", "GANTTSTask"),
        ("espnet3.wrapper.tasks.HubertTask", "HubertTask"),
        ("espnet3.wrapper.tasks.LMTask", "LMTask"),
        ("espnet3.wrapper.tasks.MTTask", "MTTask"),
        ("espnet3.wrapper.tasks.S2STTask", "S2STTask"),
        ("espnet3.wrapper.tasks.S2TTask", "S2TTask"),
        ("espnet3.wrapper.tasks.SLUTask", "SLUTask"),
        ("espnet3.wrapper.tasks.SpeakerTask", "SpeakerTask"),
        ("espnet3.wrapper.tasks.STTask", "STTask"),
        ("espnet3.wrapper.tasks.SVSTask", "SVSTask"),
        ("espnet3.wrapper.tasks.TTSTask", "TTSTask"),
        ("espnet3.wrapper.tasks.UASRTask", "UASRTask"),
    ],
)
@pytest.mark.execution_timeout(30)
def test_get_task_class_returns_correct_class(task_path, expected_cls_name):
    cls = get_task_class(task_path)
    assert cls.__name__ == expected_cls_name


def test_get_espnet_model():
    from espnet3.wrapper.tasks import ASRTask

    default_config = ASRTask.get_default_config()
    default_config["token_list"] = ["<blank>", "a", "b", "c"]
    default_config["model_conf"]["ctc_weight"] = 1.0
    model = ASRTask.build_model(Namespace(**default_config))
    model_espnet3 = get_espnet_model("espnet3.wrapper.tasks.ASRTask", default_config)
    assert str(model) == str(model_espnet3)  # Check all attributes are the same


def test_save_espnet_config(tmp_path):
    config_path = Path("test_utils") / "espnet3" / "config" / "model_ctc.yaml"
    output_file = tmp_path / "config.yaml"
    save_espnet_config(
        "espnet3.wrapper.tasks.ASRTask", load_config_with_defaults(config_path), output_file
    )
    assert output_file.exists()
