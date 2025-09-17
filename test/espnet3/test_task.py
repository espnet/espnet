# tests/test_task_wrapper.py
from pathlib import Path
import pytest
from argparse import Namespace

# Replace with your actual module path
# Example: from espneteztask import get_task_class, save_espnet_config, get_espnet_model
from espnet3.task import get_task_class, get_espnet_model, save_espnet_config
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
    default_config = ASRTask.get_default_config()
    default_config["token_list"] = ["<blank>", "a", "b", "c"]
    default_config["model_conf"]["ctc_weight"] = 1.0
    model = ASRTask.build_model(Namespace(**default_config))
    model_espnet3 = get_espnet_model("asr", default_config)
    assert str(model) == str(model_espnet3) # Check all attributes are the same


def test_save_espnet_config(tmp_path):
    config_path = Path("test_utils") / "espnet3" / "config" / "model_ctc.yaml"
    output_file = tmp_path / "config.yaml"
    save_espnet_config(
        "asr",
        load_config_with_defaults(config_path),
        output_file
    )
    assert output_file.exists()
