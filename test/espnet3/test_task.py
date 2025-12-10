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
        ("espnet2.tasks.asr.ASRTask", "ASRTask"),
        ("espnet2.tasks.asr_transducer.ASRTransducerTask", "ASRTransducerTask"),
        ("espnet2.tasks.asvspoof.ASVSpoofTask", "ASVSpoofTask"),
        ("espnet2.tasks.diar.DiarizationTask", "DiarizationTask"),
        ("espnet2.tasks.enh.EnhancementTask", "EnhancementTask"),
        ("espnet2.tasks.enh_s2t.EnhS2TTask", "EnhS2TTask"),
        (
            "espnet2.tasks.enh_tse.TargetSpeakerExtractionTask",
            "TargetSpeakerExtractionTask",
        ),
        ("espnet2.tasks.gan_svs.GANSVSTask", "GANSVSTask"),
        ("espnet2.tasks.gan_tts.GANTTSTask", "GANTTSTask"),
        ("espnet2.tasks.hubert.HubertTask", "HubertTask"),
        ("espnet2.tasks.lm.LMTask", "LMTask"),
        ("espnet2.tasks.mt.MTTask", "MTTask"),
        ("espnet2.tasks.s2st.S2STTask", "S2STTask"),
        ("espnet2.tasks.s2t.S2TTask", "S2TTask"),
        ("espnet2.tasks.slu.SLUTask", "SLUTask"),
        ("espnet2.tasks.spk.SpeakerTask", "SpeakerTask"),
        ("espnet2.tasks.st.STTask", "STTask"),
        ("espnet2.tasks.svs.SVSTask", "SVSTask"),
        ("espnet2.tasks.tts.TTSTask", "TTSTask"),
        ("espnet2.tasks.uasr.UASRTask", "UASRTask"),
    ],
)
@pytest.mark.execution_timeout(30)
def test_get_task_class_returns_correct_class(task_path, expected_cls_name):
    cls = get_task_class(task_path)
    assert cls.__name__ == expected_cls_name


def test_get_espnet_model():
    from espnet2.tasks.asr import ASRTask

    default_config = ASRTask.get_default_config()
    default_config["token_list"] = ["<blank>", "a", "b", "c"]
    default_config["model_conf"]["ctc_weight"] = 1.0
    model = ASRTask.build_model(Namespace(**default_config))
    model_espnet3 = get_espnet_model("espnet2.tasks.asr.ASRTask", default_config)
    assert str(model) == str(model_espnet3)  # Check all attributes are the same


def test_save_espnet_config(tmp_path):
    config_path = Path("test_utils") / "espnet3" / "config" / "model_ctc.yaml"
    output_file = tmp_path / "config.yaml"
    save_espnet_config(
        "espnet2.tasks.asr.ASRTask", load_config_with_defaults(config_path), output_file
    )
    assert output_file.exists()
