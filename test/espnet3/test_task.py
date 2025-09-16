# tests/test_task_wrapper.py
import yaml
import pytest
from argparse import Namespace
from omegaconf import OmegaConf

# Replace with your actual module path
# Example: from espneteztask import get_task_class, save_espnet_config, get_espnet_model
from espnet3.task import get_task_class, save_espnet_config, get_espnet_model


# ===============================================================
# Test Case Summary for Task Wrapper (espnet3.task)
# ===============================================================
#
# Tests for `get_task_class(task_name)`
# | Test Name                               | Description                                                             | Expected Result             | # noqa: E501
# |----------------------------------------|-------------------------------------------------------------------------|-----------------------------| # noqa: E501
# | test_get_task_class_returns_correct_class[param-asr]  | Maps "asr" to ASRTask                                                   | cls.__name__ == "ASRTask"   | # noqa: E501
# | test_get_task_class_returns_correct_class[param-tts]  | Maps "tts" to TTSTask                                                   | cls.__name__ == "TTSTask"   | # noqa: E501
# | test_get_task_class_returns_correct_class[param-enh]  | Maps "enh" to EnhancementTask                                           | cls.__name__ == "EnhancementTask" | # noqa: E501
# | test_get_task_class_returns_correct_class[param-st]   | Maps "st"  to STTask                                                    | cls.__name__ == "STTask"    | # noqa: E501
# | test_get_task_class_returns_correct_class[param-lm]   | Maps "lm"  to LMTask                                                    | cls.__name__ == "LMTask"    | # noqa: E501
# | test_get_task_class_returns_correct_class[param-uasr] | Maps "uasr" to UASRTask                                                 | cls.__name__ == "UASRTask"  | # noqa: E501
# | test_get_task_class_unknown_raises_keyerror           | Unknown task string should fail                                         | raises KeyError             | # noqa: E501
#
# Tests for `save_espnet_config(task, cfg, outdir)`
# | Test Name                                  | Description                                                                                                   | Key Assertions                                                                                                       | # noqa: E501
# |-------------------------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------| # noqa: E501
# | test_save_espnet_config_creates_yaml_and_transforms | Writes config.yaml and normalizes structure for ESPnet consumption                                            | file exists; preserves default_key="keepme"; lifts model/_target_ & dataset.preprocessor/_target_ to root;            | # noqa: E501
# |                                           |                                                                                                               | removes all `_target_`; converts `*_conf=None`→{}; tuples→lists; keeps remaining dataset fields (e.g., other=123)     | # noqa: E501
#


@pytest.mark.parametrize(
    "task_name, expected_cls_name",
    [
        ("asr", "ASRTask"),
        ("tts", "TTSTask"),
        ("enh", "EnhancementTask"),
        ("st", "STTask"),
        ("lm", "LMTask"),
        ("uasr", "UASRTask"),
    ],
)
def test_get_task_class_returns_correct_class(task_name, expected_cls_name):
    cls = get_task_class(task_name)
    assert cls.__name__ == expected_cls_name


def test_get_task_class_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_task_class("unknown_task")


def test_save_espnet_config_creates_yaml_and_transforms(tmp_path):
    """
    save_espnet_config should:
      - create config.yaml
      - lift model/_target_ and dataset.preprocessor/_target_ fields to root
      - remove _target_ keys
      - convert *_conf=None to {}
      - convert tuple values to lists
      - keep default_config values
    """
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "SomeModel", "foo": 1, "bar": "x"},
            "dataset": {
                "preprocessor": {"_target_": "SomePre", "norm": "global"},
                "other": 123,
            },
            "extra_tuple": (10, 20),
            "some_conf": None,  # should become {}
        }
    )

    outdir = tmp_path / "out"
    save_espnet_config("asr", cfg, str(outdir))
    outpath = outdir / "config.yaml"
    assert outpath.exists()

    data = yaml.safe_load(outpath.read_text(encoding="utf-8"))

    # Default values from get_default_config are preserved
    assert data["default_key"] == "keepme"

    # Model and preprocessor fields are lifted and _target_ removed
    assert data["foo"] == 1 and data["bar"] == "x"
    assert data["norm"] == "global"
    assert "_target_" not in data

    # *_conf=None → {}
    assert data["some_conf"] == {}

    # Tuples are converted to lists
    assert data["a_tuple"] == [1, 2]
    assert data["extra_tuple"] == [10, 20]

    # Remaining dataset values are still present
    assert data["dataset"]["other"] == 123

