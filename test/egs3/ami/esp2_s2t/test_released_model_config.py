"""Tests for the configs that decode and pack the released checkpoint."""

import ami_sot_paths
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config


def _load(name, role):
    """Load a recipe config the way run.py does, without resolving it."""
    config = load_and_merge_config(
        ami_sot_paths.RECIPE / "conf" / name,
        config_name=f"{role}.yaml",
        default_package=ami_sot_paths.TEMPLATE_PACKAGE,
        resolve=False,
    )
    return OmegaConf.to_container(config)


def test_the_released_training_config_changes_only_the_exp_tag():
    """training_whisper_sot.yaml is training.yaml with its own exp_tag."""
    released = _load("training_whisper_sot.yaml", "training")
    default = _load("training.yaml", "training")

    assert released.pop("exp_tag") == "whisper_small_sot"
    default.pop("exp_tag")
    assert released == default


def test_the_released_inference_config_decodes_config_yaml_and_model_pth():
    """inference_whisper_sot.yaml reads the two released files from exp_dir.

    The inference directory also changes, because its name comes from the name
    of the config file. All other settings are those of inference.yaml.
    """
    released = _load("inference_whisper_sot.yaml", "inference")
    default = _load("inference.yaml", "inference")

    assert released["model"].pop("s2t_train_config") == "${exp_dir}/config.yaml"
    assert released["model"].pop("s2t_model_file") == "${exp_dir}/model.pth"
    assert released.pop("inference_dir") == "${exp_dir}/inference_whisper_sot"
    for key in ("s2t_train_config", "s2t_model_file"):
        default["model"].pop(key)
    default.pop("inference_dir")
    assert released == default
