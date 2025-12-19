"""ESPnet-3 Task class."""

import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Union

import yaml
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from typeguard import typechecked

from espnet2.train.abs_espnet_model import AbsESPnetModel


def get_task_class(task_path: str):
    """Get the ESPnet-2 Task class from the given task path."""
    try:
        ez_task = get_class(task_path)
    except Exception as e:
        raise RuntimeError(f"Failed to get task class from {task_path}: {e}")
    return ez_task


@typechecked
def get_espnet_model(task: str, config: Union[Dict, DictConfig]) -> AbsESPnetModel:
    """Build and return an ESPnet model from the given task and config."""
    ez_task = get_task_class(task)

    # workaround for calling get_default_config
    original_argv = sys.argv
    sys.argv = ["dummy.py"]
    try:
        default_config = ez_task.get_default_config()
    finally:
        sys.argv = original_argv

    if isinstance(config, OmegaConf):
        default_config.update(OmegaConf.to_container(config, resolve=True))
    else:
        default_config.update(config)

    espnet_model = ez_task.build_model(Namespace(**default_config))
    return espnet_model


def save_espnet_config(
    task: str, config: Union[Dict, DictConfig], output_dir: str
) -> None:
    """Save the ESPnet config used for training to the output directory."""
    ez_task = get_task_class(task)

    # workaround for calling get_default_config
    original_argv = sys.argv
    sys.argv = ["dummy.py"]
    default_config = ez_task.get_default_config()
    sys.argv = original_argv

    resolved_config = OmegaConf.to_container(config, resolve=True)

    # set model config at the root level
    model_config = resolved_config.pop("model")
    if "_target_" in model_config:
        model_config.pop("_target_")
    default_config.update(model_config)

    # set the preprocessor config at the root level
    if config.dataset is not None and "preprocessor" in config.dataset:
        preprocess_config = resolved_config["dataset"].pop("preprocessor")
        if "_target_" in preprocess_config:
            preprocess_config.pop("_target_")
        default_config.update(preprocess_config)

    default_config.update(resolved_config)

    # Check if there is None in the config with the name "*_conf"
    for k, v in default_config.items():
        if k.endswith("_conf") and v is None:
            default_config[k] = {}

    # Convert tuple into list
    for k, v in default_config.items():
        if isinstance(v, tuple):
            default_config[k] = list(v)

    # Save the config to the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "config.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f)
