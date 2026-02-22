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
    """Resolve and return an ESPnet2 Task class from a dotted class path.

    Args:
        task_path (str): Dotted path to an ESPnet2 Task class (Hydra-style),
            e.g., ``"espnet2.tasks.asr.ASRTask"``.

    Returns:
        type: The resolved Task class.

    Raises:
        RuntimeError: If the class cannot be imported/resolved.

    Example:
        >>> cls = get_task_class(\"espnet2.tasks.asr.ASRTask\")  # doctest: +SKIP
        >>> cls.__name__  # doctest: +SKIP
        'ASRTask'
    """
    try:
        ez_task = get_class(task_path)
    except Exception as e:
        raise RuntimeError(f"Failed to get task class from {task_path}: {e}")
    return ez_task


@typechecked
def get_espnet_model(task: str, config: Union[Dict, DictConfig]) -> AbsESPnetModel:
    """Build and return an ESPnet2 model from a task class and config.

    This is a thin wrapper around ``Task.get_default_config()`` and
    ``Task.build_model(...)``. It temporarily overrides ``sys.argv`` to satisfy
    some task implementations that rely on argument parsing.

    Args:
        task (str): Dotted path to an ESPnet2 Task class.
        config (Union[Dict, DictConfig]): Model/config overrides merged into the
            task default config.

    Returns:
        AbsESPnetModel: Instantiated ESPnet2 model.

    Raises:
        RuntimeError: If the task class cannot be resolved.

    Example:
        >>> model = get_espnet_model(
        ...     "espnet2.tasks.asr.ASRTask",
        ...     {"frontend": "default"},
        ... )  # doctest: +SKIP
    """
    ez_task = get_task_class(task)

    # workaround for calling get_default_config
    original_argv = sys.argv
    sys.argv = ["dummy.py"]
    try:
        default_config = ez_task.get_default_config()
    finally:
        sys.argv = original_argv

    if OmegaConf.is_config(config):
        default_config.update(OmegaConf.to_container(config, resolve=True))
    else:
        default_config.update(config)

    espnet_model = ez_task.build_model(Namespace(**default_config))
    return espnet_model


def save_espnet_config(
    task: str, config: Union[Dict, DictConfig], output_dir: str
) -> None:
    """Save an ESPnet2-compatible training config snapshot to disk.

    This builds the full config by merging:
      - The task's default config
      - The resolved model config (flattened into root keys)
      - Selected dataset preprocessor config (flattened into root keys)
      - Any remaining user-provided config keys

    Args:
        task (str): Dotted path to an ESPnet2 Task class.
        config (Union[Dict, DictConfig]): Training config (OmegaConf or dict).
        output_dir (str): Output directory where ``config.yaml`` is written.

    Returns:
        None

    Raises:
        RuntimeError: If the task class cannot be resolved.
        OSError: If the output directory cannot be created or written.

    Example:
        >>> save_espnet_config(\"espnet2.tasks.asr.ASRTask\", cfg, \"exp\")  # doctest: +SKIP
    """
    ez_task = get_task_class(task)

    # workaround for calling get_default_config
    original_argv = sys.argv
    sys.argv = ["dummy.py"]
    default_config = ez_task.get_default_config()
    sys.argv = original_argv

    resolved_config = (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else config
    )

    # set model config at the root level
    model_config = resolved_config.pop("model")
    if "_target_" in model_config:
        model_config.pop("_target_")
    default_config.update(model_config)

    # set the preprocessor config at the root level
    dataset_config = resolved_config.get("dataset")
    if dataset_config is not None and "preprocessor" in dataset_config:
        preprocess_config = dataset_config.pop("preprocessor")
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
