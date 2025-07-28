# ESPnet-EZ Task class
# This class is a wrapper for Task classes to support custom datasets.
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Union

import yaml
from omegaconf import DictConfig, OmegaConf
from typeguard import typechecked

from espnet2.train.abs_espnet_model import AbsESPnetModel


def get_task_class(task_name: str):
    """
    Lazily import and return the task class corresponding to the given task name.

    This function avoids importing all task modules at the top-level, which helps
    prevent unnecessary dependency errors for users who only use a subset of tasks
    (e.g., ASR-only users don't need TTS or Enhancement dependencies).

    Args:
        task_name (str): The name of the task (e.g., "asr", "enh", "tts", etc.).

    Returns:
        AbsTask: The corresponding task class.

    Raises:
        KeyError: If the task_name is not recognized.

    Supported task names:
        - "asr"
        - "asr_transducer"
        - "asvspoof"
        - "diar"
        - "enh"
        - "enh_s2t"
        - "enh_tse"
        - "gan_svs"
        - "gan_tts"
        - "hubert"
        - "lm"
        - "mt"
        - "s2st"
        - "s2t"
        - "slu"
        - "spk"
        - "st"
        - "svs"
        - "tts"
        - "uasr"

    Example:
        >>> task_class = get_task_class("asr")
        >>> task = task_class()
    """
    if task_name == "asr":
        from espnet2.tasks.asr import ASRTask

        return ASRTask
    elif task_name == "asr_transducer":
        from espnet2.tasks.asr_transducer import ASRTransducerTask

        return ASRTransducerTask
    elif task_name == "asvspoof":
        from espnet2.tasks.asvspoof import ASVSpoofTask

        return ASVSpoofTask
    elif task_name == "diar":
        from espnet2.tasks.diar import DiarizationTask

        return DiarizationTask
    elif task_name == "enh":
        from espnet2.tasks.enh import EnhancementTask

        return EnhancementTask
    elif task_name == "enh_s2t":
        from espnet2.tasks.enh_s2t import EnhS2TTask

        return EnhS2TTask
    elif task_name == "enh_tse":
        from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask

        return TargetSpeakerExtractionTask
    elif task_name == "gan_svs":
        from espnet2.tasks.gan_svs import GANSVSTask

        return GANSVSTask
    elif task_name == "gan_tts":
        from espnet2.tasks.gan_tts import GANTTSTask

        return GANTTSTask
    elif task_name == "hubert":
        from espnet2.tasks.hubert import HubertTask

        return HubertTask
    elif task_name == "lm":
        from espnet2.tasks.lm import LMTask

        return LMTask
    elif task_name == "mt":
        from espnet2.tasks.mt import MTTask

        return MTTask
    elif task_name == "s2st":
        from espnet2.tasks.s2st import S2STTask

        return S2STTask
    elif task_name == "s2t":
        from espnet2.tasks.s2t import S2TTask

        return S2TTask
    elif task_name == "slu":
        from espnet2.tasks.slu import SLUTask

        return SLUTask
    elif task_name == "spk":
        from espnet2.tasks.spk import SpeakerTask

        return SpeakerTask
    elif task_name == "st":
        from espnet2.tasks.st import STTask

        return STTask
    elif task_name == "svs":
        from espnet2.tasks.svs import SVSTask

        return SVSTask
    elif task_name == "tts":
        from espnet2.tasks.tts import TTSTask

        return TTSTask
    elif task_name == "uasr":
        from espnet2.tasks.uasr import UASRTask

        return UASRTask
    else:
        raise KeyError(f"Unknown task: {task_name}")


@typechecked
def get_espnet_model(task: str, config: Union[Dict, DictConfig]) -> AbsESPnetModel:
    ez_task = get_task_class(task)

    # workaround for calling get_default_config
    original_argv = sys.argv
    sys.argv = ["dummy.py"]
    default_config = ez_task.get_default_config()
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
    ez_task = get_task_class(task)

    # workaround for calling get_default_config
    original_argv = sys.argv
    sys.argv = ["dummy.py"]
    default_config = ez_task.get_default_config()
    sys.argv = original_argv

    resolved_config = OmegaConf.to_container(config, resolve=True)

    # set model config at the root level
    model_config = resolved_config.pop("model")
    if hasattr(model_config, "_target_"):
        model_config.pop("_target_")
    default_config.update(model_config)

    # set the preprocessor config at the root level
    if hasattr(config.dataset, "preprocessor"):
        preprocess_config = resolved_config["dataset"].pop("preprocessor")
        if hasattr(preprocess_config, "_target_"):
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
