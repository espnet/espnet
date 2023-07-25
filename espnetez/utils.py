import yaml
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr_transducer import ASRTransducerTask
from espnet2.tasks.tts import TTSTask
from argparse import Namespace

TASK_CLASSES = dict(
    asr=ASRTask,
    asr_transducer=ASRTransducerTask,
    tts=TTSTask,
)


def get_task_class(task_name):
    """Returns the task class for the given task name.

    Args:
        task_name: The name of the task.

    Returns:
        The task class.
    """
    return TASK_CLASSES[task_name]


def load_yaml(path):
    with open(path, "r") as f:
        yml = yaml.load(f, Loader=yaml.Loader)
    return yml


def update_config(config, yml):
    """Updates the config with the given yaml.

    Args:
        config: The config to be updated.
        yml: The yaml to be updated.
    """
    config = vars(config)
    for k, v in yml.items():
        if isinstance(v, dict):
            if k not in config:
                config[k] = {}
            update_config(config[k], v)
        else:
            config[k] = v
    return Namespace(**config)
