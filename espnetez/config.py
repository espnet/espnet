from espnetez.utils import load_yaml
from espnetez.task import get_easy_task


def from_yaml(task, path):
    task_class = get_easy_task(task)
    config = load_yaml(path)

    # get default configuration from task class.
    default_config = task_class.get_default_config()
    default_config.update(config)

    return default_config
