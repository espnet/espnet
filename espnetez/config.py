from espnetez.utils import get_task_class, load_yaml


def from_yaml(task, path):
    task_class = get_task_class(task)
    config = load_yaml(path)

    # get default configuration from task class.
    default_config = task_class.get_default_config()
    default_config.update(config)

    return default_config
