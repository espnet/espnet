from espnetez.task import get_easy_task
from espnetez.utils import load_yaml


def from_yaml(task, path):
    task_class = get_easy_task(task)
    config = load_yaml(path)

    # get default configuration from task class.
    default_config = task_class.get_default_config()
    default_config.update(config)

    return default_config


def update_finetune_config(task, pretrain_config, path):
    finetune_config = load_yaml(path)
    default_config = get_easy_task(task).get_default_config()

    # update pretrain_config with finetune_config
    # and update distributed related configs to the default.
    for k in list(pretrain_config.keys()):
        if "dist_" in k or "_rank" in k:
            pretrain_config[k] = default_config[k]
        elif k in finetune_config.keys() and pretrain_config[k] != finetune_config[k]:
            pretrain_config[k] = finetune_config[k]

    for k in list(default_config.keys()):
        if k not in pretrain_config.keys():
            pretrain_config[k] = default_config[k]

    if "preprocessor_conf" in pretrain_config.keys():
        pretrain_config["preprocessor_conf"] = finetune_config.get(
            "preprocessor_conf", {}
        )

    return pretrain_config
