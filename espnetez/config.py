import yaml

from espnetez.task import get_ez_task


def convert_none_to_None(dic):
    for k, v in dic.items():
        if isinstance(v, dict):
            dic[k] = convert_none_to_None(dic[k])

        elif v == "none":
            dic[k] = None
    return dic


def from_yaml(task, path):
    task_class = get_ez_task(task)
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # get default configuration from task class.
    default_config = task_class.get_default_config()
    default_config.update(config)

    default_config = convert_none_to_None(default_config)

    return default_config


def update_finetune_config(task, pretrain_config, path):
    with open(path, "r") as f:
        finetune_config = yaml.load(f, Loader=yaml.Loader)
    default_config = get_ez_task(task).get_default_config()

    # update pretrain_config with finetune_config
    # and update distributed related configs to the default.
    for k in list(pretrain_config):
        if "dist_" in k or "_rank" in k:
            pretrain_config[k] = default_config[k]
        elif k in finetune_config and pretrain_config[k] != finetune_config[k]:
            pretrain_config[k] = finetune_config[k]

    for k in list(default_config):
        if k not in pretrain_config:
            pretrain_config[k] = default_config[k]

    if "preprocessor_conf" in pretrain_config:
        pretrain_config["preprocessor_conf"] = finetune_config.get(
            "preprocessor_conf", {}
        )

    pretrain_config = convert_none_to_None(pretrain_config)

    return pretrain_config
