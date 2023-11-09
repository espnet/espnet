from argparse import Namespace

import yaml


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
