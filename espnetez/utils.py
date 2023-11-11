import yaml


def load_yaml(path):
    with open(path, "r") as f:
        yml = yaml.load(f, Loader=yaml.Loader)
    return yml
