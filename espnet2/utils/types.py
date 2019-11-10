from distutils.util import strtobool
from typing import Optional

import yaml


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return int(value)


def str_or_none(value: str) -> Optional[str]:
    if value.lower() in ('none', 'null', 'nil'):
        return None
    if value is None:
        return value
    return value


def yaml_load(value: str):
    return yaml.load(value, Loader=yaml.Loader)
