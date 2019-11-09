from distutils.util import strtobool
from typing import Optional

import yaml


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def int_or_none(value: str) -> Optional[int]:
    if value in (None, 'none', 'None', 'NONE', 'null', 'Null', 'NULL'):
        return None
    return int(value)


def str_or_none(value: str) -> Optional[str]:
    if value is None:
        return None
    return value


def yaml_load(value: str):
    return yaml.load(value, Loader=yaml.Loader)
