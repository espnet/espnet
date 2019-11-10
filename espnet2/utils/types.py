from distutils.util import strtobool
from typing import Optional

import yaml
from typeguard import typechecked


@typechecked
def str2bool(value: str) -> bool:
    return bool(strtobool(value))


@typechecked
def int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return int(value)


@typechecked
def str_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    if value.lower() in ('none', 'null', 'nil'):
        return None
    return value


@typechecked
def yaml_load(value: str):
    return yaml.load(value, Loader=yaml.Loader)
