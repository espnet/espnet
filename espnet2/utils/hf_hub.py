"""Utilities for huggingface hub."""

from filelock import FileLock
import functools
import os
from typing import Any
from typing import Dict
import yaml


REWRITE_KEYS = [
    "bpemodel",
    "normalize_conf.stats_file",
]


def nested_dict_get(dictionary: Dict, dotted_key: str):
    """nested_dict_get."""
    keys = dotted_key.split(".")
    return functools.reduce(lambda d, key: d.get(key) if d else None, keys, dictionary)


def nested_dict_set(dictionary: Dict, dotted_key: str, v: Any):
    """nested_dict_set."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = v


def hf_rewrite_yaml(yaml_file: str, cached_dir: str):
    """hf_rewrite_yaml."""
    touch_path = yaml_file + ".touch"
    lock_path = yaml_file + ".lock"

    with FileLock(lock_path):
        if not os.path.exists(touch_path):

            with open(yaml_file, "r", encoding='utf-8') as f:
                d = yaml.safe_load(f)

            for rewrite_key in REWRITE_KEYS:
                v = nested_dict_get(d, rewrite_key)
                if v is not None and any(v.startswith(prefix) for prefix in ["exp", "data"]):
                    new_value = os.path.join(cached_dir, v)
                    nested_dict_set(d, rewrite_key, new_value)
            with open(yaml_file, "w", encoding="utf-8") as fw:
                yaml.safe_dump(d, fw)
            
            with open(touch_path, 'a'):
                os.utime(touch_path, None)
