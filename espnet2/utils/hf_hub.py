"""Utilities for huggingface hub."""

import functools
from typing import Any
from typing import Dict
from typing import Optional

from huggingface_hub import cached_download
from huggingface_hub import hf_hub_url
import yaml

from espnet2 import __version__


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


def hf_rewrite_yaml(filepath: str, model_id: str, revision: Optional[str]):
    """hf_rewrite_yaml."""
    with open(filepath, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for rewrite_key in REWRITE_KEYS:
        v = nested_dict_get(d, rewrite_key)
        if v is not None and any(v.startswith(prefix) for prefix in ["exp", "data"]):
            file_url = hf_hub_url(model_id, filename=v, revision=revision)
            file_path = cached_download(
                file_url, library_name="espnet", library_version=__version__
            )
            nested_dict_set(d, rewrite_key, file_path)
    with open(filepath, "w", encoding="utf-8") as fw:
        yaml.safe_dump(d, fw)
