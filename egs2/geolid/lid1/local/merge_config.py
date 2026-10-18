#!/usr/bin/env python3
"""Overlay a small GeoLID tuning config onto conf/train.yaml."""

import argparse
from pathlib import Path

import yaml


def merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("overrides", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = merge(
        yaml.safe_load(args.base.read_text()),
        yaml.safe_load(args.overrides.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
