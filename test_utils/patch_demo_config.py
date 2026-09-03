"""Patch demo.yaml to point model.dir_or_tag at a local model pack directory.

Usage:
    python3 ci/patch_demo_config.py <demo_dir> <model_dir>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

demo_dir = Path(sys.argv[1]).resolve()
model_dir = Path(sys.argv[2]).resolve()
config_path = demo_dir / "demo.yaml"
cfg = OmegaConf.load(config_path)
cfg.model.dir_or_tag = os.path.relpath(model_dir, start=demo_dir)
cfg.model.trust_user_code = True
config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
