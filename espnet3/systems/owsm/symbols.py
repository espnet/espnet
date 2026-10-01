"""Load an OWSM symbol inventory from a config value."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from hydra.utils import instantiate
from omegaconf import DictConfig


def load_symbols(spec: Any) -> List[str]:
    """Return the symbols named by ``spec``.

    Three spellings, because the inventory is over 1700 symbols: a hydra
    callable that builds it, a file with one symbol per line, or a literal
    list. A recipe owns its own inventory -- which symbols exist depends on
    which languages and tasks its corpora emit -- so there is no default here.
    """
    if isinstance(spec, DictConfig) or (
        isinstance(spec, dict) and "_target_" in spec
    ):
        spec = instantiate(spec, _convert_="all")
    if isinstance(spec, (str, os.PathLike)):
        path = Path(spec)
        if not path.is_file():
            raise RuntimeError(f"nlsyms file not found: {path}")
        return path.read_text(encoding="utf-8").split()
    return [str(symbol) for symbol in spec]
