#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Normalization pipelines.

A pipeline is configured exactly the way a VERSA metric list is -- a YAML
list of ``- name: X`` entries with per-entry keyword arguments::

    normalize:
      - name: unicode
        form: NFKC
      - name: lowercase
      - name: remove_punctuation
        keep: "'"
      - name: whitespace

The order is the order of the list, because normalization is not
commutative: removing punctuation before collapsing whitespace is not the
same as doing it after.

The resolved config travels with the pipeline and is written into the
result, so a score always carries the normalization that produced it. The
issue asks for exactly this, and it is the reason normalization is a first
class object here rather than a flag on each metric.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

from splet.normalizers import basic, whisper

# name -> factory. A factory takes the entry's keyword arguments and returns
# a ``str -> str`` callable.
NORMALIZER_CHOICES: Dict[str, Callable[..., Callable[[str], str]]] = {
    "lowercase": basic.lowercase_setup,
    "uppercase": basic.uppercase_setup,
    "remove_punctuation": basic.remove_punctuation_setup,
    "whitespace": basic.whitespace_setup,
    "remove_tokens": basic.remove_tokens_setup,
    "unicode": basic.unicode_setup,
    "whisper": whisper.whisper_setup,
}


class Normalizer:
    """An ordered pipeline of normalization steps."""

    def __init__(
        self,
        steps: Sequence[Callable[[str], str]],
        config: Sequence[Dict[str, Any]],
    ) -> None:
        """Initialize the pipeline.

        Args:
            steps: The callables to apply, in order.
            config: The config the steps were built from. Kept so that the
                result can report it verbatim.
        """
        self._steps = list(steps)
        self.config = copy.deepcopy(list(config))

    def __call__(self, text: str) -> str:
        """Apply every step in order."""
        for step in self._steps:
            text = step(text)
        return text

    def __len__(self) -> int:
        """Return the number of steps."""
        return len(self._steps)

    def __repr__(self) -> str:
        """Return a representation naming the steps in order."""
        names = ", ".join(entry["name"] for entry in self.config)
        return f"Normalizer([{names}])"


def build_normalizer(
    normalize_config: Optional[Sequence[Dict[str, Any]]],
) -> Normalizer:
    """Build a pipeline from a config list.

    Args:
        normalize_config: A list of ``{"name": ..., **kwargs}`` entries, or
            None for the identity pipeline. The identity is the default on
            purpose: SPLET never normalizes text that the caller did not ask
            it to normalize.

    Returns:
        The pipeline.

    Raises:
        ValueError: If an entry has no name or names an unknown step.
    """
    if not normalize_config:
        return Normalizer([], [])

    steps: List[Callable[[str], str]] = []
    resolved: List[Dict[str, Any]] = []
    for entry in normalize_config:
        entry = dict(entry)
        name = entry.pop("name", None)
        if name is None:
            raise ValueError(f"normalization entry has no name: {entry}")
        if name not in NORMALIZER_CHOICES:
            raise ValueError(
                f"unknown normalization step '{name}'. "
                f"Available: {sorted(NORMALIZER_CHOICES)}"
            )
        steps.append(NORMALIZER_CHOICES[name](**entry))
        resolved.append({"name": name, **entry})
    return Normalizer(steps, resolved)
