"""Helpers for the optional ``kaldiio`` dependency.

``kaldiio`` is only required to read/write Kaldi ark/scp files, and its license
restricts redistribution, which is a problem for software that is publicly
distributed (see https://github.com/espnet/espnet/issues/6529).  ESPnet
therefore does not install it by default: modules that may touch Kaldi formats
import it through :func:`import_kaldiio`, which defers the import until the
Kaldi code path is actually taken and raises an actionable error otherwise.
"""

import importlib
from types import ModuleType

KALDIIO_INSTALL_MESSAGE = (
    "`kaldiio` is not installed. It is an optional dependency of ESPnet, "
    "used only for Kaldi ark/scp I/O. "
    'Please install it with `pip install "espnet[kaldi]"` or `pip install kaldiio`.'
)


def import_kaldiio() -> ModuleType:
    """Import and return the ``kaldiio`` module.

    Returns:
        The imported ``kaldiio`` module.

    Raises:
        ImportError: If ``kaldiio`` is not installed, with a message explaining
            how to install it.
    """
    try:
        return importlib.import_module("kaldiio")
    except ImportError as e:
        raise ImportError(KALDIIO_INSTALL_MESSAGE) from e
