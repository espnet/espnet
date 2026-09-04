"""Helpers for the optional Kaldi ark/scp backend.

Kaldi ark/scp support lives in :mod:`omniio.kaldi`, which ESPnet does not
install by default: it is only needed to read or write Kaldi-format features,
and the default ``raw`` recipe path never touches them.  Modules that may hit a
Kaldi code path import it through :func:`import_kaldi_io`, which defers the
import until that branch is actually taken and otherwise raises an error saying
how to install it.

ESPnet previously used ``kaldiio`` here.  It was replaced because its license
restricts redistribution, which is a problem for anyone publicly distributing
software that depends on ESPnet (see https://github.com/espnet/espnet/issues/6529);
``omniio`` is MIT-licensed and writes byte-identical archives.
"""

import importlib
from types import ModuleType

KALDI_IO_INSTALL_MESSAGE = (
    "`omniio` is not installed. It is an optional dependency of ESPnet, "
    "used only for Kaldi ark/scp I/O. "
    'Please install it with `pip install "espnet[omniio]"` or `pip install omniio`.'
)


def import_kaldi_io() -> ModuleType:
    """Import and return the ``omniio.kaldi`` module.

    The returned module exposes the same names as ``kaldiio``
    (``ReadHelper``, ``WriteHelper``, ``load_scp``, ``load_mat``, ``save_ark``,
    ``open_like_kaldi``, ...).

    Returns:
        The imported ``omniio.kaldi`` module.

    Raises:
        ImportError: If ``omniio`` is not installed, with a message explaining
            how to install it.
    """
    try:
        return importlib.import_module("omniio.kaldi")
    except ImportError as e:
        raise ImportError(KALDI_IO_INSTALL_MESSAGE) from e
