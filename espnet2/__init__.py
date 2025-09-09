"""Initialize espnet2 package and set __version__."""

import os
from importlib import metadata as _metadata

_here = os.path.dirname(__file__)
_top_version_file = os.path.join(os.path.dirname(_here), "version.txt")
_pkg_version_file = os.path.join(_here, "version.txt")

__version__ = "0.0.0"

if os.path.exists(_top_version_file):
    with open(_top_version_file, "r") as f:
        __version__ = f.read().strip()
elif os.path.exists(_pkg_version_file):
    with open(_pkg_version_file, "r") as f:
        __version__ = f.read().strip()
else:
    try:
        __version__ = _metadata.version("espnet")
    except Exception:
        pass
