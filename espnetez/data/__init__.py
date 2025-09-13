"""Public API for the :mod:`dump` package.

This module simply re-exports all public names from the :mod:`.dump`
submodule so that they can be imported directly from the package
namespace.  It allows users to write:

    >>> from mypkg import <name>

instead of:

    >>> from mypkg.dump import <name>

Only names that do not start with an underscore are re-exported, keeping
the public API surface intentionally small and stable.  Any additional
functionality should be added to :mod:`.dump` and will automatically
appear in the package namespace via this re-export.
"""

from .dump import *  # noqa
