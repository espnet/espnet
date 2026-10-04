#!/usr/bin/env python3
"""Deprecated: this module is now `espnet2.bin.s2t_align`.

It was the only one of the four alignment modules with the algorithm in its
name, next to `align.py` and `asr_align.py`, and the name moved so that the
four read alike. Everything here forwards to the new module, so code and
recipes written against the old name keep working and say where it went::

    from espnet2.bin.s2t_align import CTCSegmentation

The script runs from either path with the same arguments.
"""

import warnings

from espnet2.bin import s2t_align
from espnet2.bin.s2t_align import CTCSegmentationTask  # noqa: F401

_MOVED = (
    "espnet2.bin.s2t_ctc_align is now espnet2.bin.s2t_align, so that the "
    "alignment modules read alike; this one forwards and will be removed in "
    "a future release"
)


class CTCSegmentation(s2t_align.CTCSegmentation):
    """Deprecated: use `espnet2.bin.s2t_align.CTCSegmentation`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(_MOVED, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


def ctc_align(**kwargs):
    """Deprecated: use `espnet2.bin.s2t_align.ctc_align`."""
    warnings.warn(_MOVED, DeprecationWarning, stacklevel=2)
    return s2t_align.ctc_align(**kwargs)


def get_parser():
    """Obtain the parser of `espnet2.bin.s2t_align`, under the old name."""
    parser = s2t_align.get_parser()
    parser.description += " (moved: run espnet2/bin/s2t_align.py instead)"
    return parser


def main(cmd=None):
    """Deprecated: run `espnet2/bin/s2t_align.py` instead."""
    warnings.warn(_MOVED, DeprecationWarning, stacklevel=2)
    return s2t_align.main(cmd)


if __name__ == "__main__":
    main()
