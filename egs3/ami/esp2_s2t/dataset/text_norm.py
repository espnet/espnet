"""Transcript normalizers the serialization may apply to segment text.

``chime8_keep_fillers`` is the CHiME-8 DASR English normalizer with filler
removal switched off. It comes from chime-utils, which is not on PyPI, so it
is an optional dependency; ``none`` needs nothing.
"""

from __future__ import annotations

from typing import Callable, Optional

_INSTALL_HINT = "pip install git+https://github.com/chimechallenge/chime-utils@main"

# Name -> one-line description, for the error message and the readme.
NORMALIZERS = {
    "none": "leave the transcript as the cutset spells it",
    "chime8_keep_fillers": (
        "CHiME-8 DASR English normalizer with filler removal switched off"
    ),
}


def get_text_norm(name: Optional[str]) -> Optional[Callable[[str], str]]:
    """Return the normalizer ``name`` asks for, or None for no normalization.

    Args:
        name: A key of :data:`NORMALIZERS`, or None, which means ``"none"``.

    Returns:
        A callable mapping one segment's text to its normalized form, or None
        when no normalization is asked for. None rather than the identity
        function, so a caller can tell the two apart and skip the call.

    Raises:
        ValueError: When ``name`` is not a known normalizer.
        ImportError: When the normalizer needs chime-utils and it is missing.
            The message carries the install command, because the package is
            not on PyPI and pip's own error would not say where to find it.
    """
    if name is None or name == "none":
        return None

    if name not in NORMALIZERS:
        raise ValueError(
            f"Unknown text_norm {name!r}; expected one of {sorted(NORMALIZERS)}."
        )

    try:
        from chime_utils.text_norm.whisper_like import EnglishTextNormalizer
    except ImportError as exc:
        raise ImportError(
            f"text_norm={name!r} needs the CHiME-8 normalizer from "
            f"chime-utils, which is not installed and is not on PyPI:\n"
            f"    {_INSTALL_HINT}\n"
            "Set text_norm: none in dataset/config.yaml to build without it."
        ) from exc

    # Not get_txt_norm("chime8"): that takes no arguments, so it deletes
    # fillers, and AMI is full of them.
    return EnglishTextNormalizer(remove_fillers=False)
