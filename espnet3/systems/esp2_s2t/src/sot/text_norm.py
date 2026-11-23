"""Text normalizers that the serialization can apply to segment text.

All esp2_s2t recipes share this lookup. Each corpus chooses its normalizer in
its own config, and this module finds the normalizer by name.

``chime8_keep_fillers`` is the CHiME-8 DASR English normalizer with filler
removal turned off. It comes from chime-utils, which is not on PyPI, so it is
an optional dependency. ``none`` needs no package.
"""

from __future__ import annotations

from typing import Callable, Optional

# chime-utils is not on PyPI, and pip rejects a URL dependency in a published
# package's metadata, so it cannot be an espnet extra.
_INSTALL_HINT = "pip install git+https://github.com/chimechallenge/chime-utils"

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
        A function that maps the text of one segment to its normalized form,
        or None when no normalization is set. None is returned instead of an
        identity function, so that a caller can skip the call.

    Raises:
        ValueError: When ``name`` is not a known normalizer.
        ImportError: When the normalizer needs chime-utils and it is missing.
            The message gives the install command, because the package is not
            on PyPI, and the pip error would not say where to find it.
    """
    if name is None or name == "none":
        return None

    if name not in NORMALIZERS:
        raise ValueError(
            f"Unknown text_norm {name!r}. Expected one of {sorted(NORMALIZERS)}."
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
