"""Speaker embedding extraction entrypoints for ESPnet3."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from espnet2.bin.spk_embed_extract import (
    extract_embed,
    get_parser as _get_parser,
    main as _main,
)


def get_embed_extract_parser():
    """Return the embedding-extraction CLI parser."""
    return _get_parser()


def main_embed_extract(cmd: Optional[Sequence[str]] = None) -> Any:
    """CLI-compatible embedding-extraction entrypoint."""
    return _main(cmd=cmd)
