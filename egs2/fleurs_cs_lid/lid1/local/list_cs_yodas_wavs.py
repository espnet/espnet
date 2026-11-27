#!/usr/bin/env python3
"""Write <root>/<lang>.paths, the CS-YODAS wav files used by the recipe.

Used by local/download_cs_yodas.sh to extract only these files.
"""

import sys
from collections import defaultdict
from pathlib import Path

from prepare_fleurs_cs_lid_data import CS_YODAS_LANGUAGES, load_cs_yodas


def main() -> None:
    """List the wav paths (relative to <root>/audio) per base language."""
    root = Path(sys.argv[1])
    audio_dir = (root / "audio").resolve()
    paths = defaultdict(set)
    for example, _ in load_cs_yodas(root):
        paths[example.labels[0]].add(str(Path(example.wav).relative_to(audio_dir)))
    for lang in CS_YODAS_LANGUAGES:
        lines = "".join(f"{path}\n" for path in sorted(paths[lang]))
        (root / f"{lang}.paths").write_text(lines, encoding="utf-8")


if __name__ == "__main__":
    main()
