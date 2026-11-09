#!/usr/bin/env python3
"""Write the noise list for on-the-fly augmentation (data/noise.scp).

    python3 local/v2/noise_list.py ${MUSAN} ${CHIME3_BACKGROUND} > data/noise.scp

The same files as the non-speech training corpora: MUSAN noise/ and music/
files whose licence is CC0, CC BY or public domain (MUSAN's per-block
LICENSE), music without vocals (ANNOTATIONS), never speech/; CHiME-3
backgrounds of the bus and street only (cafe and pedestrian have babble,
which added to a non-speech region would be labelled non-speech).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters_nonspeech import _musan_licences  # noqa: E402


def musan_files(root: Path):
    for part in ("noise", "music"):
        base = root / part
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            wavs = {p.stem: p for p in sorted(sub.glob("*.wav"))}
            ok = _musan_licences(sub / "LICENSE", wavs)
            vocals = {}
            if part == "music":
                for line in (
                    (sub / "ANNOTATIONS").read_text(errors="replace").splitlines()
                ):
                    f = line.split()
                    if len(f) >= 3 and f[2] in ("Y", "N"):
                        vocals[f[0].rsplit(".", 1)[0]] = f[2] == "Y"
            for stem, p in wavs.items():
                if ok[stem] and not (part == "music" and vocals.get(stem, True)):
                    yield stem, p


def chime3_files(root: Path):
    for p in sorted(root.rglob("*.wav")):
        tokens = set(re.split(r"[^A-Z0-9]+", p.name.upper()))
        if tokens & {"BUS", "STR"}:
            yield p.stem, p


def main():
    musan, chime3 = Path(sys.argv[1]), Path(sys.argv[2])
    for key, path in sorted([*musan_files(musan), *chime3_files(chime3)]):
        print(key, path)


if __name__ == "__main__":
    main()
