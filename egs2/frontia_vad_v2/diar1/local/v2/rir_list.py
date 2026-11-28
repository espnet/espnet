#!/usr/bin/env python3
"""Write the room impulse responses for augmentation and their list.

    python3 local/v2/rir_list.py ${VAD_CORPORA} data/rir > data/rir.scp

The impulse responses of BUT ReverbDB (CC BY 4.0;
BUT_ReverbDB_rel_19_06_RIR-Only.tgz, the RIR/*.wav of every room, microphone
and source position) are read from the archive in member order and written
as 16 kHz mono wav files under the output directory (the recipe's own data,
not the corpora directory). The list is "<id> <path>", sorted.
"""

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vad_dump import resample  # noqa: E402

ARCHIVE = "but_reverbdb/BUT_ReverbDB_rel_19_06_RIR-Only.tgz"


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    archive = root / ARCHIVE
    if not archive.exists():
        archive = root / "but_reverbdb" / "ARCHIVES" / Path(ARCHIVE).name
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(archive, "r|gz") as t:
        for m in t:
            if not (m.isfile() and m.name.endswith(".wav") and "/RIR/" in m.name):
                continue
            x, rate = soundfile.read(
                io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=True
            )
            x = resample(x[:, :1], rate, 16000)[:, 0]
            rid = "but_" + "_".join(m.name.rsplit(".", 1)[0].split("/")[1:])
            rid = "".join(c if c.isalnum() or c in "-_" else "_" for c in rid)
            path = out / f"{rid}.wav"
            soundfile.write(
                path, np.asarray(x, dtype=np.float32), 16000, subtype="FLOAT"
            )
            entries.append((rid, path.resolve()))
    for rid, path in sorted(entries):
        print(rid, path)


if __name__ == "__main__":
    main()
