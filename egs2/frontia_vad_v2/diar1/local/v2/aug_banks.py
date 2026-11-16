#!/usr/bin/env python3
"""Write the music and effects banks for on-the-fly augmentation, with licences.

    python3 local/v2/aug_banks.py ${VAD_CORPORA} data/aug_banks \
        [--music_scp data/music.scp] [--noise_scp data/noise_aug.scp]

Two lists for BatchWavAugment (wav_augment_conf music_scp and noise_scp):

- music: MUSAN music/ without vocals (CC0/BY/PD files; as local/v2/noise_list.py),
  Slakh2100 mixes without voice-like patches (CC BY 4.0), OpenMIC-2018 clips
  annotated without voice whose track licence is CC0 or CC BY, and FSD50K clips
  under the Music branch of the ontology that are CC0 or CC BY and have no speech,
  singing or vocal labels (adapters_nonspeech.fsd_kind "env").
- effects (the noise list): MUSAN noise/ and the CHiME-3 backgrounds (as
  noise_list.py), plus FSD50K "env" clips outside the Music branch (CC0/BY),
  up to --effects_hours.

The clips of Slakh2100, OpenMIC-2018 and FSD50K are read from their archives by
the readers of the training corpora (the same licence and voice filters), only
those of the train split (so nothing of the dev split is mixed into training
audio), resampled to 16 kHz, first channel, and concatenated in file order into
16-bit wav packs of about --pack_seconds (music) or --effects_pack_seconds
(effects): BatchWavAugment draws one list entry uniformly per utterance and a
random offset in it, so a pack stands for one source and its offsets for its
clips. MUSAN and CHiME-3 files are listed as they are.

<out>/licences.tsv lists every clip: bank, pack file, start and end (s) in the
pack, corpus, clip id, licence (the clip's own for FSD50K and OpenMIC-2018) and
the archive member.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters_nonspeech as ans  # noqa: E402
from noise_list import chime3_files, musan_files  # noqa: E402
from registry import AUTO, hash_split  # noqa: E402
from vad_dump import resample  # noqa: E402

FS = 16000


class PackWriter:
    """Concatenate clips into 16-bit wav packs of about pack_seconds each."""

    def __init__(self, out: Path, bank: str, source: str, pack_seconds: float):
        self.dir = out / bank
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bank, self.source = bank, source
        self.pack_samples = int(pack_seconds * FS)
        self.buffer, self.rows, self.files = [], [], []
        self.filled = 0
        self.total = 0

    def add(self, clip: str, x: np.ndarray, licence: str, member: str):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return
        self.rows.append([clip, self.filled, self.filled + len(x), licence, member])
        self.buffer.append(x)
        self.filled += len(x)
        self.total += len(x)
        if self.filled >= self.pack_samples:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        path = self.dir / f"{self.source}_{len(self.files):05d}.wav"
        x = np.concatenate(self.buffer)
        peak = float(np.abs(x).max())
        if peak > 1:
            x = x / peak
        soundfile.write(path, x, FS, subtype="PCM_16")
        self.files.append(
            (path, [[c, s / FS, e / FS, lic, m] for c, s, e, lic, m in self.rows])
        )
        self.buffer, self.rows, self.filled = [], [], 0

    def hours(self) -> float:
        return self.total / FS / 3600


class Collector:
    """The Writers interface of the readers: keeps the train-split clips only."""

    def __init__(self, pack: PackWriter, licence_of, max_hours=None):
        self.pack, self.licence_of, self.max_hours = pack, licence_of, max_hours
        self.skipped = 0

    def __getitem__(self, split: str):
        if split.startswith(AUTO):
            split = hash_split(split[len(AUTO) :])
        return _Split(self, split == "train")


class _Split:
    def __init__(self, collector: Collector, keep: bool):
        self.c, self.keep = collector, keep

    def skip(self, member, reason):
        logging.warning("skipped %s: %s", member, reason)

    def add(self, rec_id, signal, rate, channels=None, source="", **kwargs):
        c = self.c
        if not self.keep:
            c.skipped += 1
            return 0
        if c.max_hours is not None and c.pack.hours() >= c.max_hours:
            return 0
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, (channels or [0])[0]]
        c.pack.add(rec_id, resample(x, rate, FS), c.licence_of(rec_id), source)
        return 1


def fsd50k_licences(root: Path):
    """{clip id as the reader names it ("<dev|eval>_<fname>"): licence URL}."""
    md = ans.archive(root, "FSD50K.metadata.zip")
    out = {}
    for part in ("dev", "eval"):
        info = json.loads(
            ans.read_zip_member(md, f"FSD50K.metadata/{part}_clips_info_FSD50K.json")
        )
        for fname, meta in info.items():
            out[f"{part}_{fname}"] = meta.get("license", "")
    return out


def openmic_licences(root: Path):
    """{sample_key: licence} of openmic-2018-metadata.csv."""
    path = ans.archive(root, "openmic-2018-v1.0.0.tgz")
    for name, data in ans.iter_tar_select(
        path, lambda n: n.endswith("openmic-2018-metadata.csv") and not ans.is_junk(n)
    ):
        rows = ans.read_csv(data)
        lic = ans.find_column(rows[0], "licen")
        return {r["sample_key"]: r[lic] for r in rows}
    return {}


def is_music(labels):
    return "Music" in labels


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path, help="VAD_CORPORA")
    p.add_argument("out", type=Path)
    p.add_argument("--music_scp", type=Path, default=Path("data/music.scp"))
    p.add_argument("--noise_scp", type=Path, default=Path("data/noise_aug.scp"))
    p.add_argument("--pack_seconds", type=float, default=300.0)
    p.add_argument("--effects_pack_seconds", type=float, default=60.0)
    p.add_argument("--effects_hours", type=float, default=20.0)
    p.add_argument(
        "--sources",
        default="slakh2100,openmic2018,fsd50k_music,fsd50k_effects",
        help="comma list of the archive sources to read",
    )
    a = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sources = set(a.sources.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    packs = []

    if "slakh2100" in sources:
        pw = PackWriter(a.out, "music", "slakh2100", a.pack_seconds)
        ans.slakh2100(a.root / "slakh2100", Collector(pw, lambda r: "CC BY 4.0"), 0, 1)
        packs.append(pw)
    if "openmic2018" in sources:
        lic = openmic_licences(a.root / "openmic2018")
        pw = PackWriter(a.out, "music", "openmic2018", a.pack_seconds)
        ans.openmic2018(
            a.root / "openmic2018", Collector(pw, lambda r: lic.get(r, "")), 0, 1
        )
        packs.append(pw)
    fsd = a.root / "fsd50k"
    if sources & {"fsd50k_music", "fsd50k_effects"}:
        lic = fsd50k_licences(fsd)
    for name, bank, music, secs, cap in (
        ("fsd50k_music", "music", True, a.pack_seconds, None),
        ("fsd50k_effects", "effects", False, a.effects_pack_seconds, a.effects_hours),
    ):
        if name not in sources:
            continue
        pw = PackWriter(a.out, bank, name, secs)

        def select(labels, tags, music=music):
            return ans.fsd_kind(labels, tags) == "env" and is_music(labels) == music

        ans._fsd50k(
            fsd, Collector(pw, lambda r: lic.get(r, ""), max_hours=cap), 0, 1, select
        )
        packs.append(pw)
    for pw in packs:
        pw.flush()
        logging.info(
            "%s/%s: %d packs, %.2f h", pw.bank, pw.source, len(pw.files), pw.hours()
        )

    musan = list(musan_files(a.root / "musan"))
    chime3 = list(chime3_files(a.root / "chime3_background"))
    lists = {"music": [], "effects": []}
    rows = []
    for key, path in musan:
        bank = "music" if "music" in Path(path).parts else "effects"
        lists[bank].append((key, Path(path)))
        rows.append(
            [bank, Path(path), 0.0, "", "musan", key, "MUSAN per-file CC0/BY/PD", ""]
        )
    for key, path in chime3:
        lists["effects"].append((key, Path(path)))
        rows.append(
            [
                "effects",
                Path(path),
                0.0,
                "",
                "chime3_background",
                key,
                "CC BY 4.0",
                "",
            ]
        )
    for pw in packs:
        for path, clips in pw.files:
            lists[pw.bank].append((path.stem, path.resolve()))
            for clip, s, e, licence, member in clips:
                rows.append(
                    [
                        pw.bank,
                        path.resolve(),
                        f"{s:.3f}",
                        f"{e:.3f}",
                        pw.source,
                        clip,
                        licence,
                        member,
                    ]
                )
    for scp, bank in ((a.music_scp, "music"), (a.noise_scp, "effects")):
        scp.parent.mkdir(parents=True, exist_ok=True)
        with scp.open("w") as f:
            for key, path in sorted(lists[bank]):
                f.write(f"{key} {path}\n")
        logging.info("%s: %d entries", scp, len(lists[bank]))
    with (a.out / "licences.tsv").open("w") as f:
        f.write("bank\tfile\tstart_s\tend_s\tcorpus\tclip\tlicence\tsource\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


if __name__ == "__main__":
    main()
