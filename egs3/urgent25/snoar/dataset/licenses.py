"""Per-file license audit of the URGENT 2025 noise pool (+ MUSAN noise).

Writes four files to ``out_dir``:

- ``noise_licenses.tsv``: one row per noise file with its license and, for
  attribution, author, title and URL where the metadata has them;
- ``noise_attribution.tsv``: the same rows for the allowed files only, the
  credit CC BY asks for when the data or a model trained on it is shared;
- ``commercial_allow.txt``: ``<subset>/<id>`` of the URGENT noise files whose
  license allows commercial use and derivatives without share-alike (CC0,
  CC BY, public domain), the ``noise_allow_list`` of
  :class:`~espnet3.systems.snoar.src.data.urgent.UrgentMixDataset`;
- ``musan_noise_commercial.txt``: the MUSAN noise files under the same rule,
  its ``extra_noise_list``.

Licenses come from the dataset metadata, not from the audio store:

- FSD50K: ``FSD50K.metadata.zip`` (Zenodo 4060432), ``*_clips_info_FSD50K.json``;
- DNS5 Freesound clips: by Freesound id, from FSD50K and the FSDKaggle2018
  metadata (``train_post_competition.csv``,
  ``test_post_competition_scoring_clips.csv``); a clip in neither is unknown;
- FMA: ``fma_metadata/raw_tracks.csv``;
- MUSAN: ``noise/free-sound`` is public domain, ``noise/sound-bible/LICENSE``
  lists each clip.

DNS5 AudioSet clips (YouTube) and every unknown license are excluded.

    python -m egs3.urgent25.snoar.dataset.licenses OUT_DIR \
        --urgent_root .../urgent2025 --fsd50k_meta FSD50K.metadata \
        --fsdkaggle_meta FSDKaggle2018.meta --fma_meta fma_metadata \
        --musan_noise .../musan/noise
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

COMMERCIAL = ("zero", "by", "publicdomain", "mark")
COLUMNS = ("subset", "id", "license", "source", "author", "title", "url")


def parse_cc_license(url) -> str:
    """Return the short code of a Creative Commons license URL.

    Args:
        url: License URL; anything but a string gives ``"unknown"``.

    Returns:
        The code, e.g. ``"by-nc"``, or ``"unknown"``.

    Examples:
        >>> parse_cc_license("http://creativecommons.org/licenses/by-nc/3.0/")
        'by-nc'
        >>> parse_cc_license(float("nan"))
        'unknown'
    """
    if not isinstance(url, str):
        return "unknown"
    m = re.search(r"creativecommons\.org/(?:licenses|publicdomain)/([a-z+-]+)", url)
    return m.group(1) if m else "unknown"


def _read_freesound_licenses(
    fsd50k_meta: Path, fsdkaggle_meta: Optional[Path]
) -> Dict[int, dict]:
    clips: Dict[int, dict] = {}
    for part in ("dev", "eval"):
        with open(fsd50k_meta / f"{part}_clips_info_FSD50K.json") as f:
            for k, v in json.load(f).items():
                clips[int(k)] = dict(
                    license=parse_cc_license(v.get("license")),
                    author=v.get("uploader") or "",
                    title=v.get("title") or "",
                )
    if fsdkaggle_meta is not None:
        names = {"Creative Commons 0": "zero", "Attribution": "by"}
        for name in (
            "train_post_competition.csv",
            "test_post_competition_scoring_clips.csv",
        ):
            with open(fsdkaggle_meta / name, newline="") as f:
                for row in csv.DictReader(f):
                    # anything but CC0 / CC BY (e.g. Sampling+) is not allowed
                    lic = names.get(row["license"], "by-nc")
                    clips.setdefault(
                        int(row["freesound_id"]),
                        dict(license=lic, author="", title=""),
                    )
    return clips


def _read_fma_licenses(fma_meta: Path) -> Dict[int, dict]:
    tracks = {}
    with open(fma_meta / "raw_tracks.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tracks[int(row["track_id"])] = dict(
                license=parse_cc_license(row.get("license_url")),
                author=row.get("artist_name") or "",
                title=row.get("track_title") or "",
                url=row.get("track_url") or "",
            )
    return tracks


def _read_subset_ids(root: Path, subset: str) -> List[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(root / subset / "metadata.parquet", columns=["id"])
    return table.column("id").to_pylist()


def _read_musan_licenses(musan_noise: Path) -> List[Tuple]:
    clips = {}
    text = (musan_noise / "sound-bible" / "LICENSE").read_text(errors="replace")
    for block in text.split("====="):
        name = re.search(r"(noise-sound-bible-\d+)", block)
        lic = re.search(r"License:\s*(.+)", block)
        if not (name and lic):
            continue
        get = lambda p: (re.search(p, block) or [None, ""])[1].strip()  # noqa
        clips[name.group(1)] = dict(
            license={"Attribution 3.0": "by", "Public Domain": "publicdomain"}.get(
                lic.group(1).strip(), "unknown"
            ),
            author=get(r"Recorded by\s*(.+)"),
            title=get(r"Title:\s*(.+)"),
            url=get(r"(https?://\S+)"),
        )
    rows = []
    for path in sorted(glob.glob(str(musan_noise / "*" / "*.wav"))):
        name = Path(path).stem
        if "/free-sound/" in path:
            meta = dict(license="publicdomain", author="", title="", url="")
        else:
            meta = clips.get(name, dict(license="unknown"))
        rows.append(
            (
                "musan_noise",
                path,
                meta["license"],
                "musan noise LICENSE",
                meta.get("author", ""),
                meta.get("title", ""),
                meta.get("url", ""),
            )
        )
    return rows


def audit_noise_licenses(
    urgent_root: Path,
    fsd50k_meta: Path,
    fma_meta: Path,
    fsdkaggle_meta: Optional[Path] = None,
    musan_noise: Optional[Path] = None,
) -> List[Tuple]:
    """Look up the license of every URGENT 2025 (and MUSAN) noise file.

    Args:
        urgent_root: OMNIIO store of URGENT 2025 (``<subset>/metadata.parquet``).
        fsd50k_meta: Extracted ``FSD50K.metadata`` directory.
        fma_meta: Extracted ``fma_metadata`` directory.
        fsdkaggle_meta: FSDKaggle2018 ``meta`` directory; without it, DNS5
            Freesound clips outside FSD50K are unknown (excluded).
        musan_noise: MUSAN ``noise`` directory, or ``None`` to skip MUSAN.

    Returns:
        One tuple of ``COLUMNS`` per file.

    Raises:
        FileNotFoundError: If a metadata file is missing.
    """
    fsd = _read_freesound_licenses(fsd50k_meta, fsdkaggle_meta)
    fma = _read_fma_licenses(fma_meta)
    none = dict(license="unknown", author="", title="")
    rows = []
    for i in _read_subset_ids(urgent_root, "noise_fsd50k"):
        n = int(i.replace("fsd50k_", ""))
        m = fsd.get(n, none)
        rows.append(
            (
                "noise_fsd50k",
                i,
                m["license"],
                "FSD50K clips_info",
                m["author"],
                m["title"],
                f"https://freesound.org/s/{n}/",
            )
        )
    for i in _read_subset_ids(urgent_root, "noise_fma_novocal"):
        m = fma.get(int(i.replace("fma_", "")), dict(none, url=""))
        rows.append(
            (
                "noise_fma_novocal",
                i,
                m["license"],
                "fma_metadata raw_tracks.csv",
                m["author"],
                m["title"],
                m["url"],
            )
        )
    for i in _read_subset_ids(urgent_root, "noise_dns5"):
        fs = re.search(r"Freesound_[a-z]+_(\d+)_", i)
        if fs is None:
            lic = "unknown" if "Freesound" in i else "audioset-youtube"
            rows.append(("noise_dns5", i, lic, "excluded", "", "", ""))
            continue
        n = int(fs.group(1))
        m = fsd.get(n, none)
        rows.append(
            (
                "noise_dns5",
                i,
                m["license"],
                "FSD50K/FSDKaggle2018 by freesound id",
                m["author"],
                m["title"],
                f"https://freesound.org/s/{n}/",
            )
        )
    if musan_noise is not None:
        rows += _read_musan_licenses(musan_noise)
    return rows


def write_license_record(rows: List[Tuple], out_dir: Path) -> Dict[str, int]:
    """Write the license record and the two commercial-use file lists.

    Args:
        rows: Output of :func:`audit_noise_licenses`.
        out_dir: Directory for ``noise_licenses.tsv``,
            ``noise_attribution.tsv``, ``commercial_allow.txt`` and
            ``musan_noise_commercial.txt``.

    Returns:
        Counts: ``files`` audited, ``allowed`` URGENT files, ``musan`` files.

    Raises:
        ValueError: If no file is allowed (wrong metadata for this store).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = lambda v: str(v).replace("\t", " ").replace("\n", " ")  # noqa
    with open(out_dir / "noise_licenses.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(clean(v) for v in r) + "\n")
    ok = [r for r in rows if r[2] in COMMERCIAL]
    if not ok:
        raise ValueError(
            "no noise file has a commercial-use license: check that the "
            "metadata directories match the URGENT 2025 store"
        )
    allow = [f"{r[0]}/{r[1]}" for r in ok if r[0] != "musan_noise"]
    musan = [r[1] for r in ok if r[0] == "musan_noise"]
    with open(out_dir / "noise_attribution.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in ok:
            # MUSAN rows name a local path; keep only the part under noise/
            name = "/".join(Path(r[1]).parts[-3:]) if r[0] == "musan_noise" else r[1]
            f.write("\t".join(clean(v) for v in (r[0], name) + r[2:]) + "\n")
    (out_dir / "commercial_allow.txt").write_text("\n".join(allow) + "\n")
    (out_dir / "musan_noise_commercial.txt").write_text("\n".join(musan) + "\n")
    return dict(files=len(rows), allowed=len(allow), musan=len(musan))


def main(argv=None):
    """Run the audit from the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", type=Path)
    p.add_argument("--urgent_root", type=Path, required=True)
    p.add_argument("--fsd50k_meta", type=Path, required=True)
    p.add_argument("--fma_meta", type=Path, required=True)
    p.add_argument("--fsdkaggle_meta", type=Path)
    p.add_argument("--musan_noise", type=Path)
    a = p.parse_args(argv)
    rows = audit_noise_licenses(
        a.urgent_root, a.fsd50k_meta, a.fma_meta, a.fsdkaggle_meta, a.musan_noise
    )
    print(write_license_record(rows, a.out_dir))


if __name__ == "__main__":
    main()
