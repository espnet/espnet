#!/usr/bin/env python3
"""Download the corpora listed in local/v2/downloads/<corpus>.tsv and verify them.

    python3 local/v2/download.py --root ${VAD_CORPORA} [--corpora a,b] [--jobs 8]

Each manifest line: corpus_dir, relative_path, url, size_bytes, checksum
(md5:<hex>, sha256:<hex> or empty), unpack (empty, "untar": unpack the archive
into the corpus directory, "untar_strip1": the same without the archive's top
directory). A file that already has the listed size (and checksum) is kept,
so a rerun only fetches what is missing. Downloads resume (wget -c). Every
file is reported as OK or BAD; the exit status is 1 if any file is BAD. The
checksum is verified after a download; a file already there is checked by
size only.

Corpora that need an account or an agreement have no manifest: see the
README (Common Voice: local/v2/download_commonvoice.py; OpenYAP: the
vendor's data use agreement).
"""

import argparse
import hashlib
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFESTS = Path(__file__).resolve().parent / "downloads"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 << 20), b""):
            h.update(block)
    return h.hexdigest()


def complete(dest: Path, size: str) -> bool:
    return dest.exists() and (not size or dest.stat().st_size == int(size))


def fetch(row, root: Path) -> str:
    corpus, rel, url, size, check, unpack = row
    dest = root / corpus / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    present = complete(dest, size)
    for attempt in range(3):
        if complete(dest, size):
            break
        r = subprocess.run(
            ["wget", "-q", "-c", "--tries=10", "--waitretry=30", "-O", str(dest), url],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and not size:
            break
        log(f"retry {attempt + 1} {url}: rc={r.returncode} {r.stderr.strip()[-200:]}")
        time.sleep(30)
    if not complete(dest, size):
        return f"BAD size {dest}"
    algo, _, want = check.partition(":")
    # a file that was already there was checked when it was downloaded
    if not present and algo in ("md5", "sha256") and digest(dest, algo) != want:
        return f"BAD {algo} {dest}"
    if unpack:
        done = dest.with_name(dest.name + ".unpacked")
        if not done.exists():
            with tarfile.open(dest) as t:
                members = t.getmembers()
                if unpack == "untar_strip1":
                    for m in members:
                        m.name = m.name.split("/", 1)[1] if "/" in m.name else ""
                    members = [m for m in members if m.name]
                t.extractall(root / corpus, members=members, filter="data")
            done.write_text("")
    return f"OK {dest}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="corpora root")
    ap.add_argument("--corpora", default="", help="comma separated; default all")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    only = set(filter(None, a.corpora.split(",")))
    rows = []
    for m in sorted(MANIFESTS.glob("*.tsv")):
        if only and m.stem not in only:
            continue
        for line in m.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                rows.append((line.split("\t") + [""] * 6)[:6])
    rows.sort(key=lambda r: -int(r[3] or 0))  # biggest first
    log(f"{len(rows)} files, {sum(int(r[3] or 0) for r in rows) / 1e9:.1f} GB")
    bad = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for res in ex.map(lambda r: fetch(r, a.root), rows):
            log(res)
            bad += res.startswith("BAD")
    log(f"done, {bad} BAD")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
