#!/usr/bin/env python3
"""Download the corpora listed in local/v2/downloads/<corpus>.tsv and verify them.

    python3 local/v2/download.py --root ${VAD_CORPORA} [--corpora a,b] [--jobs 8]

Each manifest line: corpus_dir, relative_path, url, size_bytes, checksum
(md5:<hex>, sha256:<hex> or empty), unpack (empty, "untar": unpack the archive
into the corpus directory, "untar_strip1": the same without the archive's top
directory). A file that already has the listed size (and checksum) is kept,
so a rerun only fetches what is missing. Downloads resume (wget -c). A line
without a size takes the server's Content-Length (HEAD request) as the size;
when the server gives none, the file counts only after wget succeeds. An empty
file is always BAD, and a downloaded .zip must pass a CRC test of every member.
Every file is reported as OK or BAD; the exit status is 1 if any file is BAD. The
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
# Wikimedia and some hosts refuse clients without a descriptive User-Agent
USER_AGENT = "ESPnet-FRONTIA-VAD-dataprep/0.1 (https://github.com/espnet/espnet)"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 << 20), b""):
            h.update(block)
    return h.hexdigest()


def complete(dest: Path, size: str) -> bool:
    """dest has the expected size (a file whose size is not known is never complete)."""
    return bool(size) and dest.exists() and dest.stat().st_size == int(size)


def served_size(url: str) -> str:
    """The Content-Length of the final response to a HEAD request, or "" if none."""
    r = subprocess.run(
        ["curl", "-sIL", "--max-time", "120", "-A", USER_AGENT, url],
        capture_output=True,
        text=True,
    )
    sizes = [
        line.split(":", 1)[1].strip()
        for line in r.stdout.splitlines()
        if line.lower().startswith("content-length:")
    ]
    return sizes[-1] if sizes and sizes[-1].isdigit() and int(sizes[-1]) > 0 else ""


# seconds to wait before each request (--delay), for hosts serving many small files
DELAY = 0.0


def fetch(row, root: Path) -> str:
    corpus, rel, url, size, check, unpack = row
    dest = root / corpus / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Without a listed size, the server's Content-Length is the size to reach. A failed
    # wget -O leaves an empty file behind, which must not count as downloaded.
    time.sleep(DELAY)
    size = size or served_size(url)
    present = complete(dest, size)
    fetched = False
    for attempt in range(3):
        if complete(dest, size):
            break
        time.sleep(DELAY)
        r = subprocess.run(
            [
                "wget",
                "-q",
                "-c",
                "--tries=10",
                "--waitretry=30",
                f"--user-agent={USER_AGENT}",
                "-O",
                str(dest),
                url,
            ],
            capture_output=True,
            text=True,
        )
        fetched = r.returncode == 0
        if fetched and (not size or complete(dest, size)):
            break
        log(f"retry {attempt + 1} {url}: rc={r.returncode} {r.stderr.strip()[-200:]}")
        time.sleep(30)
    if not dest.exists() or dest.stat().st_size == 0:
        return f"BAD empty {dest}"
    if size and not complete(dest, size):
        return f"BAD size {dest} ({dest.stat().st_size} of {size} bytes)"
    if not size and not present and not fetched:
        return f"BAD download {dest} (no size known and wget failed)"
    if dest.suffix == ".zip" and not present:
        import zipfile

        try:
            with zipfile.ZipFile(dest) as z:
                bad = z.testzip()
        except zipfile.BadZipFile as e:
            return f"BAD zip {dest}: {e}"
        if bad is not None:
            return f"BAD zip {dest}: CRC error in {bad}"
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


def safe_fetch(row, root: Path) -> str:
    """fetch(), reporting an error as BAD for that file instead of stopping.

    E.g. a corpus directory in the shared root that is owned by another user.
    """
    try:
        return fetch(row, root)
    except Exception as e:  # noqa: BLE001
        return f"BAD {type(e).__name__}: {e} ({row[0]}/{row[1]})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="corpora root")
    ap.add_argument("--corpora", default="", help="comma separated corpora to fetch")
    ap.add_argument(
        "--all", action="store_true", help="fetch every manifest (incl. eval_*)"
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument(
        "--delay", type=float, default=0.0, help="seconds to wait before each request"
    )
    a = ap.parse_args()
    only = set(filter(None, a.corpora.split(",")))
    if not only and not a.all:
        ap.error("no --corpora given; pass --all to fetch every manifest")
    rows = []
    for m in sorted(MANIFESTS.glob("*.tsv")):
        # <corpus>.licences.tsv (per-file licences for the licence gate) and other
        # <corpus>.<table>.tsv side tables (e.g. tatoeba_audio.text.tsv) are not
        # downloads
        if "." in m.stem or (only and m.stem not in only):
            continue
        for line in m.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                rows.append((line.split("\t") + [""] * 6)[:6])
    rows.sort(key=lambda r: -int(r[3] or 0))  # biggest first
    log(f"{len(rows)} files, {sum(int(r[3] or 0) for r in rows) / 1e9:.1f} GB")
    bad = 0
    global DELAY
    DELAY = a.delay
    with ThreadPoolExecutor(a.jobs) as ex:
        for res in ex.map(lambda r: safe_fetch(r, a.root), rows):
            log(res)
            bad += res.startswith("BAD")
    log(f"done, {bad} BAD")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
