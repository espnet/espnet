#!/usr/bin/env python3
"""Fetch the accepted FMA tracks out of fma_full.zip by range requests, into tar shards.

    python3 local/v2/fma_fetch.py <fma_select.tsv> <out dir> [--url URL] \
        [--shard_gb 4] \
        [--threads 8]

Only the members of the rows with accepted=1 are read; zipfile checks each one's CRC32
against the central directory, and the job stops on a mismatch. The MP3 bytes are
written unchanged, under their path in fma_full.zip, into tar shards of about
--shard_gb GB (fma_full_accepted_<k>.tar). <out dir>/fma_full_accepted.tsv lists every
track: track_id, member, shard, size_bytes, crc32.
"""

import argparse
import csv
import io
import sys
import tarfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from http_range import HttpRangeFile  # noqa: E402

URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
_local = threading.local()


def zip_for(url):
    if not hasattr(_local, "z"):
        _local.z = zipfile.ZipFile(HttpRangeFile(url))
    return _local.z


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("select", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--url", default=URL)
    p.add_argument("--shard_gb", type=float, default=4.0)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    rows = [
        r
        for r in csv.DictReader(open(a.select), delimiter="\t")
        if r["accepted"] == "1"
    ]
    print(f"{len(rows)} accepted tracks", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)

    def fetch(r):
        z = zip_for(a.url)
        info = z.getinfo(r["member"])
        data = z.read(info)  # raises BadZipFile on a CRC mismatch
        return r, info, data

    shard, size, tar = 0, 0, None
    limit = a.shard_gb * 1e9
    with (
        open(a.out / "fma_full_accepted.tsv", "w") as listing,
        ThreadPoolExecutor(a.threads) as ex,
    ):
        listing.write("track_id\tmember\tshard\tsize_bytes\tcrc32\n")
        for i, (r, info, data) in enumerate(ex.map(fetch, rows)):
            if tar is None or size >= limit:
                if tar is not None:
                    tar.close()
                    shard += 1
                name = f"fma_full_accepted_{shard:03d}.tar"
                tar = tarfile.open(a.out / name, "w")
                size = 0
            ti = tarfile.TarInfo(info.filename)
            ti.size = len(data)
            ti.mtime = 0
            tar.addfile(ti, io.BytesIO(data))
            size += len(data)
            listing.write(
                f"{r['track_id']}\t{info.filename}\t{name}\t{len(data)}\t"
                f"{info.CRC:08x}\n"
            )
            if i % 1000 == 0:
                print(f"  {i}/{len(rows)} tracks, shard {shard}", flush=True)
        if tar is not None:
            tar.close()
    print(f"done: {len(rows)} tracks in {shard + 1} shards")


if __name__ == "__main__":
    main()
