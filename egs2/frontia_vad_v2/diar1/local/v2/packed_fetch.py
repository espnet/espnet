#!/usr/bin/env python3
"""Fetch the media of the accepted rows of a music_crawl.py manifest into tar shards.

    python3 local/v2/packed_fetch.py <manifest.tsv> <archive dir> <manifest_out.tsv> \
        [--shard_gb 2] [--rate 1.0]

Each accepted row's media_url is downloaded (at most --rate requests per second per
host, retried with back-off, a User-Agent that names no person) and written unchanged
into <archive dir>/<source>_<k>.tar as <source>/<item>.<ext>, without extracting
anything to disk. <manifest_out.tsv> is the input manifest with fetch_status (ok or the
error), http_status, bytes, sha256, shard and member added; rows that are not accepted
are copied with fetch_status "not fetched". A failed fetch does not stop the job; it is
recorded.
"""

import argparse
import csv
import hashlib
import io
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "frontia-vad-data-collector/1.0 (research dataset licence audit)"


class Polite:
    """At most `rate` requests per second per host."""

    def __init__(self, rate):
        self.gap, self.last = 1.0 / rate, {}

    def get(self, url, tries=4):
        host = urllib.parse.urlsplit(url).netloc
        for k in range(tries):
            wait = self.gap - (time.time() - self.last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self.last[host] = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                if e.code in (403, 404, 410, 451) or k == tries - 1:
                    return e.code, None
            except (
                Exception
            ):  # noqa: BLE001 - network errors are retried, then recorded
                if k == tries - 1:
                    return 0, None
            time.sleep(2 ** (k + 1))
        return 0, None


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:150] or "item"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("manifest", type=Path)
    p.add_argument("archive_dir", type=Path)
    p.add_argument("manifest_out", type=Path)
    p.add_argument("--shard_gb", type=float, default=2.0)
    p.add_argument("--rate", type=float, default=1.0)
    a = p.parse_args()
    a.archive_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.manifest.open(), delimiter="\t"))
    http = Polite(a.rate)
    extra = ["fetch_status", "http_status", "bytes", "sha256", "shard", "member"]
    tar, shard, size, ok = None, -1, 0, 0
    limit = a.shard_gb * 1e9
    with a.manifest_out.open("w", newline="") as fo:
        w = csv.DictWriter(
            fo, fieldnames=list(rows[0]) + extra, delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        for i, r in enumerate(rows):
            r.update({k: "" for k in extra})
            if r["accepted"] != "1" or not r["media_url"]:
                r["fetch_status"] = "not fetched"
                w.writerow(r)
                continue
            status, data = http.get(r["media_url"])
            r["http_status"] = status
            if data is None:
                r["fetch_status"] = f"error {status}"
                w.writerow(r)
                continue
            if tar is None or size >= limit:
                if tar is not None:
                    tar.close()
                shard += 1
                name = f"{r['source']}_{shard:04d}.tar"
                tar = tarfile.open(a.archive_dir / name, "w")
                size = 0
            ext = (
                Path(urllib.parse.urlsplit(r["media_url"]).path).suffix.lower()
                or ".bin"
            )
            member = f"{r['source']}/{safe(str(r['item_id']))}{ext}"
            ti = tarfile.TarInfo(member)
            ti.size, ti.mtime = len(data), 0
            tar.addfile(ti, io.BytesIO(data))
            size += len(data)
            r.update(
                fetch_status="ok",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                shard=name,
                member=member,
            )
            ok += 1
            w.writerow(r)
            if i % 200 == 0:
                print(f"  {i}/{len(rows)} rows, {ok} fetched", flush=True)
        if tar is not None:
            tar.close()
    print(f"fetched {ok} files into {shard + 1} shards")


if __name__ == "__main__":
    main()
