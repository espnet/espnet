#!/usr/bin/env python3
"""Download Common Voice (Scripted Speech 27.0, Spontaneous Speech 5.0) from MDC.

Mozilla Data Collective allows 7 dataset downloads per organisation per day;
the datasets are fetched largest first, one at a time. HTTP 403 means the
dataset's terms are not accepted yet: it is logged as NOTACCEPTED and
skipped. HTTP 429 means today's quota is used up: sleep until the reset
(Retry-After header, else 00:05 UTC). The download URL of a dataset is reused
while a transfer resumes, so a resume does not count as a new download. Each
archive is checked against the catalogue sha256. Runs until all datasets are
done; a rerun skips completed archives.

    python3 local/v2/download_commonvoice.py ${VAD_CORPORA}/commonvoice/ARCHIVES

The API key of a Mozilla Data Collective account is read from MDC_API_KEY;
the terms of each dataset must first be accepted on its page with that
account (a dataset whose terms are not accepted is logged as NOTACCEPTED).
"""

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://mozilladatacollective.com/api"
UA = {
    "User-Agent": "frontia-vad-data/1.0 (python urllib)",
    "Accept": "application/json",
}
PATTERN = re.compile(
    r"^Common Voice (Scripted Speech 27\.0|Spontaneous Speech 5\.0) - "
)


def log(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)


def catalogue():
    items, page = {}, 1
    while page <= 40:
        req = urllib.request.Request(
            f"{API}/datasets?page={page}&limit=100", headers=UA
        )
        with urllib.request.urlopen(req) as r:
            got = json.load(r).get("items", [])
        for d in got:
            if (d.get("organization") or {}).get(
                "name"
            ) == "Mozilla Common Voice" and PATTERN.match(d["name"]):
                items[d["id"]] = d
        if len(got) < 100:
            break
        page += 1
    return sorted(items.values(), key=lambda d: -int(d["sizeBytes"]))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seconds_to_reset(err):
    retry = err.headers.get("Retry-After") if err.headers else None
    if retry and retry.isdigit():
        return int(retry) + 60
    now = datetime.datetime.now(datetime.timezone.utc)
    reset = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=5, second=0, microsecond=0
    )
    return int((reset - now).total_seconds())


def download_url(dataset_id, key):
    req = urllib.request.Request(
        f"{API}/datasets/{dataset_id}/download",
        method="POST",
        headers={
            **UA,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["downloadUrl"]


def main():
    out = sys.argv[1]
    key = os.environ.get("MDC_API_KEY")
    if not key:
        sys.exit("MDC_API_KEY is not set")
    os.makedirs(out, exist_ok=True)
    items = catalogue()
    log(
        f"{len(items)} datasets, {sum(int(d['sizeBytes']) for d in items) / 1e9:.1f} GB"
    )
    for d in items:
        dest = os.path.join(out, d["filename"])
        size = int(d["sizeBytes"])
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            continue  # complete (sha256 checked when it was downloaded)
        url = None
        for attempt in range(6):
            if url is None:
                try:
                    url = download_url(d["id"], key)
                except urllib.error.HTTPError as e:
                    if e.code == 403:
                        page = f"https://mozilladatacollective.com/datasets/{d['id']}"
                        log(f"NOTACCEPTED {d['name']} {page}")
                        break
                    if e.code == 429:
                        wait = seconds_to_reset(e)
                        h, m = wait // 3600, wait % 3600 // 60
                        log(f"quota used up; sleeping {h}h{m:02d}m before {d['name']}")
                        time.sleep(wait)
                        continue
                    log(f"HTTP {e.code} for {d['name']}; retry in 5 min")
                    time.sleep(300)
                    continue
            subprocess.run(
                ["wget", "-q", "-c", "--tries=5", "--waitretry=30", "-O", dest, url],
                check=False,
            )
            if os.path.exists(dest) and os.path.getsize(dest) == size:
                break
            url = None  # URL may have expired: ask for a new one
        if not (os.path.exists(dest) and os.path.getsize(dest) == size):
            if os.path.exists(dest) and os.path.getsize(dest) == 0:
                os.remove(dest)
            continue
        have = sha256(dest)
        log(f"{'OK' if have == d['checksum'] else 'BAD'} sha256 {size} {d['name']}")
    log("done")


if __name__ == "__main__":
    main()
