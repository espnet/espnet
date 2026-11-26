"""Shared helpers of the crawled download-manifest generators (local/v2/*_manifest.py).

- Fetcher: sequential HTTP GET with a minimum interval between requests, a
  User-Agent that names the project and no person, retries with back-off, and
  Retry-After honoured on 429/503.
- licence_id: the SPDX-like id of a licence statement (URL or text).
- licence_accepted: whether an id may go into a manifest (CC0-1.0, PD, PDM-1.0,
  CC-BY-x.y; never NC, SA or ND).
- write_manifest / write_licences: the manifest (download.py format) and the
  per-item licence table (relative_path, licence, licence_url, creator,
  source_page) that the licence gate reads at list-building.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "ESPnet FRONTIA VAD data prep (https://github.com/espnet/espnet)"
MANIFEST_HEAD = "# corpus_dir\trelative_path\turl\tsize_bytes\tchecksum\tunpack\n"
LICENCE_HEAD = "relative_path\tlicence\tlicence_url\tcreator\tsource_page\n"
DOWNLOADS = Path(__file__).resolve().parent / "downloads"


class Fetcher:
    """Sequential GET with at least `interval` seconds between request starts."""

    def __init__(self, ua: str = UA, interval: float = 1.0, tries: int = 5):
        self.ua, self.interval, self.tries = ua, interval, tries
        self.last = 0.0
        self.count = 0

    def raw(self, url: str, data: bytes = None) -> bytes:
        """Return the body of url (a POST of data when data is given).

        Raises RuntimeError after `tries` failures.
        """
        for k in range(self.tries):
            wait = self.interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            self.count += 1
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"User-Agent": self.ua}
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404 or k == self.tries - 1:
                    raise RuntimeError(f"{url}: HTTP {e.code}") from e
                after = e.headers.get("Retry-After", "")
                time.sleep(int(after) if after.isdigit() else 2 ** (k + 1))
            except Exception as e:  # noqa: BLE001 - retried, then raised
                if k == self.tries - 1:
                    raise RuntimeError(f"{url}: {e}") from e
                time.sleep(2 ** (k + 1))
        raise RuntimeError(f"{url}: no response")

    def json(self, url: str, data: bytes = None):
        """Return url's body (POST of data when given) parsed as JSON."""
        return json.loads(self.raw(url, data))


def licence_id(text: str) -> str:
    """SPDX-like id of a licence URL or statement, or "" when it names none.

    CC licences: CC-BY[-NC][-SA][-ND][-<version>][-<port>], e.g. CC-BY-4.0,
    CC-BY-3.0-AT, CC-BY-SA-4.0, CC-BY (no version stated). A 4.0 URL with a
    locale (licenses/by/4.0/de/) is the 4.0 licence (4.0 has no ports).
    CC0-1.0 for CC0 or publicdomain/zero; PDM-1.0 for the Public Domain Mark
    (publicdomain/mark); PD for other public-domain statements.
    """
    t = re.sub(r"\s+", " ", (text or "").lower())
    if re.search(r"\bcc[ -]?0\b|cc[ -]?zero|publicdomain/zero", t):
        return "CC0-1.0"
    if re.search(r"publicdomain/mark|public domain mark|\bpdm\b", t):
        return "PDM-1.0"
    m = re.search(r"licenses/(by(?:-[a-z]+)*)/(\d\.\d)(?:/([a-z]{2})\b)?", t)
    if m:
        # 4.0 has no ported versions: a locale after it (by/4.0/de/) names a
        # translation of the same licence
        port = m.group(3) and m.group(2) != "4.0"
        port = f"-{m.group(3).upper()}" if port else ""
        return f"CC-{m.group(1).upper()}-{m.group(2)}{port}"
    if re.search(r"\bcc[ _-]?by\b|creative ?commons\W.{0,30}attribution", t):
        mods = ""
        if re.search(r"\bnc\b|non-?commercial", t):
            mods += "-NC"
        if re.search(r"\bsa\b|share-?alike|share alike", t):
            mods += "-SA"
        if re.search(r"\bnd\b|no-?deriv", t):
            mods += "-ND"
        v = re.search(r"(\d\.\d)(?:[ /-]*\b(at|de|ch|us|nl|fr|es|it)\b)?", t)
        ver = f"-{v.group(1)}" if v else ""
        port = f"-{v.group(2).upper()}" if v and v.group(2) else ""
        return f"CC-BY{mods}{ver}{port}"
    if re.search(r"public ?domain|licenses/publicdomain", t):
        return "PD"
    return ""


def licence_accepted(lid: str) -> bool:
    """True for CC0-1.0, PD, PDM-1.0 and CC-BY with no NC, SA or ND."""
    if lid in ("CC0-1.0", "PD", "PDM-1.0"):
        return True
    return bool(re.fullmatch(r"CC-BY(-\d\.\d)?(-[A-Z]{2})?", lid))


def clean(text: str) -> str:
    """text without HTML tags, tabs and newlines."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def write_manifest(out: Path, corpus: str, rows, comments=()):
    """Write rows (relative_path, url, size, checksum) as out/<corpus>.tsv."""
    path = out / f"{corpus}.tsv"
    with path.open("w") as o:
        o.write(MANIFEST_HEAD)
        for c in comments:
            o.write(f"# {clean(c)}\n")
        for rel, url, size, check in rows:
            o.write(f"{corpus}\t{rel}\t{url}\t{size}\t{check}\n")
    return path


def write_licences(out: Path, corpus: str, rows):
    """Write rows as out/<corpus>.licences.tsv.

    Each row: relative_path, licence, licence_url, creator, source_page.
    """
    path = out / f"{corpus}.licences.tsv"
    with path.open("w") as o:
        o.write(LICENCE_HEAD)
        for r in rows:
            o.write("\t".join(clean(str(x)) for x in r) + "\n")
    return path


def log(*a):
    print(*a, file=sys.stderr, flush=True)
