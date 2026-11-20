"""A read-only, seekable file over HTTP range requests (zipfile on a remote archive)."""

import io
import time
import urllib.request


class HttpRangeFile(io.RawIOBase):
    """Seekable file read by HTTP Range requests; the server must accept ranges."""

    def __init__(self, url: str, retries: int = 5):
        self.url, self.retries, self.pos = url, retries, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.headers.get("Accept-Ranges", "") != "bytes":
                raise OSError(f"{url}: no byte ranges")
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        rng = f"bytes={self.pos}-{self.pos + n - 1}"
        for k in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": rng})
                with urllib.request.urlopen(req, timeout=300) as r:
                    if r.status != 206:
                        raise OSError(f"range request answered {r.status}")
                    data = r.read()
                if len(data) == n:
                    self.pos += n
                    return data
                raise OSError(f"short read {len(data)} of {n}")
            except OSError:
                if k == self.retries - 1:
                    raise
                time.sleep(2**k)
        return b""

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
