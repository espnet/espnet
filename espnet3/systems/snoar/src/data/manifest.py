"""JSONL manifests and audio crops used by the SNOAR datasets.

One manifest per split, one JSON object per line, so a corpus of any size is
a single file (no per-utterance files on shared storage):

* recordings: ``{"id", "audio", "num_samples", "speech_ref"?, "noise_ref"?}``
  where the references are optional and used only for evaluation;
* noise pool: ``{"audio", "start", "end", "group"?}`` in samples, one row
  per noise-only region (a whole file, or a VAD gap of a recording).

``audio`` is a file path, or ``<blob>@<start_byte>:<end_byte>`` for one encoded
file stored at that byte range of a larger blob (OMNIIO-style audio stores),
so a corpus can be read without extracting it to per-utterance files.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".opus")
_BLOB_RANGE = re.compile(r"^(?P<blob>.+)@(?P<start>\d+):(?P<end>\d+)$")


def blob_path(blob: str | Path, start_byte: int, end_byte: int) -> str:
    """Return the ``audio`` value of a file stored at a byte range of a blob."""
    return f"{blob}@{int(start_byte)}:{int(end_byte)}"


def _audio_source(path: str):
    """Return what ``soundfile`` should open for an ``audio`` value."""
    m = _BLOB_RANGE.match(str(path))
    if m is None or Path(path).exists():
        return path
    start, end = int(m["start"]), int(m["end"])
    with open(m["blob"], "rb") as f:
        f.seek(start)
        return io.BytesIO(f.read(end - start))


def read_manifest(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL manifest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the manifest is empty or a row lacks ``audio``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "audio" not in row:
                raise ValueError(f"{path}:{lineno}: row has no `audio` field")
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def write_manifest(rows: Iterable[Dict[str, Any]], path: str | Path) -> int:
    """Write ``rows`` as JSONL atomically and return how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    tmp.replace(path)
    return n


def list_audio(source: str | Path) -> List[Path]:
    """List the audio files of a directory (recursively), a list file or a file.

    A list file is a text file with one path per line (``wav.scp`` style
    ``<id> <path>`` lines are accepted too).

    Raises:
        FileNotFoundError: If ``source`` does not exist or holds no audio.
    """
    source = Path(source)
    if source.is_dir():
        files = sorted(
            p for p in source.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES
        )
    elif source.is_file() and source.suffix.lower() in AUDIO_SUFFIXES:
        files = [source]
    elif source.is_file():
        files = []
        with open(source, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    files.append(Path(parts[-1]))
    else:
        raise FileNotFoundError(f"no such file or directory: {source}")
    if not files:
        raise FileNotFoundError(
            f"no audio files ({', '.join(AUDIO_SUFFIXES)}) in {source}"
        )
    return files


def describe_recordings(
    files: Iterable[Path], ids: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    """Return recording manifest rows (id, path, length, rate) for ``files``."""
    rows = []
    ids = list(ids) if ids is not None else None
    for i, p in enumerate(files):
        info = sf.info(str(p))
        rows.append(
            {
                "id": ids[i] if ids else Path(p).stem,
                "audio": str(Path(p).resolve()),
                "num_samples": int(info.frames),
                "sample_rate": int(info.samplerate),
            }
        )
    return rows


def read_audio(
    path: str, start: int = 0, stop: Optional[int] = None, fs: Optional[int] = None
) -> np.ndarray:
    """Read samples ``[start, stop)`` of channel 0 as float32.

    Raises:
        ValueError: If the file's rate differs from ``fs``.
    """
    x, rate = sf.read(
        _audio_source(path), start=start, stop=stop, dtype="float32", always_2d=True
    )
    if fs is not None and rate != fs:
        raise ValueError(
            f"{path} is {rate} Hz, the recipe expects {fs} Hz; resample the "
            "corpus first (SNOAR does not resample on the fly)"
        )
    return x[:, 0]


def fit_length(x: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad or truncate ``x`` to ``length`` samples."""
    if len(x) >= length:
        return x[:length]
    return np.pad(x, (0, length - len(x)))


def crossfade_concat(pieces: List[np.ndarray], length: int, fade: int) -> np.ndarray:
    """Concatenate ``pieces`` with raised-cosine crossfades up to ``length``."""
    out = np.zeros(length, dtype=np.float32)
    pos = 0
    for p in pieces:
        if pos >= length:
            break
        p = np.asarray(p, dtype=np.float32).copy()
        f = min(fade, len(p) // 2, pos // 2)
        if f > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(f) + 0.5) / f)).astype(
                np.float32
            )
            p[:f] *= ramp
            out[pos - f : pos] *= ramp[::-1]
            pos -= f
        end = min(length, pos + len(p))
        out[pos:end] += p[: end - pos]
        pos = end
    return out
