"""Read audio and annotations straight from corpus archives and directories.

Nothing is unpacked to disk: tar members are streamed, zip and parquet
members are read by name, and RAR/7z members are piped through 7zz.
"""

import io
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".sph")
SEVENZ = os.environ.get("SEVENZ", "7zz")


def _match(name: str, suffixes: Optional[Sequence[str]]) -> bool:
    return suffixes is None or name.lower().endswith(tuple(suffixes))


def iter_files(
    root: Union[Path, str], suffixes: Optional[Sequence[str]] = AUDIO_SUFFIXES
) -> Iterator[Tuple[str, Path]]:
    """(path relative to root, path) of every matching file, sorted."""
    root = Path(root)
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            if _match(fn, suffixes):
                p = Path(dp) / fn
                yield str(p.relative_to(root)), p


def iter_tar(
    path: Union[Path, str], suffixes: Optional[Sequence[str]] = AUDIO_SUFFIXES
) -> Iterator[Tuple[str, bytes]]:
    """(member name, bytes) of every matching regular member, in archive order."""
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if m.isfile() and _match(m.name, suffixes):
                yield m.name, t.extractfile(m).read()


def iter_zip(
    path: Union[Path, str], suffixes: Optional[Sequence[str]] = AUDIO_SUFFIXES
) -> Iterator[Tuple[str, bytes]]:
    """(member name, bytes) of every matching member, sorted by name."""
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith("/") and _match(name, suffixes):
                yield name, z.read(name)


def zip_names(path: Union[Path, str]) -> list:
    """Member names of a zip archive."""
    with zipfile.ZipFile(path) as z:
        return z.namelist()


def read_zip_member(path: Union[Path, str], name: str) -> bytes:
    with zipfile.ZipFile(path) as z:
        return z.read(name)


def iter_parquet(
    path: Union[Path, str], columns: Optional[Sequence[str]] = None
) -> Iterator[dict]:
    """Rows of a parquet file as dicts, one row group at a time."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    for g in range(f.num_row_groups):
        for row in f.read_row_group(g, columns=columns).to_pylist():
            yield row


def list_7z(path: Union[Path, str]) -> list:
    """Member names of a RAR/7z archive (files only)."""
    out = subprocess.run(
        [SEVENZ, "l", "-slt", "-ba", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names, name, is_dir = [], None, False
    for line in out.splitlines() + [""]:
        if line.startswith("Path = "):
            name = line[len("Path = ") :]
        elif line.startswith("Attributes = "):
            is_dir = "D" in line.split("=", 1)[1].split()[0] if "=" in line else False
        elif line == "" and name is not None:
            if not is_dir:
                names.append(name)
            name, is_dir = None, False
    return names


def read_7z_member(path: Union[Path, str], name: str) -> bytes:
    """Bytes of one member of a RAR/7z archive."""
    return subprocess.run(
        [SEVENZ, "x", "-so", str(path), name], check=True, capture_output=True
    ).stdout


def ffmpeg_exe() -> Optional[str]:
    """$FFMPEG, else the static build of imageio-ffmpeg, else ffmpeg on PATH.

    A static build avoids a system ffmpeg whose shared libraries a compute
    node may lack.
    """
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def decode(data: Union[bytes, Path, str]) -> Tuple[np.ndarray, int]:
    """(samples x channels float32, rate) of an audio file or its bytes.

    libsndfile reads WAV/FLAC/OGG/MP3; anything else (m4a, aac, webm, opus in
    some containers) goes through ffmpeg (see ffmpeg_exe()).
    """
    src = io.BytesIO(data) if isinstance(data, bytes) else str(data)
    try:
        x, rate = soundfile.read(src, dtype="float32", always_2d=True)
        return x, rate
    except Exception:  # noqa: BLE001 - try ffmpeg for the formats libsndfile lacks
        if ffmpeg_exe() is None:
            raise
    cmd = [
        ffmpeg_exe(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        "pipe:0" if isinstance(data, bytes) else str(data),
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    out = subprocess.run(
        cmd,
        input=data if isinstance(data, bytes) else None,
        check=True,
        capture_output=True,
    ).stdout
    x, rate = soundfile.read(io.BytesIO(out), dtype="float32", always_2d=True)
    return x, rate
