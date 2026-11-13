"""Audio stored in Omni-IO archives, addressed through a Kaldi-style scp.

An Omni-IO archive keeps compressed audio (FLAC by default) back to back in
large ``blob_<i>.bin`` files and indexes it in ``metadata.parquet``, the layout
of ``omniio.blob.blob.Blob``. Packing utterances into a few large files avoids
the per-file metadata load that millions of small audio files put on a shared
file system.

ESPnet keeps its usual ``wav.scp`` next to the archive, one line per
utterance pointing at the utterance's byte range::

    utt_id /abs/path/blob_0.bin:<start_byte>:<end_byte>

so that every data-directory tool that works on ``wav.scp`` (subsetting,
fix_data_dir.sh, filter_scp.pl, ...) keeps working, and loading an utterance
is one positional read from a bin file the reader keeps open.

A Kaldi ``segments`` file over such a ``wav.scp`` (``OmniIOSegmentsReader``,
data type ``omniio_segments``) reads parts of long recordings: for seekable
codecs only the part is read and decoded, so a 10 s segment of an hour-long
recording costs milliseconds instead of decoding the recording.
"""

import collections
import collections.abc
import io
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.segments import seconds_to_sample

_EBML = b"\x1aE\xdf\xa3"  # WebM/Opus header: not seekable here


class _ByteRange(io.RawIOBase):
    """Seekable read-only view of bytes [start, end) of an open descriptor.

    soundfile reads through it, so seeking inside a FLAC entry fetches only
    the frames it needs.
    """

    def __init__(self, fd: int, start: int, end: int):
        self.fd, self.start, self.size, self.pos = fd, start, end - start, 0

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

    def readinto(self, buf):
        n = max(0, min(len(buf), self.size - self.pos))
        data = os.pread(self.fd, n, self.start + self.pos)
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def segment_bounds(start: float, end: float, rate: int, frames: int) -> Tuple[int, int]:
    """Sample range of a Kaldi segment, as SegmentsExtractor cuts it.

    ``seconds_to_sample(start):seconds_to_sample(end)``, with ``end == -1``
    meaning the end of the recording.
    """
    first = seconds_to_sample(start, rate)
    last = frames if end == -1 else seconds_to_sample(end, rate)
    if not 0 <= first < last <= frames:
        raise RuntimeError(
            f"segment {start}-{end} s is samples {first}-{last}, "
            f"outside a recording of {frames} samples"
        )
    return first, last


def _segment_times(start: str, end: str, where: str) -> Tuple[float, float]:
    """Parse and check one segment's times (seconds; end -1: to the end).

    Refused: non-numbers, NaN/inf, a negative start (int() would truncate
    -0.00001 s to sample 0), and an end at or before the start.
    """
    try:
        st, et = float(start), float(end)
    except ValueError:
        raise RuntimeError(f"{where}: times must be numbers: {start} {end}")
    if not (math.isfinite(st) and math.isfinite(et)):
        raise RuntimeError(f"{where}: times must be finite: {start} {end}")
    if st < 0:
        raise RuntimeError(f"{where}: negative start time {start}")
    if et != -1 and et <= st:
        raise RuntimeError(f"{where}: end {end} is not after start {start}")
    return st, et


def parse_omniio_entry(entry: str) -> Tuple[str, int, int]:
    """Split ``<bin path>:<start_byte>:<end_byte>`` (the path may contain ':')."""
    path, start, end = entry.rsplit(":", 2)
    return path, int(start), int(end)


class OmniIOEntryReader:
    """Read single ``<bin path>:<start_byte>:<end_byte>`` entries.

    Examples:
        >>> read = OmniIOEntryReader()
        >>> rate, array = read('/data/format.1/wav/blob_0.bin:0:31052')

    Like ``kaldi_ark`` with ``max_cache_fd``, the bin files are kept open
    (up to ``max_cache_fd`` of them, least recently used closed first), so
    reading an entry is a single positional read rather than an
    open/seek/read/close: every open is a metadata request, which is what is
    slow on shared file systems. Reads use ``os.pread``, which does not move a
    shared file offset.

    The descriptors belong to the process that opened them: a pickled copy
    (e.g. a dataset sent to spawned DataLoader workers) starts with an empty
    cache, and a forked process opens its own instead of using, or closing,
    the inherited ones.

    FLAC, WAV and OGG entries are decoded here exactly as
    ``omniio.interface.audio_read`` decodes them (soundfile, float32,
    (frames, channels)); WebM/Opus entries are handed to ``audio_read``.

    Args:
        always_2d: Keep the channel axis even for single-channel audio.
        max_cache_fd: Bin files kept open; 0 opens and closes per read.
    """

    @typechecked
    def __init__(self, always_2d: bool = False, max_cache_fd: int = 32):
        """Set the output shape and how many bin files are kept open."""
        self.always_2d = always_2d
        self.max_cache_fd = max_cache_fd
        self._fds: "collections.OrderedDict[str, int]" = collections.OrderedDict()
        self._pid = os.getpid()

    def __getstate__(self):
        """Pickle without the descriptors: their numbers mean nothing elsewhere."""
        state = self.__dict__.copy()
        state["_fds"] = collections.OrderedDict()
        return state

    def _own_fds(self) -> "collections.OrderedDict[str, int]":
        """The cache of this process; forget descriptors inherited through fork."""
        if self._pid != os.getpid():
            # inherited copies of the parent's descriptors: leave them alone
            # (closing them here is harmless but they may be shared with other
            # objects of this process created before the fork)
            self._fds = collections.OrderedDict()
            self._pid = os.getpid()
        return self._fds

    def _fd(self, path: str) -> int:
        """Descriptor of `path` from the cache, opening it if needed."""
        fds = self._own_fds()
        fd = fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            fds[path] = fd
            # keep at least this one open while it is used; with max_cache_fd=0
            # it is closed after the read
            while len(fds) > max(self.max_cache_fd, 1):
                os.close(fds.popitem(last=False)[1])
        else:
            fds.move_to_end(path)
        return fd

    def _read_bytes(self, path: str, start: int, size: int) -> bytes:
        """Read `size` bytes at `start` of `path`, through the descriptor cache."""
        if self.max_cache_fd <= 0 or not hasattr(os, "pread"):
            with open(path, "rb") as f:
                f.seek(start)
                return f.read(size)
        data = os.pread(self._fd(path), size, start)
        if len(data) != size:
            raise RuntimeError(f"short read from {path}: {len(data)} of {size} bytes")
        return data

    def __call__(self, entry: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Return (sampling rate, signal) of `entry`, or (None, None) for "None"."""
        import soundfile

        if entry == "None":
            # an utterance without audio, as format_wav_scp.py writes it
            return None, None
        path, start, end = parse_omniio_entry(entry)
        blob = self._read_bytes(path, start, end - start)
        if blob[:4] == _EBML:
            from omniio.interface import audio_read

            result = audio_read(path, start_offset=start, file_size=end - start)
            array, rate = result.array, result.sample_rate
        else:
            array, rate = soundfile.read(
                io.BytesIO(blob), dtype="float32", always_2d=True
            )  # (frames, channels)
        if not self.always_2d and array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        return int(rate), array

    def read_segment(
        self, entry: str, start: float, end: float
    ) -> Tuple[int, np.ndarray]:
        """Return (rate, signal) of seconds `start`..`end` of `entry`.

        Cut as ``SegmentsExtractor`` cuts a Kaldi segment (``end == -1``: to
        the end). FLAC/WAV/OGG are seeked: only the segment is read and
        decoded. WebM is decoded whole and sliced.
        """
        try:
            return self._read_segment(entry, start, end)
        finally:
            if self.max_cache_fd <= 0:
                self.close()  # no cache: never keep a descriptor past the read

    def _read_segment(self, entry: str, start: float, end: float):
        import soundfile

        path, first_byte, end_byte = parse_omniio_entry(entry)
        if hasattr(os, "pread") and os.pread(self._fd(path), 4, first_byte) != _EBML:
            src = _ByteRange(self._fd(path), first_byte, end_byte)
            with soundfile.SoundFile(src) as f:
                rate = f.samplerate
                first, last = segment_bounds(start, end, rate, f.frames)
                f.seek(first)
                array = f.read(last - first, dtype="float32", always_2d=True)
            if len(array) != last - first:
                raise RuntimeError(f"short read of segment {start}-{end} s of {entry}")
            if not self.always_2d and array.shape[1] == 1:
                array = array[:, 0]
            return int(rate), array
        rate, array = self(entry)
        first, last = segment_bounds(start, end, rate, len(array))
        return rate, array[first:last]

    def read_frames(self, entry: str, first: int, last: int) -> Tuple[int, np.ndarray]:
        """Return (rate, samples `first`..`last` - 1) of `entry`.

        FLAC/WAV/OGG are seeked: only these samples are read and decoded. WebM
        is decoded whole and sliced. Raises ValueError unless
        0 <= first < last <= the number of samples.
        """
        try:
            return self._read_frames(entry, first, last)
        finally:
            if self.max_cache_fd <= 0:
                self.close()  # no cache: never keep a descriptor past the read

    def _read_frames(self, entry: str, first: int, last: int):
        import soundfile

        path, first_byte, end_byte = parse_omniio_entry(entry)
        if hasattr(os, "pread") and os.pread(self._fd(path), 4, first_byte) != _EBML:
            src = _ByteRange(self._fd(path), first_byte, end_byte)
            with soundfile.SoundFile(src) as f:
                rate = f.samplerate
                if not 0 <= first < last <= f.frames:
                    raise ValueError(
                        f"samples {first}-{last} out of the {f.frames} of {entry}"
                    )
                f.seek(first)
                array = f.read(last - first, dtype="float32", always_2d=True)
            if len(array) != last - first:
                raise RuntimeError(f"short read of samples {first}-{last} of {entry}")
            if not self.always_2d and array.shape[1] == 1:
                array = array[:, 0]
            return int(rate), array
        rate, array = self(entry)
        if not 0 <= first < last <= len(array):
            raise ValueError(
                f"samples {first}-{last} out of the {len(array)} of {entry}"
            )
        return rate, array[first:last]

    def close(self):
        """Close the bin file descriptors this process opened."""
        fds = self._own_fds()
        while fds:
            os.close(fds.popitem()[1])

    def __del__(self):
        """Close the cached descriptors when the reader is collected."""
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


class OmniIOAudioScpReader(OmniIOEntryReader, collections.abc.Mapping):
    """Reader for an scp whose entries point into Omni-IO archives.

    Examples:
        utt1 /data/format.1/wav/blob_0.bin:0:31052
        utt2 /data/format.1/wav/blob_0.bin:31052:92707

        >>> reader = OmniIOAudioScpReader('wav.scp')
        >>> rate, array = reader['utt1']

    Entries are read by ``OmniIOEntryReader``, which keeps the bin files open.

    Args:
        fname: The scp file.
        always_2d: Keep the channel axis even for single-channel audio.
        max_cache_fd: Bin files kept open; 0 opens and closes per read.
    """

    @typechecked
    def __init__(
        self,
        fname: Union[Path, str],
        always_2d: bool = False,
        max_cache_fd: int = 32,
    ):
        """Read the scp; the bin files are opened on first use."""
        super().__init__(always_2d=always_2d, max_cache_fd=max_cache_fd)
        self.fname = fname
        self.data = read_2columns_text(fname)

    def __getitem__(self, key: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Return (sampling rate, signal) of `key`, or (None, None) if absent."""
        return self(self.data[key])

    def read_span(self, key: str, start: int, end: int) -> Tuple[int, np.ndarray]:
        """Return (sampling rate, samples `start`..`end` - 1) of `key`.

        Only that part of the archive entry is read (see ``read_frames``).
        """
        return self.read_frames(self.data[key], start, end)

    def get_path(self, key: str) -> str:
        """Return the scp entry of `key` (``<bin>:<start>:<end>`` or "None")."""
        return self.data[key]

    def __contains__(self, item):
        """Whether the scp lists `item`."""
        return item in self.data

    def __len__(self):
        """Number of utterances in the scp."""
        return len(self.data)

    def __iter__(self):
        """Iterate over the utterance ids."""
        return iter(self.data)

    def keys(self):
        """The utterance ids."""
        return self.data.keys()


class OmniIOSegmentsReader(collections.abc.Mapping):
    """Reader of a Kaldi ``segments`` file over a ``wav.scp`` of Omni-IO entries.

    Examples:
        segments (wav.scp is read from the same directory):
            win1 rec1 0.0 10.0
            win2 rec1 10.0 20.0

        >>> reader = OmniIOSegmentsReader('dump/raw/train_win/segments')
        >>> rate, array = reader['win2']

    Each segment is cut as ``SegmentsExtractor`` cuts it when dumping, but at
    load time and reading only the segment's part of the archive entry.

    Args:
        fname: The segments file.
        wav_scp: The recordings' scp; default: ``wav.scp`` next to `fname`.
        always_2d: Keep the channel axis even for single-channel audio.
        max_cache_fd: Bin files kept open; 0 opens and closes per read.
    """

    @typechecked
    def __init__(
        self,
        fname: Union[Path, str],
        wav_scp: Optional[Union[Path, str]] = None,
        always_2d: bool = False,
        max_cache_fd: int = 32,
    ):
        """Read the segments and the recordings' scp."""
        self.fname = fname
        self.wav = read_2columns_text(wav_scp or Path(fname).parent / "wav.scp")
        self.segments: Dict[str, Tuple[str, float, float]] = {}
        with open(fname, encoding="utf-8") as f:
            for linenum, line in enumerate(f, 1):
                sps = line.split()
                if not sps:
                    continue
                where = f"{fname}:{linenum}"
                if len(sps) != 4:
                    raise RuntimeError(f"{where}: expected 4 columns: {line!r}")
                key, rec, start, end = sps
                if key in self.segments:
                    raise RuntimeError(f"{where}: segment {key} is duplicated")
                if rec not in self.wav:
                    raise RuntimeError(f"{where}: {rec} is not in the wav.scp")
                self.segments[key] = (rec, *_segment_times(start, end, where))
        self.reader = OmniIOEntryReader(always_2d=always_2d, max_cache_fd=max_cache_fd)

    def __getitem__(self, key: str) -> Tuple[int, np.ndarray]:
        """Return (sampling rate, signal) of segment `key`."""
        rec, start, end = self.segments[key]
        if self.wav[rec] == "None":
            raise RuntimeError(f"segment {key}: recording {rec} has no audio")
        return self.reader.read_segment(self.wav[rec], start, end)

    def __contains__(self, item):
        """Whether `item` is a segment."""
        return item in self.segments

    def __len__(self):
        """Number of segments."""
        return len(self.segments)

    def __iter__(self):
        """Iterate over the segment ids."""
        return iter(self.segments)

    def keys(self):
        """The segment ids."""
        return self.segments.keys()

    def close(self):
        """Close the bin files this process opened."""
        self.reader.close()


class OmniIOAudioWriter:
    """Write audio into one Omni-IO archive and an scp pointing into it.

    The archive is ``outdir/blob_<i>.bin`` + ``outdir/metadata.parquet`` with
    the columns that ``omniio.blob.blob.Blob`` writes (``id``, ``start_byte``,
    ``end_byte``, ``bin_index``, ``path`` and the audio metadata of
    ``omniio.audio.write.audio_write``), so other Omni-IO tools can read it.
    A new bin file is started once ``max_bin_size`` bytes would be exceeded.
    The writer owns ``outdir``: archive files left there by an earlier run
    (``blob_*.bin``, ``metadata.parquet``) are removed when it starts, so a
    rerun never leaves stale entries behind.

    Examples:
        >>> with OmniIOAudioWriter('dump/format.1/wav', 'dump/format.1/wav.scp') as w:
        ...     w['utt1'] = 16000, numpy_array

    Args:
        outdir: Archive directory.
        scpfile: The scp file to write.
        format: Audio codec inside the archive ("flac", "wav" or "webm").
        bit_depth: PCM bit depth (ignored for "webm").
        max_bin_size: Maximum size of one bin file in bytes.
    """

    @typechecked
    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format: str = "flac",
        bit_depth: int = 16,
        max_bin_size: int = 320 * 1024 * 1024,
    ):
        """Start a new archive in `outdir`, removing one left by an earlier run."""
        self.dir = Path(outdir).absolute()
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in [*self.dir.glob("blob_*.bin"), self.dir / "metadata.parquet"]:
            stale.unlink(missing_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format
        self.bit_depth = bit_depth
        self.max_bin_size = max_bin_size

        self.bin_index = 0
        self.offset = 0
        self.fbin = self._bin_path(self.bin_index).open("wb")
        self.rows: List[Dict] = []
        self.ids = set()
        self.closed = False

    def _bin_path(self, index: int) -> Path:
        """Path of the index-th bin file (the naming of omniio's Blob)."""
        return self.dir / f"blob_{index}.bin"

    def __setitem__(self, key: str, value: Tuple[int, np.ndarray]):
        """Encode (rate, signal) and append it to the archive under `key`.

        Args:
            key: Utterance id; must be new to this writer.
            value: (sampling rate, signal of shape (frames,) or (frames, ch)).
        """
        from omniio.audio.write import audio_write

        if self.closed:
            raise RuntimeError("writer is closed")
        rate, signal = value
        if key in self.ids:
            raise RuntimeError(f"{key} is duplicated")
        raw_bytes, meta = audio_write(
            (np.asarray(signal), int(rate)),
            key,
            target_format=self.format,
            target_bit_depth=self.bit_depth,
        )
        n_bytes = len(raw_bytes)
        if self.offset > 0 and self.offset + n_bytes > self.max_bin_size:
            self.fbin.close()
            self.bin_index += 1
            self.offset = 0
            self.fbin = self._bin_path(self.bin_index).open("wb")

        self.fbin.write(raw_bytes)
        path = str(self._bin_path(self.bin_index))
        start, end = self.offset, self.offset + n_bytes
        meta.update(
            id=key, start_byte=start, end_byte=end, bin_index=self.bin_index, path=path
        )
        self.rows.append(meta)
        self.ids.add(key)
        self.offset = end
        self.fscp.write(f"{key} {path}:{start}:{end}\n")

    def __enter__(self):
        """Use the writer as a context manager; it is closed on exit."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the writer, writing metadata.parquet."""
        self.close()

    def close(self):
        """Finish the archive: write metadata.parquet and close the files.

        An archive with no entry leaves no bin and no metadata file. Calling
        close() again does nothing.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.closed:
            return
        self.closed = True
        self.fbin.close()
        if not self.rows:
            self._bin_path(0).unlink(missing_ok=True)
        else:
            pq.write_table(
                pa.Table.from_pylist(self.rows), self.dir / "metadata.parquet"
            )
        self.fscp.close()
