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
"""

import collections
import collections.abc
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text


def parse_omniio_entry(entry: str) -> Tuple[str, int, int]:
    """Split ``<bin path>:<start_byte>:<end_byte>`` (the path may contain ':')."""
    path, start, end = entry.rsplit(":", 2)
    return path, int(start), int(end)


class OmniIOAudioScpReader(collections.abc.Mapping):
    """Reader for an scp whose entries point into Omni-IO archives.

    Examples:
        utt1 /data/format.1/wav/blob_0.bin:0:31052
        utt2 /data/format.1/wav/blob_0.bin:31052:92707

        >>> reader = OmniIOAudioScpReader('wav.scp')
        >>> rate, array = reader['utt1']

    Like ``kaldi_ark`` with ``max_cache_fd``, the bin files are kept open
    (up to ``max_cache_fd`` of them, least recently used closed first), so
    reading an utterance is a single positional read rather than an
    open/seek/read/close: every open is a metadata request, which is what is
    slow on shared file systems. Reads use ``os.pread``, which does not move a
    shared file offset, so DataLoader workers forked with these descriptors do
    not interfere with each other.

    FLAC, WAV and OGG entries are decoded here exactly as
    ``omniio.interface.audio_read`` decodes them (soundfile, float32,
    (frames, channels)); WebM/Opus entries are handed to ``audio_read``.

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
        self.fname = fname
        self.always_2d = always_2d
        self.max_cache_fd = max_cache_fd
        self.data = read_2columns_text(fname)
        self._fds: "collections.OrderedDict[str, int]" = collections.OrderedDict()

    def _read_bytes(self, path: str, start: int, size: int) -> bytes:
        """Read `size` bytes at `start` of `path`, through the descriptor cache."""
        if self.max_cache_fd <= 0 or not hasattr(os, "pread"):
            with open(path, "rb") as f:
                f.seek(start)
                return f.read(size)
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
            while len(self._fds) > self.max_cache_fd:
                os.close(self._fds.popitem(last=False)[1])
        else:
            self._fds.move_to_end(path)
        data = os.pread(fd, size, start)
        if len(data) != size:
            raise RuntimeError(f"short read from {path}: {len(data)} of {size} bytes")
        return data

    def __getitem__(self, key: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Return (sampling rate, signal) of `key`, or (None, None) if absent."""
        import soundfile

        entry = self.data[key]
        if entry == "None":
            # an utterance without audio, as format_wav_scp.py writes it
            return None, None
        path, start, end = parse_omniio_entry(entry)
        blob = self._read_bytes(path, start, end - start)
        if blob[:4] == b"\x1aE\xdf\xa3":  # WebM/Opus (EBML header)
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

    def close(self):
        """Close the cached bin file descriptors."""
        while self._fds:
            os.close(self._fds.popitem()[1])

    def __del__(self):
        """Close the cached descriptors when the reader is collected."""
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

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
