"""Shared helpers for writing voice-activity data directories.

A VAD data directory is a Kaldi diarization directory with one pseudo
speaker per recording: every recording gets the union of all speakers'
speech as its only "speaker", so that diar.sh with ``--num_spk 1`` trains a
single speech/non-speech output.

    wav.scp   <rec-id> <path or command |>
    segments  <seg-id> <rec-id> <start> <end>
    utt2spk   <seg-id> <rec-id>
"""

import logging
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Interval = Tuple[float, float]


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Union of possibly overlapping intervals, sorted by start time.

    Intervals that touch (end == next start) are merged; empty or inverted
    intervals are dropped.
    """
    merged: List[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def clip_intervals(intervals: Iterable[Interval], duration: float) -> List[Interval]:
    """Clip intervals to [0, duration]; intervals left empty are dropped."""
    out = []
    for start, end in intervals:
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            out.append((start, end))
    return out


def write_vad_dir(
    out_dir: Path,
    recordings: Dict[str, str],
    speech: Dict[str, List[Interval]],
) -> int:
    """Write wav.scp, segments and utt2spk; return the number of recordings.

    Args:
        out_dir: Output data directory.
        recordings: rec-id -> wav.scp entry (a path or a command ending in "|").
        speech: rec-id -> merged speech intervals in seconds. Recordings with
            no speech at all are skipped: diar.sh would give them a label with
            zero speakers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = sorted(r for r in recordings if speech.get(r))
    with (
        (out_dir / "wav.scp").open("w") as wav,
        (out_dir / "segments").open("w") as seg,
        (out_dir / "utt2spk").open("w") as u2s,
    ):
        for rec in kept:
            wav.write(f"{rec} {recordings[rec]}\n")
            for start, end in speech[rec]:
                if round(end * 100) <= round(start * 100):
                    continue  # empty at the 10 ms resolution of segments
                # rec-id prefix keeps utt2spk and spk2utt in the same sort order
                utt = f"{rec}-{round(start * 100):08d}-{round(end * 100):08d}"
                seg.write(f"{utt} {rec} {start:.2f} {end:.2f}\n")
                u2s.write(f"{utt} {rec}\n")
    return len(kept)


def audio_info(path: Path) -> Tuple[float, int]:
    """(duration in seconds, number of channels) of a wav/flac or NIST SPHERE file.

    SPHERE headers are read directly, since SPHERE files may be compressed
    (shorten), which libsndfile cannot open; wav.scp decodes them with sph2pipe.
    """
    if path.suffix.lower() == ".sph":
        with path.open("rb") as f:
            header = f.read(1024).decode("latin-1")
        fields = {}
        for line in header.splitlines()[2:]:
            parts = line.split()
            if len(parts) == 3:
                fields[parts[0]] = parts[2]
        rate = int(fields["sample_rate"])
        return int(fields["sample_count"]) / rate, int(fields.get("channel_count", 1))
    import soundfile

    info = soundfile.info(str(path))
    return info.duration, info.channels


def readable_audio_info(path: Path, skipped: List[str]) -> Optional[Tuple[float, int]]:
    """audio_info(), or None (and `path` appended to `skipped`) if unreadable.

    Corpus copies can hold empty or truncated files from interrupted
    downloads; those are listed instead of stopping the whole preparation.
    """
    try:
        duration, channels = audio_info(path)
    except Exception as e:  # noqa: BLE001 - any decoder error means unreadable
        logging.warning("unreadable audio %s: %s", path, e)
        skipped.append(str(path))
        return None
    if duration <= 0:
        logging.warning("empty audio %s", path)
        skipped.append(str(path))
        return None
    return duration, channels


def write_skipped(out_dir: Path, skipped: List[str]):
    """Record the unreadable files of this directory in skipped_audio.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skipped_audio.txt").write_text("".join(f"{s}\n" for s in skipped))
    if skipped:
        logging.warning("%d unreadable files listed in %s", len(skipped), out_dir)


def wav_scp_entry(path: Path, sph2pipe: str = "sph2pipe") -> str:
    """wav.scp entry for a single-channel file, decoding SPHERE with sph2pipe."""
    if path.suffix.lower() == ".sph":
        return f"{sph2pipe} -f wav -p {shlex.quote(str(path))} |"
    return str(path)


def iter_annotation_files(source: Path, suffix: str):
    """Yield (file name, text) of every file ending in `suffix`.

    `source` is either a directory (searched recursively) or a zip archive, so
    corpus annotations can be read without unpacking them.
    """
    import zipfile

    if source.is_dir():
        for path in sorted(source.glob(f"**/*{suffix}")):
            yield path.name, path.read_text(encoding="latin-1")
    else:
        with zipfile.ZipFile(source) as z:
            for name in sorted(z.namelist()):
                if name.endswith(suffix):
                    yield Path(name).name, z.read(name).decode("latin-1")
