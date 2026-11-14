"""Shared audio and manifest output for raw-corpus LID preparation."""

import argparse
import io
import json
from math import gcd
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly


def parser(description):
    """Create the common corpus preparation argument parser.

    Args:
        description: CLI help text for the corpus script.

    Returns:
        ArgumentParser with output-dir and max-utterances options.

    Example:
        >>> args = parser("Prepare audio").parse_args(["--output-dir", "data/test"])
    """
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Limit each language/split for a preparation smoke test",
    )
    return result


def hf_files(repo, revision, source_dir, patterns, small_run=False):
    """Download pinned Parquet shards with Xet, reusing completed local files.

    Args:
        repo: Hugging Face dataset repository ID.
        revision: Source revision to resolve to a commit.
        source_dir: Local download destination.
        patterns: Glob patterns selecting published Parquet shards.
        small_run: Download only the first matching shard when true.

    Returns:
        A pair of local file paths and the resolved source commit ID.

    Example:
        >>> files, revision = hf_files(repo, commit, "download", ["data/dev-*.parquet"])
    """
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().dataset_info(repo, revision=revision)
    files = sorted(
        s.rfilename
        for s in info.siblings
        if any(fnmatch(s.rfilename, p) for p in patterns)
    )
    if small_run:
        files = files[:1]
    if not files:
        raise FileNotFoundError(f"No Parquet files match {patterns} in {repo}")
    root = snapshot_download(
        repo,
        repo_type="dataset",
        revision=info.sha,
        allow_patterns=files,
        local_dir=source_dir,
        max_workers=4,
    )
    return [Path(root) / name for name in files], info.sha


def parquet_rows(files):
    """Read embedded audio without remote Dataset loading scripts.

    Args:
        files: Paths to downloaded Parquet shards.

    Yields:
        Source row dictionaries, including embedded audio fields.

    Example:
        >>> row = next(parquet_rows([Path("dev.parquet")]))
    """
    import pyarrow.parquet as pq

    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            yield from batch.to_pylist()


def save_audio(source, output, start=None, end=None, channel=None):
    """Write a 16 kHz mono utterance, optionally cutting a source recording.

    Args:
        source: SoundFile-compatible path or in-memory audio stream.
        output: Destination WAV path.
        start: Optional segment start in seconds.
        end: Optional segment end in seconds.
        channel: Channel index; None averages all channels.

    Returns:
        None. Writes a PCM16 WAV at output.

    Example:
        >>> save_audio("recording.wav", "segment.wav", start=1.0, end=3.0)
    """
    output = Path(output)
    with sf.SoundFile(source) as audio:
        rate = audio.samplerate
        offset = 0 if start is None else round(start * rate)
        audio.seek(offset)
        frames = -1 if end is None else round(end * rate) - offset
        speech = audio.read(frames, dtype="float32", always_2d=True)
    speech = speech.mean(axis=1) if channel is None else speech[:, channel]
    if rate != 16000:
        factor = gcd(rate, 16000)
        speech = resample_poly(speech, 16000 // factor, rate // factor)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.wav")
    sf.write(temporary, speech, 16000, subtype="PCM_16")
    temporary.replace(output)


def write_manifest(output_dir, split, examples, provenance, parallel=False):
    """Materialize examples and write an audio manifest with source metadata.

    Args:
        output_dir: Root directory for prepared splits.
        split: Published split name.
        examples: Iterable of (ID, language, audio source, start, end, channel).
            Audio sources may be paths or Hugging Face audio dictionaries.
        provenance: Source revision and selection settings saved to source.json.
        parallel: Use the active ESPnet3 runner backend for file-backed examples.
            Keep false for streaming embedded audio or temporary Babel WAVs.

    Returns:
        Number of utterances written to split/manifest.tsv.

    Example:
        >>> rows = [("utt1", "eng", "recording.wav", None, None, None)]
        >>> write_manifest("data/example", "test", rows, {"split": "test"})
    """
    destination = Path(output_dir).resolve() / split
    destination.mkdir(parents=True, exist_ok=True)
    if parallel:
        from egs3.voxlingua107.esp2_lid.src.download import DownloadProvider
        from egs3.voxlingua107.esp2_lid.src.manifest import ManifestRunner

        tasks = []
        for example in examples:
            example = list(example)
            example[2] = str(Path(example[2]).expanduser().resolve())
            tasks.append({"destination": str(destination), "example": example})
        count = ManifestRunner(
            DownloadProvider(tasks),
            output_dir=destination,
            shard_subdir=".prepare",
            resume=False,
        )(range(len(tasks)))
    else:
        temporary = destination / "manifest.tsv.tmp"
        count = 0
        with temporary.open("w", encoding="utf-8") as manifest:
            for example in examples:
                manifest.write(_write_example(destination, example))
                count += 1
        temporary.replace(destination / "manifest.tsv")
    (destination / "source.json").write_text(
        json.dumps({**provenance, "utterances": count}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{destination}: {count} utterances", flush=True)
    return count


def _write_example(destination, example):
    """Materialize one utterance and return its manifest row."""
    import hashlib

    utt_id, language, source, start, end, channel = example
    name = hashlib.sha256(utt_id.encode()).hexdigest()
    audio_path = destination / "audio" / f"{name}.wav"
    if isinstance(source, dict):
        source = io.BytesIO(source["bytes"]) if source.get("bytes") else source["path"]
    save_audio(source, audio_path, start, end, channel)
    return f"{utt_id}\t{audio_path}\t{language}\n"
