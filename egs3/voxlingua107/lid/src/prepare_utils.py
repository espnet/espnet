"""Shared audio and manifest output for raw-corpus LID preparation."""

import argparse
import io
import json
from math import gcd
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly


def parser(description):
    """Common destinations and optional small-run limit."""
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
    """Download pinned Parquet shards with Xet, reusing completed local files."""
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
    """Read embedded audio without depending on remote Dataset loading scripts."""
    import pyarrow.parquet as pq

    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            yield from batch.to_pylist()


def save_audio(source, output, start=None, end=None, channel=None):
    """Write a 16 kHz mono utterance, optionally cutting a source recording."""
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


def write_manifest(output_dir, split, examples, provenance):
    """Materialize (ID, language, audio source, start, end, channel) examples."""
    import hashlib

    destination = Path(output_dir).resolve() / split
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "manifest.tsv.tmp"
    count = 0
    with temporary.open("w", encoding="utf-8") as manifest:
        for utt_id, language, source, start, end, channel in examples:
            # Source IDs may contain path separators; retain them in the manifest.
            name = hashlib.sha256(utt_id.encode()).hexdigest()
            audio_path = destination / "audio" / f"{name}.wav"
            if isinstance(source, dict):
                source = (
                    io.BytesIO(source["bytes"])
                    if source.get("bytes")
                    else source["path"]
                )
            save_audio(source, audio_path, start, end, channel)
            manifest.write(f"{utt_id}\t{audio_path}\t{language}\n")
            count += 1
    temporary.replace(destination / "manifest.tsv")
    (destination / "source.json").write_text(
        json.dumps({**provenance, "utterances": count}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{destination}: {count} utterances", flush=True)
    return count
