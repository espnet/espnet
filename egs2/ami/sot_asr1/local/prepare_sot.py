#!/usr/bin/env python3
"""Prepare SOT (Serialized Output Training) data from Lhotse CutSets.

Reads Lhotse CutSet .jsonl.gz manifests, orders speakers within each cut by
start time, and writes Kaldi-format data directories for ESPnet2 training.

Output files (in ``--output_dir``):
    wav.scp, text, utt2spk, spk2utt
"""

import argparse
import logging
import os
from typing import Optional

from lhotse import CutSet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SPEAKER_CHANGE_TOKEN = " ???? "


def round_nearest(value: float, resolution: float) -> float:
    return round(value / resolution) * resolution


def merge_supervisions(supervisions, max_timestamp_pause: float):
    """Merge adjacent supervisions for the same speaker within pause threshold."""
    merged = []
    for sup in sorted(supervisions, key=lambda x: x.start):
        seg = {
            "start": sup.start,
            "end": sup.start + sup.duration,
            "text": sup.text,
        }
        if merged and (sup.start - merged[-1]["end"]) <= max_timestamp_pause:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1]["text"] + " " + seg["text"]
        else:
            merged.append(seg)
    return merged


def get_transcript_units(
    cut, use_timestamps: bool, max_timestamp_pause: float, lowercase: bool = True
):
    """Extract per-speaker transcript units from a cut. Returns list of dicts
    with keys: speaker, text, start."""
    units = []
    speakers = sorted({s.speaker for s in cut.supervisions})

    for speaker_id in speakers:
        spk_supervisions = [s for s in cut.supervisions if s.speaker == speaker_id]
        merged = merge_supervisions(spk_supervisions, max_timestamp_pause)

        # ``per_spk_flags`` is an AMI-specific custom attribute marking whether
        # a speaker's last segment runs off the end of the cut window.
        last_segment_unfinished = getattr(cut, "per_spk_flags", {}).get(
            speaker_id, False
        )

        for idx, seg in enumerate(merged):
            text = seg["text"].strip()
            if not text:
                continue
            if lowercase:
                text = text.lower()

            if use_timestamps:
                start_ts = f"<|{round_nearest(seg['start'], 0.02):.2f}|>"
                skip_end = (idx == len(merged) - 1) and last_segment_unfinished
                if skip_end:
                    text = f"{start_ts} {text}"
                else:
                    end_ts = f"<|{round_nearest(seg['end'], 0.02):.2f}|>"
                    text = f"{start_ts} {text}{end_ts}"

            units.append(
                {"speaker": speaker_id, "text": text, "start": seg["start"]}
            )

    return units


def merge_speaker_units(units, use_timestamps: bool):
    """Merge all utterance units per speaker into one entry."""
    spk_map = {}
    for u in units:
        spk_map.setdefault(u["speaker"], []).append(u)

    joiner = "" if use_timestamps else " "
    return [
        {
            "speaker": spk,
            "text": joiner.join(u["text"] for u in utts),
            "start": min(u["start"] for u in utts),
        }
        for spk, utts in spk_map.items()
    ]


def build_sot_text(
    cut, use_timestamps: bool, max_timestamp_pause: float, lowercase: bool = True
):
    """Build the serialized SOT text for a single cut. Speakers are ordered
    by their earliest start time."""
    units = get_transcript_units(
        cut, use_timestamps, max_timestamp_pause, lowercase=lowercase
    )
    items = sorted(merge_speaker_units(units, use_timestamps), key=lambda x: x["start"])
    return SPEAKER_CHANGE_TOKEN.join(x["text"] for x in items)


def get_audio_entry(cut, segments_dir: Optional[str] = None) -> Optional[str]:
    """Get wav.scp entry for a cut's audio. If the cut is a strict sub-segment
    of a longer recording and ``segments_dir`` is provided, extracts the
    audio segment to a WAV file and returns its path; otherwise returns the
    plain recording file path."""
    if not getattr(cut, "recording", None):
        return None

    audio_path = next(
        (src.source for src in cut.recording.sources if src.type == "file"), None
    )
    if audio_path is None:
        return None

    is_segment = cut.start > 0.01 or abs(cut.duration - cut.recording.duration) > 0.01
    if is_segment and segments_dir is not None:
        import soundfile as sf

        seg_path = os.path.join(segments_dir, f"{cut.id}.wav")
        if not os.path.exists(seg_path):
            audio = cut.load_audio()  # shape: (channels, samples)
            sf.write(seg_path, audio[0], cut.sampling_rate)
        return seg_path
    return audio_path


def process_cutset(
    cutset: CutSet,
    use_timestamps: bool,
    max_timestamp_pause: float,
    lowercase: bool = True,
    segments_dir: Optional[str] = None,
):
    """Process a CutSet. Returns (entries, stats) where entries is a list of
    (utt_id, wav_path, text) tuples and stats is a dict of skip counters."""
    entries = []
    stats = {"total": 0, "skipped_too_long": 0, "skipped_no_audio": 0,
             "skipped_no_text": 0, "processed": 0}

    for cut in cutset:
        stats["total"] += 1
        # Whisper timestamp tokens only cover 0-30 s; longer cuts would
        # produce OOV timestamp IDs.
        if cut.duration >= 30.0:
            stats["skipped_too_long"] += 1
            continue

        audio_path = get_audio_entry(cut, segments_dir=segments_dir)
        if audio_path is None:
            stats["skipped_no_audio"] += 1
            continue

        text = build_sot_text(
            cut,
            use_timestamps=use_timestamps,
            max_timestamp_pause=max_timestamp_pause,
            lowercase=lowercase,
        )
        if not text.strip():
            stats["skipped_no_text"] += 1
            continue

        # Append <|endoftext|> so the model learns to terminate. The ESPnet
        # preprocessor adds the SOT prefix [<|en|>, <|transcribe|>] and the
        # model's add_sos_eos adds <|startoftranscript|> as SOS.
        entries.append((cut.id, audio_path, f"{text} <|endoftext|>"))
        stats["processed"] += 1

    return entries, stats


def write_kaldi_data(entries, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    entries.sort(key=lambda x: x[0])
    paths = {k: os.path.join(output_dir, k) for k in
             ("wav.scp", "text", "utt2spk", "spk2utt")}
    with (
        open(paths["wav.scp"], "w") as f_wav,
        open(paths["text"], "w") as f_text,
        open(paths["utt2spk"], "w") as f_u2s,
        open(paths["spk2utt"], "w") as f_s2u,
    ):
        for utt_id, wav_path, text in entries:
            f_wav.write(f"{utt_id} {wav_path}\n")
            f_text.write(f"{utt_id} {text}\n")
            # For SOT, each utterance group is its own "speaker".
            f_u2s.write(f"{utt_id} {utt_id}\n")
            f_s2u.write(f"{utt_id} {utt_id}\n")
    logger.info(f"Wrote {len(entries)} entries to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SOT data from Lhotse CutSets for ESPnet2 training.",
    )
    parser.add_argument("--cutset_paths", type=str, nargs="+", required=True,
                        help="Paths to Lhotse CutSet .jsonl.gz files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for Kaldi-format data")
    parser.add_argument("--use_timestamps",
                        type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=False,
                        help="Include timestamps in text output")
    parser.add_argument("--max_timestamp_pause", type=float, default=2.0,
                        help="Max pause (seconds) for merging adjacent segments")
    parser.add_argument("--lowercase",
                        type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=True,
                        help="Lowercase transcript text (default: true)")
    args = parser.parse_args()

    logger.info(f"Use timestamps: {args.use_timestamps}, "
                f"max_timestamp_pause: {args.max_timestamp_pause}")

    segments_dir = os.path.join(args.output_dir, "segments_wav")
    os.makedirs(segments_dir, exist_ok=True)

    all_entries = []
    total_stats = {"total": 0, "skipped_too_long": 0, "skipped_no_audio": 0,
                   "skipped_no_text": 0, "processed": 0}

    for cutset_path in args.cutset_paths:
        logger.info(f"Loading cutset: {cutset_path}")
        entries, stats = process_cutset(
            CutSet.from_file(cutset_path),
            use_timestamps=args.use_timestamps,
            max_timestamp_pause=args.max_timestamp_pause,
            lowercase=args.lowercase,
            segments_dir=segments_dir,
        )
        all_entries.extend(entries)
        for k, v in stats.items():
            total_stats[k] += v

    logger.info(f"Stats: {total_stats}")

    # Deduplicate by utt_id, keep first seen.
    seen = set()
    unique = [e for e in all_entries if not (e[0] in seen or seen.add(e[0]))]
    if len(unique) != len(all_entries):
        logger.warning(
            f"Removed {len(all_entries) - len(unique)} duplicate utt_ids"
        )

    write_kaldi_data(unique, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
