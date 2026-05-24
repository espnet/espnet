"""Prepare Kaldi-style data directories for MagicData-RAMC ASR.

Input layout (after tar extraction of MagicData-RAMC.tar.gz):
    <raw>/DataPartition/{train,dev,test}.tsv   -- split definitions
    <raw>/MDT2021S003/WAV/<reco_id>.wav        -- ~30 min dual-channel recordings
    <raw>/MDT2021S003/TXT/<reco_id>.txt        -- timestamped transcriptions

Output (one such 5-file group per split, expected by ESPnet's asr.sh):
    <out>/<split>/wav.scp    reco_id  -> SoX pipe (mixdown to mono 16 kHz)
    <out>/<split>/segments   utt_id   -> reco_id, start_sec, end_sec
    <out>/<split>/text       utt_id   -> cleaned transcription
    <out>/<split>/utt2spk    utt_id   -> spk_id
    <out>/<split>/spk2utt    spk_id   -> list of utt_ids

utt_id format: "<spk_id>-<reco_id>-<start_ms 07d>-<end_ms 07d>".
This makes spk_id a strict prefix of utt_id and keeps all utterances of a
speaker contiguous after sorting, satisfying fix_data_dir.sh / validate_data_dir.sh.
"""

import argparse
import os
import re
import sys
from pathlib import Path

SPECIAL_TAGS = ["[+]", "[*]", "[LAUGHTER]", "[SONANT]", "[MUSIC]"]
INVALID_SPK = "G00000000"  # README defines this as the "no speaker" placeholder

# Keep CJK ideographs (incl. Extension A) + ASCII letters/digits.
# Everything else (Chinese & ASCII punctuation, whitespace) is dropped.
KEEP_PATTERN = re.compile(r"[一-鿿㐀-䶿A-Za-z0-9]")

# Raw TXT line: "[start,end]\tspk\tgender,lang\ttranscription"
LINE_RE = re.compile(r"^\[([\d.]+),([\d.]+)\]\s+(\S+)\s+\S+\s+(.*)$")


def clean_text(text, strip_markers, drop_special_segments):
    """Clean a raw transcription line.

    Two independent filter modes:
      * drop_special_segments=True : drop any segment whose raw text contains
        any of [+], [*], [LAUGHTER], [SONANT], [MUSIC].
      * strip_markers=True : strip those markers from the kept text (no drop).

    [*] (unintelligible) is unconditionally dropped because it never carries
    recoverable text. If both flags are False, markers are left literally in
    the text (except [*]).

    Returns the cleaned string, or None if the segment must be dropped.
    """
    if "[*]" in text:
        return None
    if drop_special_segments and any(tag in text for tag in SPECIAL_TAGS):
        return None
    if strip_markers:
        for tag in SPECIAL_TAGS:
            text = text.replace(tag, "")
    cleaned = "".join(KEEP_PATTERN.findall(text))
    if not cleaned:
        return None
    return cleaned


def parse_split_tsv(tsv_path):
    """Return reco_ids (basename without .wav) belonging to this split.

    Each tsv begins with a header-like row (a bare directory path) followed
    by "<filename>\t<size>" rows. We keep only rows whose first column ends
    with ".wav".
    """
    names = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split("\t")[0]
            base = os.path.basename(first)
            if not base.endswith(".wav"):
                continue
            names.append(base[: -len(".wav")])
    return names


def parse_txt(txt_path):
    """Yield (start_sec, end_sec, spk_id, raw_text) for each annotation line."""
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            m = LINE_RE.match(line)
            if not m:
                continue  # silently skip malformed lines
            start, end, spk, text = m.groups()
            yield float(start), float(end), spk, text


def write_split(out_dir, wav_scp, segments, text_lines, utt2spk):
    """Write the five Kaldi-style files, sorted by primary key."""
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_scp.sort(key=lambda x: x[0])
    segments.sort(key=lambda x: x[0])
    text_lines.sort(key=lambda x: x[0])
    utt2spk.sort(key=lambda x: x[0])

    with open(out_dir / "wav.scp", "w", encoding="utf-8") as f:
        for reco_id, cmd in wav_scp:
            f.write(f"{reco_id} {cmd}\n")
    with open(out_dir / "segments", "w", encoding="utf-8") as f:
        for utt_id, reco_id, s, e in segments:
            f.write(f"{utt_id} {reco_id} {s:.3f} {e:.3f}\n")
    with open(out_dir / "text", "w", encoding="utf-8") as f:
        for utt_id, t in text_lines:
            f.write(f"{utt_id} {t}\n")
    with open(out_dir / "utt2spk", "w", encoding="utf-8") as f:
        for utt_id, spk in utt2spk:
            f.write(f"{utt_id} {spk}\n")

    spk2utts = {}
    for utt, spk in utt2spk:
        spk2utts.setdefault(spk, []).append(utt)
    with open(out_dir / "spk2utt", "w", encoding="utf-8") as f:
        for spk in sorted(spk2utts):
            f.write(f"{spk} {' '.join(sorted(spk2utts[spk]))}\n")


def process_split(split, raw_dir, out_dir, args, available_wavs):
    wav_dir = raw_dir / "MDT2021S003" / "WAV"
    txt_dir = raw_dir / "MDT2021S003" / "TXT"
    tsv_path = raw_dir / "DataPartition" / f"{split}.tsv"

    if not tsv_path.is_file():
        print(f"[{split}] missing partition file: {tsv_path}", file=sys.stderr)
        return

    reco_ids = parse_split_tsv(tsv_path)

    wav_scp, segments, text_lines, utt2spk = [], [], [], []
    stats = dict(
        kept=0,
        missing_audio=0,
        missing_txt=0,
        invalid_spk=0,
        bad_duration=0,
        dropped_special=0,
        bad_text_len=0,
    )

    for reco_id in reco_ids:
        if reco_id not in available_wavs:
            stats["missing_audio"] += 1
            continue
        wav_path = wav_dir / f"{reco_id}.wav"
        txt_path = txt_dir / f"{reco_id}.txt"
        if not txt_path.is_file():
            stats["missing_txt"] += 1
            continue

        # SoX pipe: read native wav, downmix to mono, resample to 16 kHz 16-bit PCM.
        # ESPnet's scripts/audio/format_wav_scp.sh consumes this pipe in stage 4
        # and cuts it into per-utterance audio using the segments file.
        wav_scp.append(
            (reco_id,
             f"sox {wav_path} -t wav -r 16000 -b 16 -c 1 - remix - |")
        )

        kept_in_file = False
        for start, end, spk, raw_text in parse_txt(txt_path):
            if spk == INVALID_SPK:
                stats["invalid_spk"] += 1
                continue

            dur_ms = (end - start) * 1000.0
            if dur_ms < args.filter_min_time or dur_ms > args.filter_max_time:
                stats["bad_duration"] += 1
                continue

            cleaned = clean_text(
                raw_text,
                strip_markers=args.filter_special_symbols,
                drop_special_segments=args.drop_special_segments,
            )
            if cleaned is None:
                stats["dropped_special"] += 1
                continue

            n = len(cleaned)
            if n < args.filter_min_text or n > args.filter_max_text:
                stats["bad_text_len"] += 1
                continue

            start_ms = int(round(start * 1000))
            end_ms = int(round(end * 1000))
            utt_id = f"{spk}-{reco_id}-{start_ms:07d}-{end_ms:07d}"

            segments.append((utt_id, reco_id, start, end))
            text_lines.append((utt_id, cleaned))
            utt2spk.append((utt_id, spk))
            stats["kept"] += 1
            kept_in_file = True

        # No surviving utt for this recording -> drop the orphan wav.scp entry.
        if not kept_in_file:
            wav_scp.pop()

    write_split(out_dir / split, wav_scp, segments, text_lines, utt2spk)

    print(
        f"[{split}] kept={stats['kept']}  "
        f"missing_audio={stats['missing_audio']} "
        f"missing_txt={stats['missing_txt']} "
        f"invalid_spk={stats['invalid_spk']} "
        f"bad_dur={stats['bad_duration']} "
        f"dropped_special={stats['dropped_special']} "
        f"bad_text_len={stats['bad_text_len']}",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw",
        help="Directory containing the unpacked MagicData-RAMC corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory under which <split>/{wav.scp,segments,text,utt2spk,spk2utt} "
             "will be written.",
    )
    parser.add_argument(
        "--filter-special-symbols",
        action="store_true",
        help="Strip [+], [*], [LAUGHTER], [SONANT], [MUSIC] markers from text "
             "while keeping the surrounding transcription. [*] segments are "
             "always dropped (unintelligible).",
    )
    parser.add_argument(
        "--drop-special-segments",
        action="store_true",
        help="Drop any segment whose raw text contains [+], [*], [LAUGHTER], "
             "[SONANT], or [MUSIC]. More aggressive than --filter-special-symbols; "
             "if both are set, drop wins (stripping is moot).",
    )
    parser.add_argument(
        "--filter-min-time",
        type=int,
        default=0,
        help="Drop segments shorter than this duration (in ms).",
    )
    parser.add_argument(
        "--filter-max-time",
        type=int,
        default=1000 * 60 * 10,
        help="Drop segments longer than this duration (in ms).",
    )
    parser.add_argument(
        "--filter-min-text",
        type=int,
        default=0,
        help="Drop segments with fewer than this many cleaned characters.",
    )
    parser.add_argument(
        "--filter-max-text",
        type=int,
        default=1000,
        help="Drop segments with more than this many cleaned characters.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    wav_dir = raw_dir / "MDT2021S003" / "WAV"

    if not wav_dir.is_dir():
        sys.exit(f"WAV directory not found: {wav_dir}")

    available_wavs = {p.stem for p in wav_dir.glob("*.wav")}
    print(f"Found {len(available_wavs)} wavs under {wav_dir}", file=sys.stderr)

    for split in ("train", "dev", "test"):
        process_split(split, raw_dir, out_dir, args, available_wavs)

    print("Data preparation completed successfully.", file=sys.stderr)


if __name__ == "__main__":
    main()
