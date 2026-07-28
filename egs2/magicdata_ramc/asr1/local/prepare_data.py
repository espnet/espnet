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

Filtering / tokenisation policy:
  * Duration, text-length, and tag-based drops only apply to the train split.
    Dev / test are kept as released (per ESPnet recipe convention) so that
    reported CER/WER reflects the full evaluation distribution.
  * Paralinguistic tags ([+], [*], [LAUGHTER], [SONANT], [MUSIC]) are
    preserved as atomic tokens inside the cleaned transcription; local/data.sh
    writes them into data/nlsyms.txt and run.sh passes the file via
    --nlsyms_txt so ESPnet's token-list builder keeps them as single tokens.
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

# Tag-aware splitter: capturing group keeps the tag itself in the split output.
TAG_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TAGS) + ")")

# Raw TXT line: "[start,end]\tspk\tgender,lang\ttranscription"
LINE_RE = re.compile(r"^\[([\d.]+),([\d.]+)\]\s+(\S+)\s+\S+\s+(.*)$")


def clean_text(text, mode, drop_special_segments):
    """Clean a raw transcription line.

    Tokenisation: the line is split on SPECIAL_TAGS so each tag is preserved
    as an atomic token; the surrounding text is filtered by KEEP_PATTERN
    (CJK + alnum). The kept pieces are concatenated without spaces (char-level
    Mandarin convention); ESPnet's tokenize_text.py, fed with
    `--non_linguistic_symbols data/nlsyms.txt`, will recognise the tags as
    single tokens during token-list construction.

    Filtering policy:
      * mode='train' :
          - drop a segment whose cleaned text is literally `[*]` (pure
            unintelligible: no learnable acoustic-to-symbol mapping),
          - optionally drop any segment containing any SPECIAL_TAG when
            drop_special_segments=True (more aggressive cleaning).
      * mode='eval' (dev / test) :
          - apply none of the above; the only drop is the technically
            required one of an empty cleaned string (validate_data_dir.sh
            rejects it). This preserves the released split as-is for fair
            evaluation, per ESPnet recipe convention.

    Returns the cleaned string, or None if the segment must be dropped.
    """
    pieces = []
    for part in TAG_SPLIT_RE.split(text):
        if not part:
            continue
        if part in SPECIAL_TAGS:
            pieces.append(part)
        else:
            chars = "".join(KEEP_PATTERN.findall(part))
            if chars:
                pieces.append(chars)
    cleaned = "".join(pieces)

    if not cleaned:
        return None

    if mode == "train":
        if cleaned == "[*]":
            return None
        if drop_special_segments and any(tag in cleaned for tag in SPECIAL_TAGS):
            return None
    return cleaned


def parse_split_tsv(tsv_path):
    """Return reco_ids (basename without .wav) belonging to this split.

    Each tsv begins with a header-like row (a bare directory path) followed
    by "<filename>\t<size>" rows. We keep only rows whose first column ends
    with ".wav".
    """
    names = []
    # utf-8-sig strips a leading BOM if present (TSV files saved on Windows
    # may include one). Backslash → forward slash before basename() so paths
    # like "MDT2021S003\WAV\foo.wav" resolve correctly on POSIX.
    with open(tsv_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split("\t")[0].replace("\\", "/")
            base = os.path.basename(first)
            if not base.endswith(".wav"):
                continue
            names.append(base[: -len(".wav")])
    return names


def parse_txt(txt_path):
    """Yield (start_sec, end_sec, spk_id, raw_text) for each annotation line."""
    # utf-8-sig strips a leading BOM (some TXT annotations are saved with one
    # by Windows editors; without this the BOM stays in the first line and
    # LINE_RE silently misses it).
    with open(txt_path, encoding="utf-8-sig") as f:
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

    # Per ESPnet recipe convention: filter only the training split. The dev /
    # test splits are kept as released so that reported CER/WER reflects the
    # full evaluation distribution rather than a hand-picked subset.
    is_train = split == "train"
    mode = "train" if is_train else "eval"

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
            (reco_id, f"sox {wav_path} -t wav -r 16000 -b 16 -c 1 - remix - |")
        )

        kept_in_file = False
        for start, end, spk, raw_text in parse_txt(txt_path):
            if spk == INVALID_SPK:
                stats["invalid_spk"] += 1
                continue

            if is_train:
                dur_ms = (end - start) * 1000.0
                if dur_ms < args.filter_min_time or dur_ms > args.filter_max_time:
                    stats["bad_duration"] += 1
                    continue

            cleaned = clean_text(
                raw_text,
                mode=mode,
                drop_special_segments=args.drop_special_segments,
            )
            if cleaned is None:
                stats["dropped_special"] += 1
                continue

            if is_train:
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
        "--drop-special-segments",
        action="store_true",
        help="Train-only: drop any segment whose cleaned text contains [+], "
        "[LAUGHTER], [SONANT], or [MUSIC]. By default these are kept and the "
        "tags are emitted as non-linguistic symbols (see data/nlsyms.txt). "
        "Train-only pure-[*] segments are always dropped (no learnable "
        "audio-to-symbol mapping). Dev/test splits are kept as released.",
    )
    parser.add_argument(
        "--filter-min-time",
        type=int,
        default=0,
        help="Train-only: drop segments shorter than this duration (in ms). "
        "Dev/test ignore this filter.",
    )
    parser.add_argument(
        "--filter-max-time",
        type=int,
        default=1000 * 60 * 10,
        help="Train-only: drop segments longer than this duration (in ms). "
        "Dev/test ignore this filter.",
    )
    parser.add_argument(
        "--filter-min-text",
        type=int,
        default=0,
        help="Train-only: drop segments with fewer than this many cleaned "
        "characters. Dev/test ignore this filter.",
    )
    parser.add_argument(
        "--filter-max-text",
        type=int,
        default=1000,
        help="Train-only: drop segments with more than this many cleaned "
        "characters. Dev/test ignore this filter.",
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
