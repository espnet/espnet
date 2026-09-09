#!/usr/bin/env python3
# Copyright 2026 Carnegie Mellon University (Haerin Kim)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Convert the Seoul Corpus (flac + Praat TextGrid) into Kaldi-style data dirs.

The Seoul Corpus ships 240 ten-minute flac files (40 speakers x 6 sessions) and
one Praat TextGrid per flac.  Each TextGrid holds seven interval tiers::

    1 phoneme       2,3 pWord.prono.   4 utt.prono.
                    5,6 pWord.ortho.   7 utt.ortho.

We read an utterance-level tier ("utt.ortho." by default, i.e. the orthographic
transcription; "utt.prono." gives the pronounced/colloquial form) and turn every
interval that carries real speech into one ASR utterance.

Intervals labelled only with a non-speech tag (``<SIL>``, ``<NOISE>``,
``<VOCNOISE>``, ``<LAUGH>``, ``<UNKNOWN>``, ``<PRIVATE.INFO>``) or with
``<IVER>`` (the interviewer speaking) are dropped.  ``<LAUGH-word>`` means the
speaker said "word" while laughing, so it is rewritten to plain "word".

The utterance tier is segmented at every pause, which leaves a lot of very short
(~1.5 s on average) fragments.  ``--merge_gap`` can glue neighbours separated by
nothing but silence/noise back together -- never across the interviewer or across
a fragment we had to throw away, so no merged utterance can hide speech that is
missing from its transcript -- but it defaults to 0, i.e. off, and the recipe
keeps the corpus's own utterance boundaries.  Turning it on changes what an
"utterance" is in every split including test, so scores stop being comparable
with anything measured on the shipped segmentation.
"""

import argparse
import re
import sys
from pathlib import Path

# Tags that never contain speech of the interviewee -> the interval is dropped.
NON_SPEECH_TAGS = {
    "<SIL>",
    "<NOISE>",
    "<VOCNOISE>",
    "<LAUGH>",
    "<IVER>",
    "<IVER-NOISE>",
    "<UNKNOWN>",
    "<PRIVATE.INFO>",
    "<XXX>",
}
# "<LAUGH-그래서>" -> "그래서" (speech produced while laughing).
LAUGH_WORD = re.compile(r"<LAUGH-([^>]+)>")
ANY_TAG = re.compile(r"<[^>]*>")
FILENAME = re.compile(r"^(s\d{2})([mf])(\d{2})([mf])(\d)$")


def read_textgrid(path):
    """Read a Praat long-format TextGrid into ``[(tier_name, intervals)]``.

    The corpus files are UTF-16 with CRLF line endings, but be forgiving and
    fall back to UTF-8 so that re-encoded copies still work.

    Args:
        path: Path to the ``.TextGrid`` file.

    Returns:
        A list of ``(name, intervals)`` pairs, one per interval tier, where
        ``intervals`` is a list of ``(xmin, xmax, text)`` tuples.
    """
    raw = Path(path).read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"cannot decode {path}")

    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]

    tiers = []
    name, xmin, xmax, intervals = None, None, None, None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("name = "):
            if name is not None:
                tiers.append((name, intervals))
            name = line[len("name = ") :].strip().strip('"')
            intervals = []
        elif line.startswith("xmin = "):
            xmin = float(line[len("xmin = ") :])
        elif line.startswith("xmax = "):
            xmax = float(line[len("xmax = ") :])
        elif line.startswith("text = ") and name is not None:
            # A label may be wrapped over several lines; keep reading until the
            # closing quote.  Praat escapes a literal quote by doubling it.
            chunk = line[len("text = ") :].strip()
            while not (len(chunk) >= 2 and chunk.endswith('"')) or chunk.count('"') % 2:
                i += 1
                if i >= len(lines):
                    break
                chunk += " " + lines[i].strip()
            intervals.append((xmin, xmax, chunk.strip('"').replace('""', '"')))
        i += 1
    if name is not None:
        tiers.append((name, intervals))
    return tiers


def normalize(label):
    """Normalize one interval label into an ASR transcript.

    Args:
        label: Raw text of the TextGrid interval.

    Returns:
        The cleaned transcript, or ``""`` if the interval holds no usable
        speech (silence, noise, interviewer, or an unhandled tag).
    """
    label = LAUGH_WORD.sub(r"\1", label).strip()
    if not label:
        return ""
    tokens = [tok for tok in label.split() if tok not in NON_SPEECH_TAGS]
    label = " ".join(tokens).strip()
    # Anything still bracketed is a tag we do not know how to voice: skip the
    # whole utterance rather than train on a transcript with a hole in it.
    if not label or ANY_TAG.search(label):
        return ""
    return " ".join(label.split())


def build_utterances(intervals, merge_gap, max_duration):
    """Turn the intervals of one tier into merged utterance candidates.

    Args:
        intervals: ``(xmin, xmax, text)`` tuples of the utterance tier.
        merge_gap: Merge two speech chunks when at most this many seconds of
            silence/noise separate them.  ``0`` disables merging.
        max_duration: Never let a merged utterance grow beyond this (sec).

    Returns:
        A ``(utterances, n_dropped)`` pair, where ``utterances`` is a list of
        ``(xmin, xmax, transcript)`` tuples.
    """
    utterances = []
    n_dropped = 0
    pending = None  # [xmin, xmax, [words...]] of the utterance being grown
    gap = 0.0  # silence accumulated since the end of `pending`
    for xmin, xmax, label in intervals:
        transcript = normalize(label)
        if not transcript:
            n_dropped += 1
            # Silence and noise may be swallowed by a merge, but the
            # interviewer -- or a fragment whose transcript we could not
            # recover -- must break the utterance.
            if label.strip() in NON_SPEECH_TAGS - {"<IVER>", "<IVER-NOISE>"}:
                gap += xmax - xmin
            else:
                if pending is not None:
                    utterances.append((pending[0], pending[1], " ".join(pending[2])))
                pending, gap = None, 0.0
            continue

        if (
            pending is not None
            and gap <= merge_gap
            and xmax - pending[0] <= max_duration
        ):
            pending[1] = xmax
            pending[2].append(transcript)
        else:
            if pending is not None:
                utterances.append((pending[0], pending[1], " ".join(pending[2])))
            pending = [xmin, xmax, [transcript]]
        gap = 0.0
    if pending is not None:
        utterances.append((pending[0], pending[1], " ".join(pending[2])))
    return utterances, n_dropped


def parse_recording_id(rec_id):
    """Split e.g. ``s01m16f1`` into its speaker id and session number.

    Args:
        rec_id: Basename of a flac/TextGrid file, without extension.

    Returns:
        A ``(speaker_id, spk_num, gender, age, session)`` tuple.
    """
    m = FILENAME.match(rec_id)
    if m is None:
        raise ValueError(f"unexpected Seoul Corpus file name: {rec_id}")
    spk_num, gender, age, _iver_gender, session = m.groups()
    return f"{spk_num}{gender}{age}", spk_num, gender, age, session


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare Kaldi data directories for the Seoul Corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sound_dir", required=True, help="directory with *.flac")
    parser.add_argument("--label_dir", required=True, help="directory with *.TextGrid")
    parser.add_argument("--out_dir", required=True, help="output data directory")
    parser.add_argument(
        "--speakers",
        required=True,
        help="space/comma separated speaker numbers to keep, e.g. 's01 s02'",
    )
    parser.add_argument(
        "--tier",
        default="utt.ortho.",
        help="TextGrid tier to transcribe from",
    )
    parser.add_argument(
        "--min_duration", type=float, default=0.2, help="drop shorter segments (sec)"
    )
    parser.add_argument(
        "--max_duration", type=float, default=20.0, help="drop longer segments (sec)"
    )
    parser.add_argument(
        "--merge_gap",
        type=float,
        default=0.0,
        help="merge neighbouring utterances separated by at most this much "
        "silence/noise; 0 (the default) keeps the corpus segmentation",
    )
    return parser


def main():
    args = get_parser().parse_args()
    keep = set(args.speakers.replace(",", " ").split())
    sound_dir, label_dir = Path(args.sound_dir), Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_scp, segments, texts, utt2spk = {}, [], [], []
    n_dropped = 0
    for tg in sorted(label_dir.glob("*.TextGrid")):
        rec_id = tg.stem
        spk_id, spk_num = parse_recording_id(rec_id)[:2]
        if spk_num not in keep:
            continue
        flac = sound_dir / f"{rec_id}.flac"
        if not flac.exists():
            print(f"warning: no audio for {rec_id}, skipped", file=sys.stderr)
            continue

        tiers = dict(read_textgrid(tg))
        if args.tier not in tiers:
            raise ValueError(f"{tg} has no tier {args.tier!r} (has {list(tiers)})")

        utterances, n_dropped_here = build_utterances(
            tiers[args.tier], args.merge_gap, args.max_duration
        )
        n_dropped += n_dropped_here

        n_kept_here = 0
        for xmin, xmax, transcript in utterances:
            if not args.min_duration <= xmax - xmin <= args.max_duration:
                n_dropped += 1
                continue
            # Keep the utterance id prefixed by the speaker id so that Kaldi's
            # utt2spk stays sorted consistently with the utterance list.
            utt_id = f"{rec_id}_{round(xmin * 100):07d}_{round(xmax * 100):07d}"
            segments.append(f"{utt_id} {rec_id} {xmin:.3f} {xmax:.3f}")
            texts.append(f"{utt_id} {transcript}")
            utt2spk.append(f"{utt_id} {spk_id}")
            n_kept_here += 1
        if n_kept_here:
            wav_scp[rec_id] = f"{rec_id} {flac.resolve()}"

    if not segments:
        raise RuntimeError(f"no utterances selected for {out_dir}")

    for fname, lines in (
        ("wav.scp", sorted(wav_scp.values())),
        ("segments", sorted(segments)),
        ("text", sorted(texts)),
        ("utt2spk", sorted(utt2spk)),
    ):
        (out_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"{out_dir}: {len(segments)} utterances from {len(wav_scp)} recordings "
        f"({n_dropped} intervals dropped)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
