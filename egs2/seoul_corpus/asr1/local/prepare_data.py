#!/usr/bin/env python3

"""Convert the Seoul Corpus (flac + Praat TextGrid) into Kaldi-style data dirs.

The Seoul Corpus ships 240 ten-minute flac files (40 speakers x 6 sessions) and
one Praat TextGrid per flac.  Each TextGrid holds seven interval tiers::

    1 phoneme       2,3 pWord.prono.   4 utt.prono.
                    5,6 pWord.ortho.   7 utt.ortho.

We read an utterance-level tier ("utt.ortho." by default, i.e. the orthographic
transcription; "utt.prono." gives the pronounced/colloquial form).

The corpus annotates non-speech events on that same tier, each in its own
interval, and they are kept as transcription targets rather than thrown away:
``<SIL>``, ``<NOISE>``, ``<VOCNOISE>``, ``<LAUGH>``, ``<UNKNOWN>`` and
``<PRIVATE.INFO>`` all survive into the text as single tokens (register them with
asr.sh's --bpe_nlsyms so sentencepiece keeps them atomic).  ``<LAUGH-그래서>``
means 그래서 was said while laughing and becomes ``<LAUGH> 그래서``.

``<IVER>`` is the interviewer talking, which is not this speaker's transcript, so
those intervals are dropped -- and because the tags around them are kept, an
``<IVER>`` interval is what ends an utterance.  Utterances therefore run from one
interviewer turn to the next, holding the speech and the non-speech events in
between in tier order.

Those spans reach 376 s, which is far too long to train on, so a span longer than
``--max_duration`` is cut into pieces instead of being thrown away.  The cuts are
placed on a ``<SIL>`` interval wherever one is available inside the window -- a
silence is where a turn can be broken without splitting a phrase -- and fall back
to the last interval boundary that fits when the window holds no silence.  Either
way a cut lands on an annotated boundary, never inside a word, and no audio is
lost.
"""

import argparse
import re
import sys
from pathlib import Path

# Non-speech events that stay in the transcript, one token each.
SPECIAL_TAGS = (
    "<SIL>",
    "<NOISE>",
    "<VOCNOISE>",
    "<LAUGH>",
    "<UNKNOWN>",
    "<PRIVATE.INFO>",
)
# The interviewer's turns are not this speaker's transcript: dropped, and they
# are what separates one utterance from the next.
BOUNDARY_TAGS = {"<IVER>", "<IVER-NOISE>"}
# "<LAUGH-그래서>" -> "<LAUGH> 그래서" (speech produced while laughing).
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
    """Turn one interval label into transcript tokens.

    Args:
        label: Raw text of the TextGrid interval.

    Returns:
        A list of tokens, or ``None`` if the interval is not transcribable (the
        interviewer talking, an empty label, or a tag we cannot represent).
    """
    label = LAUGH_WORD.sub(r"<LAUGH> \1", label).strip()
    if not label:
        return None
    tokens = label.split()
    if any(tok in BOUNDARY_TAGS for tok in tokens):
        return None
    # Any other bracketed token is a tag this recipe does not know how to spell,
    # so end the utterance rather than emit a transcript with a hole in it.
    for tok in tokens:
        if tok.startswith("<") and tok not in SPECIAL_TAGS:
            return None
    return tokens


def join_tokens(tokens):
    """Join interval labels into one transcript, with no space before a tag.

    ``["<SIL>", "그래서", "<VOCNOISE>"]`` becomes ``"<SIL> 그래서<VOCNOISE>"``,
    which BPE-encodes as ``["▁", "<SIL>", "▁그래서", "<VOCNOISE>"]``.

    sentencepiece splits the text at every ``--bpe_nlsyms`` symbol, so a space
    in front of a tag survived as a bare "▁" piece of its own: 46623 of them,
    which put "▁" at 21.7% of every target token and the three commonest tokens
    at 31.2% -- the mass valid ``acc`` pinned itself to while the model learned
    nothing but the label prior.  Dropping just that space removes 97% of them
    and takes the training text from 449611 pieces to 399848.

    The space *after* a tag is kept on purpose.  Removing both sides also works
    but costs more than it saves: a word that follows a tag then loses its
    word-initial form (``<VOCNOISE>서울`` splits into characters rather than
    ``▁서울``), which was 38746 pieces and left the same word tokenized two
    ways.  The one "▁" that stays in front of a tag is sentencepiece's dummy
    prefix on an utterance that opens with one (1413 of 7284); turning that off
    would strip the word-initial marker from every word in the corpus.

    Args:
        tokens: Tokens of one utterance, in tier order.

    Returns:
        The transcript as a single string.
    """
    parts = []
    for tok in tokens:
        if parts and tok not in SPECIAL_TAGS:
            parts.append(" ")
        parts.append(tok)
    return "".join(parts)


def split_span(items, max_duration):
    """Cut one interviewer-to-interviewer span into pieces of at most a length.

    Args:
        items: ``(xmin, xmax, tokens)`` tuples of one span, in tier order.
        max_duration: Longest piece to emit (sec).

    Returns:
        A list of ``(xmin, xmax, transcript)`` tuples covering the whole span.
    """
    out = []
    start = 0
    while start < len(items):
        end = start
        while (
            end + 1 < len(items) and items[end + 1][1] - items[start][0] <= max_duration
        ):
            end += 1
        cut = end
        if end + 1 < len(items):
            # A cut is needed: prefer to end the piece on a silence.
            for k in range(end, start, -1):
                if items[k][2] == ["<SIL>"]:
                    cut = k
                    break
        out.append(
            (
                items[start][0],
                items[cut][1],
                join_tokens([tok for it in items[start : cut + 1] for tok in it[2]]),
            )
        )
        start = cut + 1
    return out


def build_utterances(intervals, max_duration):
    """Group the intervals of one tier into utterances.

    Args:
        intervals: ``(xmin, xmax, text)`` tuples of the utterance tier.
        max_duration: Longest utterance to emit (sec); longer spans are cut.

    Returns:
        A ``(utterances, n_dropped)`` pair, where ``utterances`` is a list of
        ``(xmin, xmax, transcript)`` tuples in tier order.
    """
    utterances = []
    n_dropped = 0
    span = []
    for xmin, xmax, label in intervals:
        tokens = normalize(label)
        if tokens is None:
            n_dropped += 1
            if span:
                utterances += split_span(span, max_duration)
                span = []
            continue
        span.append((xmin, xmax, tokens))
    if span:
        utterances += split_span(span, max_duration)
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
    # Not required, so that --print_special_tags can be used on its own; main()
    # checks them instead.
    parser.add_argument("--sound_dir", help="directory with *.flac")
    parser.add_argument("--label_dir", help="directory with *.TextGrid")
    parser.add_argument("--out_dir", help="output data directory")
    parser.add_argument(
        "--speakers",
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
        "--max_duration",
        type=float,
        default=30.0,
        help="cut interviewer-to-interviewer spans longer than this (sec)",
    )
    parser.add_argument(
        "--print_special_tags",
        action="store_true",
        help="print the special tags, one per line, and exit; local/data.sh "
        "writes them to data/nlsyms.txt so that run.sh does not have to repeat "
        "the list and asr.sh can drop them before scoring CER and WER",
    )
    return parser


def main():
    args = get_parser().parse_args()
    if args.print_special_tags:
        print("\n".join(SPECIAL_TAGS))
        return
    for name in ("sound_dir", "label_dir", "out_dir", "speakers"):
        if getattr(args, name) is None:
            get_parser().error(f"--{name} is required")
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
            tiers[args.tier], args.max_duration
        )
        n_dropped += n_dropped_here

        n_kept_here = 0
        for xmin, xmax, transcript in utterances:
            if xmax - xmin < args.min_duration:
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
