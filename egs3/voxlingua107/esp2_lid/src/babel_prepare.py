"""Prepare LID segments from a locally obtained Babel LDC corpus split."""

import re
import subprocess
import tempfile
from pathlib import Path

from egs3.voxlingua107.esp2_lid.src.prepare_utils import parser, write_manifest

# Nonlexical events omitted by Babel ASR's prepare_acoustic_training_data.pl.
NONLEXICAL = re.compile(
    r"\(\(\)\)|<(?:sta|male-to-female|female-to-male|lipsmack|breath|cough|"
    r"laugh|click|ring|dtmf|int|foreign|overlap|prompt|no-speech)>|~"
)
TIMESTAMP = re.compile(r"^\[([0-9.]+)\]$")


def segments(transcript):
    """Pair consecutive timestamps, keeping segments containing lexical speech.

    Args:
        transcript: Path to a Babel transcript with bracketed timestamps.

    Yields:
        Start and end times in seconds for segments with lexical speech.

    Example:
        >>> spans = list(segments(Path("transcription/recording.txt")))
    """
    start, text = None, ""
    with transcript.open(encoding="utf-8-sig") as source:
        for line in source:
            line = line.strip()
            timestamp = TIMESTAMP.fullmatch(line)
            if timestamp:
                end = float(timestamp[1])
                if start is not None and end < start:
                    raise ValueError(
                        f"Decreasing timestamp in {transcript}: {start} -> {end}"
                    )
                if (
                    start is not None
                    and end > start
                    and NONLEXICAL.sub("", text).strip()
                ):
                    yield start, end
                start, text = end, ""
            else:
                text = line


def main():
    """Read raw audio/transcription directories; no LDC download is attempted.

    Example:
        python -m egs3.voxlingua107.esp2_lid.src.babel_prepare \
            --source-dir /corpora/babel/dev --language asm --output-dir data/babel
    """
    argparser = parser(__doc__)
    argparser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Raw split directory containing audio/ and transcription/",
    )
    argparser.add_argument(
        "--language", required=True, help="ISO-639-3 label, e.g. asm"
    )
    argparser.add_argument("--split", default="dev")
    argparser.add_argument("--sph2pipe", default="sph2pipe")
    args = argparser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def examples():
        count = 0
        for transcript in sorted((args.source_dir / "transcription").glob("*.txt")):
            audio = args.source_dir / "audio" / f"{transcript.stem}.wav"
            with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
                if not audio.exists():
                    source = audio.with_suffix(".sph")
                    audio = Path(temporary) / "recording.wav"
                    subprocess.run(
                        [
                            args.sph2pipe,
                            "-f",
                            "wav",
                            "-p",
                            "-c",
                            "1",
                            str(source),
                            str(audio),
                        ],
                        check=True,
                    )
                for index, (start, end) in enumerate(segments(transcript)):
                    if args.max_utterances is not None and count >= args.max_utterances:
                        return
                    utt_id = f"babel_{args.language}_{transcript.stem}_{index:06d}"
                    yield utt_id, args.language, audio, start, end, 0
                    count += 1

    write_manifest(
        args.output_dir,
        args.split,
        examples(),
        {
            "source_dir": str(args.source_dir.resolve()),
            "language": args.language,
            "split": args.split,
            "max_utterances": args.max_utterances,
        },
    )


if __name__ == "__main__":
    main()
