"""Download and cut VoxPopuli ASR recordings for LID evaluation."""

import ast
import csv
import gzip
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from egs3.voxlingua107.esp2_lid.dataset.builder import _ISO3_CODES
from egs3.voxlingua107.esp2_lid.src.prepare_utils import parser, write_manifest

BASE_URL = "https://dl.fbaipublicfiles.com/voxpopuli"
LANGUAGES = "en de fr es pl it ro hu cs nl fi hr sk sl et lt".split()


def download(url, path):
    """Retain completed downloads and replace only after a successful transfer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        temporary.replace(path)


def annotations(path, split, limit=None):
    """Select ASR utterances as in ESPnet2 OWSM's prepare_voxpopuli.py."""
    # Match the large annotation-field allowance in ESPnet2 OWSM.
    csv.field_size_limit(50000 * 1024 * 1024)
    selected = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="|"):
            if row["split"] != split:
                continue
            if not (row["original_text"].strip() or row["normed_text"].strip()):
                continue
            vad = ast.literal_eval(row["vad"])
            if not vad:
                continue
            start, end = vad[0][0], vad[-1][-1]
            if end - start < 0.1:
                continue
            selected.append((row, start, end))
            if limit is not None and len(selected) >= limit:
                break
    return selected


def download_recordings(source_dir, sessions, base_url=BASE_URL):
    """Fetch required members of official yearly tar files using HTTP ranges.

    Archive seeking skips unrelated recordings. Completed recordings are reused;
    interrupted members are retried. No archive paths are extracted unchecked.
    """
    import fsspec

    missing = defaultdict(set)
    for session in sessions:
        year = session[:4]
        name = f"original/{year}/{session}_original.ogg"
        if not (source_dir / "raw_audios" / name).is_file():
            missing[year].add(name)
    for year, names in sorted(missing.items()):
        url = f"{base_url}/audios/original_{year}.tar"
        print(f"Reading {url}: {len(names)} required recordings", flush=True)
        with fsspec.open(url, "rb", block_size=64 * 1024) as remote:
            with tarfile.open(fileobj=remote, mode="r:") as archive:
                for member in archive:
                    name = member.name.removeprefix("./")
                    if name not in names or not member.isfile():
                        continue
                    target = source_dir / "raw_audios" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".part")
                    with (
                        archive.extractfile(member) as audio,
                        temporary.open("wb") as output,
                    ):
                        shutil.copyfileobj(audio, output, length=16 * 1024 * 1024)
                    temporary.replace(target)
                    names.remove(name)
                    if not names:
                        break
        if names:
            raise FileNotFoundError(f"Recordings absent from {url}: {sorted(names)}")


def main():
    """Download annotations/audio and preserve train/dev/test split membership."""
    argparser = parser(__doc__)
    argparser.add_argument("--source-dir", type=Path, required=True)
    argparser.add_argument(
        "--languages", nargs="+", choices=LANGUAGES, default=LANGUAGES
    )
    argparser.add_argument(
        "--splits", nargs="+", choices=["train", "dev", "test"], default=["test"]
    )
    argparser.add_argument(
        "--skip-download-audio",
        action="store_true",
        help="Use existing raw_audios/original/<year>/*.ogg",
    )
    args = argparser.parse_args()
    for split in args.splits:
        selected = {}
        for language in args.languages:
            path = args.source_dir / "annotations" / f"asr_{language}.tsv.gz"
            download(f"{BASE_URL}/annotations/asr/{path.name}", path)
            selected[language] = annotations(path, split, args.max_utterances)
        if not args.skip_download_audio:
            # Languages share yearly archives; scan each archive only once.
            sessions = {r[0]["session_id"] for rows in selected.values() for r in rows}
            download_recordings(args.source_dir, sessions)

        def examples():
            for language, rows in selected.items():
                for row, start, end in rows:
                    session = row["session_id"]
                    audio = (
                        args.source_dir
                        / "raw_audios"
                        / "original"
                        / session[:4]
                        / f"{session}_original.ogg"
                    )
                    yield f"voxpopuli_{session}_{row['id_']}", _ISO3_CODES[
                        language
                    ], audio, start, end, None

        write_manifest(
            args.output_dir,
            split,
            examples(),
            {
                "base_url": BASE_URL,
                "languages": args.languages,
                "split": split,
                "max_utterances_per_language": args.max_utterances,
            },
        )


if __name__ == "__main__":
    main()
