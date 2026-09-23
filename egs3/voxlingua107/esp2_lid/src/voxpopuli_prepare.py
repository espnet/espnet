"""Download and cut VoxPopuli ASR recordings for LID evaluation."""

import ast
import csv
import gzip
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from egs3.voxlingua107.esp2_lid.dataset.builder import _ISO3_CODES
from egs3.voxlingua107.esp2_lid.src.download import DownloadProvider, DownloadRunner
from egs3.voxlingua107.esp2_lid.src.prepare_utils import parser, write_manifest
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.parallel import set_parallel

BASE_URL = "https://dl.fbaipublicfiles.com/voxpopuli"
LANGUAGES = "en de fr es pl it ro hu cs nl fi hr sk sl et lt".split()


def annotations(path, split, limit=None):
    """Select ASR utterances using the ESPnet2 OWSM rules.

    Args:
        path: Gzip-compressed annotation TSV path.
        split: Published train, dev or test split.
        limit: Optional maximum number of selected utterances.

    Returns:
        List of (annotation row, first VAD start, last VAD end) tuples.

    Example:
        >>> rows = annotations(Path("asr_en.tsv.gz"), "test", limit=2)
    """
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

    Args:
        source_dir: Corpus destination containing raw_audios/.
        sessions: Set of required recording IDs, including their year prefix.
        base_url: Official source URL, overridable for local fixtures.

    Returns:
        None. Writes selected raw_audios/original/<year>/*.ogg recordings.

    Example:
        >>> download_recordings(Path("download/voxpopuli"), {"2013-recording"})
    """
    missing = defaultdict(set)
    for session in sessions:
        year = session[:4]
        name = f"original/{year}/{session}_original.ogg"
        if not (source_dir / "raw_audios" / name).is_file():
            missing[year].add(name)
    tasks = [
        {
            "source_dir": str(source_dir),
            "year": year,
            "names": sorted(names),
            "base_url": base_url,
        }
        for year, names in sorted(missing.items())
    ]
    RecordingsRunner(
        DownloadProvider(tasks), output_dir=source_dir / ".recordings", resume=False
    )(range(len(tasks)))


class RecordingsRunner(BaseRunner):
    """Fetch each yearly archive once, with disjoint recordings per worker."""

    @staticmethod
    def forward(index, dataset, **env):
        """Read only the required members of one remote yearly tar file."""
        task = dataset[index]
        _download_year(
            Path(task["source_dir"]), task["year"], set(task["names"]), task["base_url"]
        )


def _download_year(source_dir, year, names, base_url):
    """Extract selected members from a single yearly archive using HTTP ranges."""
    import fsspec

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
    """Download annotations/audio and preserve train/dev/test split membership.

    Example:
        python -m egs3.voxlingua107.esp2_lid.src.voxpopuli_prepare \
            --source-dir download/voxpopuli --languages en --output-dir data/voxpopuli
    """
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
    argparser.add_argument(
        "--parallel-config",
        type=Path,
        help="ESPnet3 parallel YAML (env, n_workers, options)",
    )
    args = argparser.parse_args()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    set_parallel(
        OmegaConf.load(args.parallel_config)
        if args.parallel_config
        else OmegaConf.create({"env": "local"})
    )
    tasks = [
        {
            "url": f"{BASE_URL}/annotations/asr/asr_{language}.tsv.gz",
            "path": str(args.source_dir / "annotations" / f"asr_{language}.tsv.gz"),
        }
        for language in args.languages
    ]
    DownloadRunner(
        DownloadProvider(tasks),
        output_dir=args.source_dir / ".annotations",
        resume=False,
    )(range(len(tasks)))
    for split in args.splits:
        selected = {}
        for language in args.languages:
            path = args.source_dir / "annotations" / f"asr_{language}.tsv.gz"
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
                    ], str(audio), start, end, None

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
            parallel=True,
        )


if __name__ == "__main__":
    main()
