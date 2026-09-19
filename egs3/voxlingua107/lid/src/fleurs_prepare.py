"""Prepare FLEURS LID audio from the source used by ESPnet2 FLEURS ASR."""

from pathlib import Path

from egs3.voxlingua107.lid.dataset.builder import _ISO3_CODES
from egs3.voxlingua107.lid.src.prepare_utils import (
    hf_files,
    parquet_rows,
    parser,
    write_manifest,
)

# FLEURS tags outside VoxLingua's source directory names, including legacy aliases.
LANGUAGES = {
    **_ISO3_CODES,
    "ff": "ful",
    "ga": "gle",
    "he": "heb",
    "ig": "ibo",
    "jv": "jav",
    "ky": "kir",
    "lg": "lug",
    "nb": "nor",
    "ny": "nya",
    "om": "orm",
    "or": "ori",
    "tn": "tsn",
    "wo": "wol",
    "xh": "xho",
    "zu": "zul",
    "fil": "tgl",
}
REPOSITORY = "google/xtreme_s"
REVISION = "e71d660cb63834e9aec8462f50fd0b00232b5aaf"


def language_code(tag):
    """Map the FLEURS locale to the ISO-639-3 labels used for LID."""
    code = tag.split("_")[0]
    return LANGUAGES.get(code, code)


def main():
    """Download selected languages and preserve the published split names."""
    argparser = parser(__doc__)
    argparser.add_argument("--source-dir", required=True)
    argparser.add_argument("--languages", nargs="+", default=["all"])
    argparser.add_argument(
        "--splits", nargs="+", choices=["train", "validation", "test"], default=["test"]
    )
    argparser.add_argument("--revision", default=REVISION)
    args = argparser.parse_args()
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPOSITORY, revision=args.revision)
    languages = args.languages
    if languages == ["all"]:
        languages = sorted(
            {
                s.rfilename.split("/")[0].removeprefix("fleurs.")
                for s in info.siblings
                if s.rfilename.startswith("fleurs.")
                and not s.rfilename.startswith("fleurs.all/")
            }
        )
    for split in args.splits:

        def examples():
            for tag in languages:
                files, _ = hf_files(
                    REPOSITORY,
                    info.sha,
                    args.source_dir,
                    [f"fleurs.{tag}/{split}/*.parquet"],
                    args.max_utterances is not None,
                )
                for index, row in enumerate(parquet_rows(files)):
                    if args.max_utterances is not None and index >= args.max_utterances:
                        break
                    # Sentence IDs repeat for different speakers; use the audio ID.
                    utt_id = f"fleurs_{tag}_{Path(row['audio']['path']).stem}"
                    yield utt_id, language_code(tag), row["audio"], None, None, None

        write_manifest(
            args.output_dir,
            split,
            examples(),
            {
                "repository": REPOSITORY,
                "revision": info.sha,
                "languages": languages,
                "split": split,
                "max_utterances_per_language": args.max_utterances,
            },
        )


if __name__ == "__main__":
    main()
