"""Prepare ML-SUPERB 2.0 LID data using ESPnet2 ASR's label normalization."""

from egs3.voxlingua107.esp2_lid.src.prepare_utils import (
    hf_files,
    parquet_rows,
    parser,
    write_manifest,
)

REPOSITORY = "espnet/ml_superb_hf"
REVISION = "5a0634ecdbffdc99d8b55eb2b2486f3deb89691c"
LID_MAP = {"org_jpn": "jpn", "lga": "lug", "ory": "ori", "arb": "ara"}


def normalize_lid(split, uttid, lid):
    """Follow ESPnet2 ML-SUPERB label normalization, including dialect labels.

    Args:
        split: Published train, dev or dev_dialect split.
        uttid: Source utterance ID, used for the ms_speech dialect labels.
        lid: Source language label.

    Returns:
        Normalized language code, or None for excluded Norwegian train/dev rows.

    Example:
        >>> normalize_lid("dev", "sample", "org_jpn")
        'jpn'
    """
    lid = lid.strip()
    if split == "dev_dialect" and uttid.startswith("ms_speech_"):
        parts = uttid.split("_")
        if len(parts) >= 3 and parts[2] in {"tam", "tel", "guj"}:
            return parts[2]
    lid = LID_MAP.get(lid, lid)
    if split in {"train", "dev"} and lid in {"nno", "nob", "nor"}:
        return None
    return lid


def main():
    """Download selected splits and materialize their embedded audio.

    Example:
        python -m egs3.voxlingua107.esp2_lid.src.ml_superb2_prepare \
            --source-dir download/ml_superb2 --output-dir data/ml_superb2
    """
    argparser = parser(__doc__)
    argparser.add_argument("--source-dir", required=True)
    argparser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "dev", "dev_dialect"],
        default=["dev", "dev_dialect"],
    )
    argparser.add_argument("--revision", default=REVISION)
    args = argparser.parse_args()
    for split in args.splits:
        files, revision = hf_files(
            REPOSITORY,
            args.revision,
            args.source_dir,
            [f"data/{split}-*.parquet"],
            args.max_utterances is not None,
        )

        def examples():
            count = 0
            for row in parquet_rows(files):
                language = normalize_lid(split, row["id"], row["language"])
                if language is None:
                    continue
                if args.max_utterances is not None and count >= args.max_utterances:
                    break
                yield f"ml_superb2_{row['id']}", language, row[
                    "audio"
                ], None, None, None
                count += 1

        write_manifest(
            args.output_dir,
            split,
            examples(),
            {
                "repository": REPOSITORY,
                "revision": revision,
                "split": split,
                "max_utterances": args.max_utterances,
            },
        )


if __name__ == "__main__":
    main()
