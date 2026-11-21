#!/usr/bin/env python3
"""Convert the existing FLEURS ASR recipe's TSV files to LID manifests."""

import argparse
import csv
from pathlib import Path


def prepare(source_dir, output_dir):
    import pycountry

    for split in ("train", "dev", "test"):
        utterances = {}
        with (source_dir / f"{split}.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                # create_dataset.py prefixes the transcript with [locale].
                locale = row["sentence"].split("]", 1)[0].removeprefix("[")
                code = locale.split("_", 1)[0]
                language = pycountry.languages.get(
                    **({"alpha_2": code} if len(code) == 2 else {"alpha_3": code})
                )
                if language is None:
                    raise ValueError(f"Unknown FLEURS language: {locale}")
                iso3 = language.alpha_3
                path = Path(row["path"]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                utt = f"{iso3}_{path.stem}"
                if utt in utterances:
                    raise ValueError(f"Duplicate FLEURS utterance: {utt}")
                utterances[utt] = (path, iso3)
        target = output_dir / f"{split}_fleurs_lang"
        target.mkdir(parents=True, exist_ok=True)
        with (
            (target / "wav.scp").open("w") as wavs,
            (target / "utt2lang").open("w") as labels,
        ):
            for utt, (path, lang) in sorted(utterances.items()):
                wavs.write(f"{utt} {path}\n")
                labels.write(f"{utt} {lang}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=Path("downloads/all"))
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir)
