import argparse
import csv
import sys

from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")

def _get_args(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Prepare FSD50K dataset.")
    
    parser.add_argument("--train", type=Path, required=True, help="Path to train/dev audio files.")
    parser.add_argument("--test", type=Path, required=True, help="Path to test audio files.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata directory.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output directory.")
    
    args = parser.parse_args(argv)

    # sanity check
    assert args.train.exists()
    assert args.test.exists()
    assert args.metadata.exists()

    return args


def _read_rows(args):
    # labels
    with open(args.metadata / "vocabulary.csv") as f:
        vocab = [row[1] for row in csv.reader(f)]
    assert len(set(vocab)) == len(vocab)
    vocab = set(vocab)

    # data
    rows = []

    # data: train+dev
    with open(args.metadata / "dev.csv") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row["fname"],
                "path": args.train / f"{row['fname']}.wav",
                "labels": row["labels"].replace(",", " "),
                "split": row["split"],
            })

    # data: test
    with open(args.metadata / "eval.csv") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row["fname"],
                "path": args.test / f"{row['fname']}.wav",
                "labels": row["labels"].replace(",", " "),
                "split": "test",
            })

    # sanity check
    labels = set()
    ids = []
    for row in rows:
        assert row["path"].exists()
        assert row["split"] in SPLIT_NAMES
        ids.append(row["id"])
        for label in row["labels"].split():
            assert label in vocab
            labels.add(label)

    assert len(vocab) == len(labels) == 200
    assert len(rows) == len(ids) == len(set(ids)) == 51197

    return rows


if __name__ == "__main__":
    print("Start preparing FSD50K dataset.")
    args = _get_args()
    print(args)

    rows = _read_rows(args)

    args.output.mkdir(parents=True)

    files = {}
    for split in SPLIT_NAMES:
        (args.output / split).mkdir()
        files[split] = {
            "labels": open(args.output / split / "text", "w"),
            "path": open(args.output / split / "wav.scp", "w"),
            "dummy": open(args.output / split / "utt2spk", "w"),
        }

    for index, row in enumerate(rows):
        key = f"fsd50k-{row['split']}-{index:05d}-{row['id']}"
        print(f"{key} {row['path']}", file=files[row["split"]]["path"])
        print(f"{key} {row['labels']}", file=files[row["split"]]["labels"])
        print(f"{key} dummy", file=files[row["split"]]["dummy"])

    print("Done preparing FSD50K dataset.")
