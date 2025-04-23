import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

SPLIT_NAMES = ("train", "valid", "test")
INSTRUMENT_FAMILIES = [
    "bass",
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "organ",
    "reed",
    "string",
    "synth_lead",
    "vocal",
]

PITCH_FAMILIES = list(range(128))


def _get_args(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Prepare Nsynth dataset.")

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="The path of audio files",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="instrument",
        choices=["instrument", "pitch"],
        help="Type of tasks - instrument or pitch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="The path of output",
    )

    args = parser.parse_args(argv)

    assert args.root.exists()

    return args


def _read_data(args, split):
    root_dir = os.path.normpath(f"{args.root}/nsynth-{split}")

    if not os.path.isdir(args.root):
        raise ValueError(f"The given root path is not a directory. Got {args.root}")

    json_path = os.path.join(root_dir, "examples.json")
    if not os.path.isfile(json_path):
        raise ValueError("The given root path does not contain an examples.json")

    print(f"Loading NSynth data from split {split} at {args.root}")

    with open(json_path, "r") as fp:
        attrs = json.load(fp)

    label_key = "pitch" if args.task == "pitch" else "instrument_family_str"
    names_and_labels = []
    for k, v in tqdm(attrs.items(), desc="Loading data", total=len(attrs)):
        names_and_labels.append((k, v[label_key]))

    return names_and_labels


if __name__ == "__main__":
    print("Start preparing Nsynth dataset.")
    args = _get_args()
    print(args)

    args.output.mkdir(parents=True, exist_ok=True)

    files = {}
    for split in SPLIT_NAMES:
        (args.output / split).mkdir(exist_ok=True)
        files[split] = {
            "labels": open(args.output / split / "text", "w"),
            "path": open(args.output / split / "wav.scp", "w"),
            "dummy": open(args.output / split / "utt2spk", "w"),
        }

        names_and_labels = _read_data(args, split)
        missing_audio = 0

        for index, (name, label) in tqdm(
            enumerate(names_and_labels),
            desc="Writing kaldi data",
            total=len(names_and_labels),
        ):
            key = f"nsynth-{split}-{index:05d}-{name}"
            path = f"{args.root}/nsynth-{split}/audio/{name}.wav"
            path = os.path.abspath(path)
            if not os.path.exists(path):
                missing_audio += 1
                continue

            print(f"{key} {path}", file=files[split]["path"])
            print(f"{key} {label}", file=files[split]["labels"])
            print(f"{key} dummy", file=files[split]["dummy"])
        print(f"Missing audio files in {split}: {missing_audio}")

    print("Done preparing Nsynth dataset.")
