import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Prepare kosp2e data for ESPnet.")
    parser.add_argument(
        "--kosp2e_root",
        type=Path,
        required=True,
        help="Path to the root of the kosp2e dataset, \
        containing wavs/ and downloads/metadata/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the output data directory.",
    )
    args = parser.parse_args()

    data_types = ["covid", "kss", "stylekqc", "zeroth"]
    wavs_base_path = args.kosp2e_root / "wavs"
    metadata_base_path = args.kosp2e_root / "downloads/metadata"

    # 1. Load all metadata and create lookup structures
    metadata = {}
    for dt in data_types:
        try:
            df_train = pd.read_excel(metadata_base_path / f"{dt}_train.xlsx").set_index(
                "id"
            )
            df_dev = pd.read_excel(metadata_base_path / f"{dt}_dev.xlsx").set_index(
                "id"
            )
            df_test = pd.read_excel(metadata_base_path / f"{dt}_test.xlsx").set_index(
                "id"
            )
            metadata[dt] = {"train": df_train, "val": df_dev, "test": df_test}
        except FileNotFoundError as e:
            print(f"Warning: Metadata for {dt} not found, skipping: {e}")

    # 2. Process data in a single pass
    file_contents = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    for data_type in tqdm(data_types, desc="Processing data types"):
        if data_type not in metadata:
            continue
        wav_dir = wavs_base_path / data_type
        if not wav_dir.is_dir():
            print(f"Warning: Directory not found for data_type {data_type}, skipping.")
            continue

        for wav_file in tqdm(
            sorted(os.listdir(wav_dir)), desc=f"Processing {data_type}"
        ):
            utt_id = wav_file.split(".")[0]

            split = None
            meta_df = None
            for s in ["train", "val", "test"]:
                if utt_id in metadata[data_type][s].index:
                    split = s
                    meta_df = metadata[data_type][s]
                    break

            if split is None:
                continue

            # IMPORTANT: Replace 'speaker_id' with the actual column name
            # for speaker ID in your metadata.
            # Using utt_id as spk_id is generally incorrect for multi-speaker data.
            row = meta_df.loc[utt_id]
            spk_id = row.get(
                "speaker_id", utt_id
            )  # Fallback to utt_id if no speaker_id column
            transcription = row["sentence"]
            wav_path = f"wavs/{data_type}/{wav_file}"

            file_contents[split]["text"].append(f"{utt_id} {transcription}\n")
            file_contents[split]["wav.scp"].append(f"{utt_id} {wav_path}\n")
            file_contents[split]["utt2spk"].append(f"{utt_id} {spk_id}\n")

    # 3. Write all files
    for split in ["train", "val", "test"]:
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["text", "wav.scp", "utt2spk"]:
            lines = sorted(file_contents[split][filename])
            with open(split_dir / filename, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"Wrote and sorted {split_dir / filename}")

        # 4. Generate spk2utt from utt2spk
        spk2utt = defaultdict(list)
        for line in file_contents[split]["utt2spk"]:
            utt, spk = line.strip().split(maxsplit=1)
            spk2utt[spk].append(utt)

        with open(split_dir / "spk2utt", "w", encoding="utf-8") as f:
            for spk in sorted(spk2utt.keys()):
                utts = " ".join(sorted(spk2utt[spk]))
                f.write(f"{spk} {utts}\n")
        print(f"Wrote {split_dir / 'spk2utt'}")


if __name__ == "__main__":
    main()
