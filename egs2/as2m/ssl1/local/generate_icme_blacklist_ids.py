import pandas as pd
import os
import glob
import json
from tqdm import tqdm

write_dir_path = (
    "/work/nvme/bbjs/sbharadwaj/espnet/egs2/as2m/ssl1/local/icme_blacklist_ids"
)

voxlingua = "/work/hdd/bbjs/shared/corpora/voxlingua107/*/*.wav"  # take first 11 characters of filename to create the youtube id
ub8k = "/work/hdd/bbjs/shared/corpora/wavcaps/json_files/blacklist/blacklist_exclude_ub8k_esc50_vggsound.json"  # has a list of freesound ids in json keyed by ub8k

# both these are csv files with a freesound_id column
fsdkaggle_train = (
    "/u/sbharadwaj/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv"
)
fsdkaggl_train = "/u/sbharadwaj/FSDKaggle2018.meta/train_post_competition.csv"

# take the file name as freesound id for both of these
fsd50k_dev = "/work/hdd/bbjs/shared/corpora/fsd50k/FSD50K.dev_audio/*.wav"
fsd50k_eval = "/work/hdd/bbjs/shared/corpora/fsd50k/FSD50K.eval_audio/*.wav"

# take the file name as freesound id
fma_small = "/work/hdd/bbjs/shared/corpora/fma/fma_small/*/*.mp3"

# Create output directory if it doesn't exist
os.makedirs(write_dir_path, exist_ok=True)


# Generate YouTube IDs file
def generate_youtube_ids():
    print("Generating YouTube IDs from VoxLingua dataset...")
    youtube_ids = []
    files = glob.glob(voxlingua)

    for file_path in tqdm(files, desc="Processing VoxLingua files"):
        filename = os.path.basename(file_path)
        youtube_id = filename[:11]  # First 11 characters
        youtube_ids.append(youtube_id)

    output_path = os.path.join(write_dir_path, "youtube_ids.txt")
    with open(output_path, "w") as f:
        for id in youtube_ids:
            f.write(f"{id}\n")

    print(f"✓ Successfully generated {len(youtube_ids)} YouTube IDs -> {output_path}")


# Generate FreeSound IDs file
def generate_freesound_ids():
    print("Generating FreeSound IDs from multiple sources...")

    # Load blacklist
    print("  Loading blacklist...")
    with open(ub8k, "r") as f:
        blacklist_data = json.load(f)
    blacklisted_ids = blacklist_data.get("FreeSound", [])
    print(f"  Found {len(blacklisted_ids)} blacklisted IDs")

    freesound_ids = set()  # Using set to avoid duplicates

    # Process FSDKaggle CSV files
    for csv_path in [fsdkaggle_train, fsdkaggl_train]:
        if os.path.exists(csv_path):
            print(f"  Processing {os.path.basename(csv_path)}...")
            df = pd.read_csv(csv_path)
            if "freesound_id" in df.columns:
                new_ids = df["freesound_id"].astype(str).tolist()
                freesound_ids.update(new_ids)
                print(f"  Added {len(new_ids)} IDs from {os.path.basename(csv_path)}")

    # Process FSD50K dev and eval WAV files
    for path_pattern in [fsd50k_dev, fsd50k_eval]:
        path_desc = "FSD50K dev" if "dev" in path_pattern else "FSD50K eval"
        files = glob.glob(path_pattern)
        print(f"  Processing {path_desc} ({len(files)} files)...")

        for file_path in tqdm(files, desc=f"  {path_desc}"):
            filename = os.path.basename(file_path)
            freesound_id = os.path.splitext(filename)[0]  # Remove extension
            freesound_ids.add(freesound_id)
        print(f"After adding fsd50k ids to freesound: {len(freesound_ids)}")

    for x in blacklisted_ids:
        freesound_ids.add(x)

    output_path = os.path.join(write_dir_path, "freesound_ids.txt")
    with open(output_path, "w") as f:
        for id in sorted(freesound_ids):
            f.write(f"FreeSound.{id}\n")

    print(
        f"✓ Successfully generated {len(freesound_ids)} FreeSound IDs -> {output_path}"
    )


# Generate FMA IDs file
def generate_fma_ids():
    print("Generating FMA IDs from FMA small dataset...")
    fma_ids = []
    files = glob.glob(fma_small)

    for file_path in tqdm(files, desc="Processing FMA files"):
        filename = os.path.basename(file_path)
        fma_id = os.path.splitext(filename)[0]  # Remove extension
        fma_ids.append(fma_id)

    output_path = os.path.join(write_dir_path, "fma_ids.txt")
    with open(output_path, "w") as f:
        for id in sorted(fma_ids):
            f.write(f"fma.{id}\n")

    print(f"✓ Successfully generated {len(fma_ids)} FMA IDs -> {output_path}")


# Run all functions
def main():
    print("\n=== Starting ID Generation Process ===\n")

    # Create output directory
    os.makedirs(write_dir_path, exist_ok=True)
    print(f"Output directory: {write_dir_path}")

    # Generate all IDs
    generate_youtube_ids()
    print()
    generate_freesound_ids()
    print()
    generate_fma_ids()

    print("\n=== ID Generation Complete ===")
    print(f"All ID files generated successfully in {write_dir_path}")


if __name__ == "__main__":
    main()
