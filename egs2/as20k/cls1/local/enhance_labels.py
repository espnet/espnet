import json
import os
import csv

# VARIABLE
json_labels_path = "/work/hdd/bbjs/shared/corpora/audioset/enhanced_labels/balanced_train_data_type1_2_mean.json"
output_path = "/work/nvme/bbjs/sbharadwaj/espnet/egs2/audioverse/v1/dump/audioset20k.enhanced_mean/train/text"

# FIX
wav_scp_path = "/work/nvme/bbjs/sbharadwaj/espnet/egs2/audioverse/v1/data/audioset20k/train/wav.scp"
text_in_path = (
    "/work/nvme/bbjs/sbharadwaj/espnet/egs2/audioverse/v1/dump/audioset20k/train/text"
)
class_labels_path = "/work/hdd/bbjs/shared/corpora/audioset/class_labels_indices.csv"

# 1. Read wav.scp and map utterance_id to youtube_id
utterance_to_yt = {}
with open(wav_scp_path, "r", newline="") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        utterance_id, wav_path = parts[0], parts[1]
        # Extract youtube_id from wav path (filename without extension)
        youtube_id = os.path.splitext(os.path.basename(wav_path))[0]
        utterance_to_yt[utterance_id] = youtube_id

# 2. Build a map for mid2name from class_labels_indices.csv
mid_to_name = {}
with open(class_labels_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mid = row["mid"]
        # Keep quotes if they exist in the original
        name = row["display_name"]
        mid_to_name[mid] = name.replace(" ", "_")

# 3. Load the enhanced labels JSON
with open(json_labels_path, "r") as f:
    json_data = json.load(f)

# Create a map from youtube_id to labels (as MIDs)
yt_to_mids = {}
if "data" in json_data:
    for item in json_data["data"]:
        if "video_id" in item and "labels" in item:
            video_id = item["video_id"]
            labels_mids = item["labels"].split(",")
            yt_to_mids[video_id] = labels_mids

# Determine the number of spaces used in the original file
space_separator = " "  # Default
with open(text_in_path, "r", newline="") as f:
    for line in f:
        parts = line.rstrip("\n").split(" ", 1)
        if len(parts) > 1:
            # Count the number of spaces between utterance_id and first label
            original_line = line.rstrip("\n")
            pos = original_line.find(parts[0]) + len(parts[0])
            spaces = 0
            while pos < len(original_line) and original_line[pos] == " ":
                spaces += 1
                pos += 1
            if spaces > 0:
                space_separator = " " * spaces
            break

# 4. Process text.enhanced and write new file
with open(text_in_path, "r", newline="") as f_in, open(
    output_path, "w", newline=""
) as f_out:
    for line in f_in:
        original_line = line.rstrip("\n")
        if not original_line:
            f_out.write(line)  # Preserve exact empty lines
            continue

        parts = original_line.split(" ", 1)
        if len(parts) < 2:
            f_out.write(line)  # Keep exact line as is if not in expected format
            continue

        utterance_id = parts[0]
        current_labels = parts[1] if len(parts) > 1 else ""

        # Try to get new labels
        youtube_id = utterance_to_yt.get(utterance_id)
        if youtube_id and youtube_id in yt_to_mids:
            # Convert MIDs to names
            new_labels = []
            for mid in yt_to_mids[youtube_id]:
                if mid in mid_to_name:
                    new_labels.append(mid_to_name[mid])

            if new_labels:
                # Write with new labels, maintain original format
                new_line = f"{utterance_id}{space_separator}{' '.join(new_labels)}\n"
                f_out.write(new_line)
                continue

        # If we couldn't find new labels, write the exact original line
        f_out.write(line)

print(f"Conversion complete. New file written to {output_path}")
