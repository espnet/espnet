import os
import random
import sys
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
import tempfile

import numpy as np
from tqdm import tqdm

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]


def read_data_file(filename, mid2name):
    data = []
    if "unbalanced_train_segments" in filename:
        wav_directory = "unbalanced_wav"
    elif "balanced_train_segments" in filename:
        wav_directory = "balance_wav"
    elif "eval_segments" in filename:
        wav_directory = "eval_wav"
    else:
        raise ValueError("Unknown data file")
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Reading data files"):
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.strip().split(",", maxsplit=3)
            # Extract the fields
            yt_id = parts[0]
            start_seconds = float(parts[1])
            end_seconds = float(parts[2])
            labels = parts[3].replace('"', "").split(",")
            labels = [mid2name[label.strip()] for label in labels]
            data.append(
                {
                    "yt_id": yt_id,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "labels": labels,
                    "wav_directory": wav_directory,
                }
            )
    return data


def read_mid2name_map(mid2name_file):
    mid2name = {}
    unique_names = set()
    with open(mid2name_file, "r") as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Reading mid2name map"):
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.strip().split(",", maxsplit=2)
            label_name = parts[2].replace('"', "").strip().replace(" ", "_")
            if label_name in unique_names:
                print(f"Warning: Duplicate label name {label_name}")
            unique_names.add(label_name)
            mid2name[parts[1]] = label_name
    return mid2name


def generate_sampling_weights(train_set, mid2name):
    label_count = {name: 0 for name in mid2name.values()}
    for item in train_set:
        for label in item["labels"]:
            label_count[label] += 1
    label_weight = {
        name: 1000.0 / (count + 0.01) for name, count in label_count.items()
    }
    instance_wise_sampling_weights = np.zeros(len(train_set))
    for idx, item in enumerate(train_set):
        for label in item["labels"]:
            instance_wise_sampling_weights[idx] += label_weight[label]
    return instance_wise_sampling_weights


def process_chunk(chunk, name, include_weights, data_sampling_weights):
    """
    Process a chunk of (uttid, item) pairs.
    Write the outputs to temporary files and return their paths along with the count of missing wav files.
    """
    temp_dir = tempfile.mkdtemp(prefix="process_dataset_")
    pid = os.getpid()
    temp_paths = {
        "text": os.path.join(temp_dir, f"text_{pid}_{id(chunk)}.tmp"),
        "wav": os.path.join(temp_dir, f"wav_{pid}_{id(chunk)}.tmp"),
        "utt2spk": os.path.join(temp_dir, f"utt2spk_{pid}_{id(chunk)}.tmp"),
    }
    if include_weights:
        temp_paths["weight"] = os.path.join(temp_dir, f"weight_{pid}_{id(chunk)}.tmp")

    # Open all temporary files for writing.
    file_handles = {key: open(path, "w") for key, path in temp_paths.items()}

    missing = 0
    for uttid, item in chunk:
        wav_directory = item["wav_directory"]
        wav_path = os.path.join(DATA_READ_ROOT, wav_directory, item["yt_id"] + ".wav")
        if not os.path.exists(wav_path):
            missing += 1
            continue

        # Create a unique utterance ID and formatted output lines.
        utt_id = f'{name}-{uttid}-{item["yt_id"]}'
        text_line = f"{utt_id} {' '.join(item['labels'])}\n"
        wav_line = f"{utt_id} {wav_path}\n"
        utt2spk_line = f"{utt_id} {utt_id}\n"

        file_handles["text"].write(text_line)
        file_handles["wav"].write(wav_line)
        file_handles["utt2spk"].write(utt2spk_line)
        if include_weights:
            weight_line = f"{utt_id} {data_sampling_weights[uttid]}\n"
            file_handles["weight"].write(weight_line)

    # Close all temporary files.
    for fh in file_handles.values():
        fh.close()

    return temp_paths, missing


def chunkify(data, chunk_size):
    """Yield successive chunks of size 'chunk_size' from the data."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def process_dataset_parallel(
    dataset,
    name,
    text_write_path,
    wav_scp_write_path,
    utt2spk_write_path,
    sample_weight_path=None,
    data_sampling_weights=None,
    num_workers=None,
    chunk_size=1000,
):
    """
    Process the dataset in parallel using chunked processing with temporary files.

    Each chunk is processed by a worker that writes its results to temporary files.
    After processing, the main process merges these temporary files into the final output files.

    Args:
        dataset: The dataset to process.
        name: Dataset split name (e.g., 'train', 'test').
        text_write_path: Final output path for text.
        wav_scp_write_path: Final output path for wav.scp.
        utt2spk_write_path: Final output path for utt2spk.
        sample_weight_path: Final output path for sample weights (optional).
        data_sampling_weights: Dictionary of sampling weights if sample_weight_path is provided.
        num_workers: Number of parallel workers (None uses all available cores).
        chunk_size: Number of items per chunk.

    Returns:
        The total number of missing wav files.
    """
    output_paths = {
        "text": text_write_path,
        "wav": wav_scp_write_path,
        "utt2spk": utt2spk_write_path,
    }
    include_weights = (
        sample_weight_path is not None and data_sampling_weights is not None
    )
    if include_weights:
        output_paths["weight"] = sample_weight_path

    # Clear or create the final output files.
    for path in output_paths.values():
        with open(path, "w") as f:
            pass

    # Enumerate dataset items so each item has an index.
    indexed_items = list(enumerate(dataset))
    chunks = list(chunkify(indexed_items, chunk_size))

    total_missing = 0
    temp_files_results = []

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    process_chunk,
                    [
                        (chunk, name, include_weights, data_sampling_weights)
                        for chunk in chunks
                    ],
                ),
                total=len(chunks),
                desc=f"Processing {name} set in chunks",
            )
        )
        for temp_paths, missing in results:
            total_missing += missing
            temp_files_results.append(temp_paths)

    # Merge temporary files into final output files.
    for key, final_path in output_paths.items():
        with open(final_path, "a") as fout:
            for temp_dict in temp_files_results:
                temp_file = temp_dict.get(key)
                if temp_file and os.path.exists(temp_file):
                    with open(temp_file, "r") as fin:
                        fout.write(fin.read())
                    os.remove(temp_file)

    print(f"Missing {total_missing} wav files in {name} set.")
    return total_missing


if __name__ == "__main__":
    mid2name = read_mid2name_map(
        os.path.join(DATA_READ_ROOT, "class_labels_indices.csv")
    )
    eval_set = read_data_file(
        os.path.join(DATA_READ_ROOT, "eval_segments.csv"), mid2name
    )
    train_set = read_data_file(
        os.path.join(DATA_READ_ROOT, "unbalanced_train_segments.csv"), mid2name
    )
    train_set_bal = read_data_file(
        os.path.join(DATA_READ_ROOT, "balanced_train_segments.csv"), mid2name
    )
    train_set = train_set + train_set_bal

    # Create validation split from eval
    # Since the code for BEATs evals is not public, it is hard to estimate how they create
    # val set. However, it seems like AST is using all of the training data. To replicate this
    # setup we do not use remove any data from train set and use 10% of eval set as val set.
    # AST- https://github.com/YuanGongND/ast/tree/master
    random.seed(42)
    random.shuffle(eval_set)
    total_len = len(eval_set)
    val_len = total_len // 10
    val_set = eval_set[:val_len]

    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    print(f"Eval set size: {len(eval_set)}")

    data_sampling_weights = generate_sampling_weights(train_set, mid2name)

    for dataset, name in [(train_set, "train"), (val_set, "val"), (eval_set, "eval")]:
        missing_wav_file = 0
        text_write_path = os.path.join(DATA_WRITE_ROOT, name, "text")
        wav_scp_write_path = os.path.join(DATA_WRITE_ROOT, name, "wav.scp")
        utt2spk_write_path = os.path.join(DATA_WRITE_ROOT, name, "utt2spk")

        os.makedirs(os.path.dirname(text_write_path), exist_ok=True)
        process_dataset_parallel(
            dataset,
            name,
            text_write_path,
            wav_scp_write_path,
            utt2spk_write_path,
            sample_weight_path=(
                os.path.join(DATA_WRITE_ROOT, "utt2weight") if name == "train" else None
            ),
            data_sampling_weights=data_sampling_weights,
            num_workers=max(cpu_count() - 2, 1),
        )
