"""Prepare the IAM handwriting dataset for ESPnet ASR training

Usage:
    python local/data_prep.py
        [--feature_dim 100]
        [--downsampling_factor 0.5]

Expects data to be in:
    downloads/
        lines.txt      # labels for IAM dataset (from ascii.tgz)
        lines/
            */*/*.png  # "lines" data for IAM dataset (from lines.tgz)
        train.txt      # id's for train/valid/test splits
        valid.txt
        test.txt

Required packages:
    Pillow
"""

import argparse
import os

import numpy as np
from PIL import Image

from espnet.utils.cli_writers import file_writer_helper


def prepare_text(lines_file_path, output_dir, split_ids):
    """Create 'text' file (map of ids to transcripts) in Kaldi format

    Parameters
    ----------
    lines_file_path : str
        The file path of the "lines.txt" label file of the IAM dataset
    output_dir : str
        The folder path for output, e.g. data/
    split_ids : list of str
        a list of ids to process, e.g. train/valid/test splits

    Returns
    -------
    skipped_ids : list of str
        a list of ids that were skipped,
        e.g. due to having an empty transcription
    """
    output_lines = []
    skipped_ids = []
    with open(lines_file_path) as lines_file:
        for line in lines_file.readlines():
            # skip comment lines
            if line[0] == "#":
                continue

            line_split = line.strip().split()

            # extract line ID
            line_id = line_split[0]

            if line_id in split_ids:
                # extract and format transcription into Kaldi style
                transcription = " ".join(line_split[8:])
                transcription = transcription.replace("|", " ")
                transcription = transcription.strip()

                if transcription == "":
                    print(f"Line {line_id} has an empty transcript, skipping")
                    skipped_ids.append(line_id)
                    continue

                output_lines.append(f"{line_id} {transcription}\n")

    output_lines.sort()

    with open(os.path.join(output_dir, "text"), "w") as out_file:
        out_file.writelines(output_lines)

    return skipped_ids


def prepare_utt2spk_spk2utt(output_dir, split_ids, ids_to_skip=[]):
    """Create (dummy) utt2spk and spk2utt files to satisfy Kaldi format

    Parameters
    ----------
    output_dir : str
        The folder path for output, e.g. data/
    split_ids : list of str
        a list of example ids to process, e.g. train/valid/test splits
    ids_to_skip : list of str
        A list of ids to exclude from the output,
        e.g. to ignore previously skipped ids due to empty transcript
    """
    output_lines = [f"{x} {x}\n" for x in split_ids if x not in ids_to_skip]
    output_lines.sort()

    with open(os.path.join(output_dir, "utt2spk"), "w") as out_file:
        out_file.writelines(output_lines)

    with open(os.path.join(output_dir, "spk2utt"), "w") as out_file:
        out_file.writelines(output_lines)


def prepare_feats(
    img_dir,
    output_dir,
    split_ids,
    ids_to_skip=[],
    feature_dim=100,
    downsampling_factor=0.5,
):
    """Create feats.scp file from OCR images

    Parameters
    ----------
    output_dir : str
        The folder path for output, e.g. data/
    split_ids : list of str
        a list of ids to process, e.g. train/valid/test splits
    ids_to_skip : list of str
        A list of ids to exclude from the output,
        e.g. to ignore previously skipped ids due to empty transcript
    feature_dim : int (default=100)
        The hidden dimension for each feature matrix,
        all images are resized to this height
    downsampling_factor : float (default=0.5)
        A multiplier for the width dimension (analogous to the 'time'
        dimension in ASR) of each image, used to reduce the length for
        faster training. Empirically, a value of 0.5 is found to achieve
        the best performance for this dataset
    """
    output_ark = os.path.join(output_dir, "feats.ark")
    output_scp = os.path.join(output_dir, "feats.scp")
    writer = file_writer_helper(
        wspecifier=f"ark,scp:{output_ark},{output_scp}",
        filetype="mat",
        write_num_frames=f"ark,t:{output_dir}/num_frames.txt",
        compress=False,
    )

    num_processed = 0
    total_length = 0

    for img_id in split_ids:
        if img_id in ids_to_skip:
            continue

        dir, subdir, index = img_id.split("-")
        img_path = os.path.join(
            img_dir, dir, f"{dir}-{subdir}", f"{dir}-{subdir}-{index}.png"
        )
        with Image.open(img_path) as img:
            # resize images to common height (feature_dim)
            # and downsample width by downsampling_factor
            aspect_ratio = img.width / img.height
            n_frames = int((aspect_ratio * feature_dim) * downsampling_factor)
            img = img.resize((n_frames, feature_dim))
            img_arr = np.array(img, dtype=np.float32).transpose()
            assert img_arr.shape[1] == feature_dim

            # write to data/output_dir/feats.scp
            writer[img_id] = img_arr

            # update counters for logging
            num_processed += 1
            total_length += img_arr.shape[0]

    print(
        f"Extracted features for {num_processed} examples "
        f"to {os.path.join(output_dir, 'feats.scp')},"
        f"average length is {total_length / num_processed:.02f}"
    )


if __name__ == "__main__":
    downloads_dir = "downloads/"
    data_dir = "data/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=100,
        help="Feature dimension to resize each image feature to",
    )
    parser.add_argument(
        "--downsampling_factor",
        type=float,
        default=0.5,
        help="Factor to downsample the length of each image feature to",
    )

    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        with open(os.path.join(downloads_dir, f"{split}.txt")) as split_file:
            split_ids = [line.strip() for line in split_file.readlines()]
            split_ids.sort()
            split_dir = os.path.join(data_dir, f"{split}/")
            os.makedirs(split_dir, exist_ok=True)

            lines_file_path = os.path.join(downloads_dir, "lines.txt")
            skipped_lines = prepare_text(lines_file_path, split_dir, split_ids)
            prepare_utt2spk_spk2utt(split_dir, split_ids, skipped_lines)
            prepare_feats(
                os.path.join(downloads_dir, "lines/"),
                split_dir,
                split_ids,
                skipped_lines,
                args.feature_dim,
                args.downsampling_factor,
            )
