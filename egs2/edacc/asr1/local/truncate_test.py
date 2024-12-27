import os
import shutil
import sys
import wave


def truncate_test_set(test_dir, utterance_splits):
    """
    Truncate specified utterances in a Kaldi test set directory.

    Args:
        test_dir (str): Path to the Kaldi test set directory.
        utterance_splits (dict):
            A dictionary where keys are original utterance IDs,
            and values are lists of tuples, each containing
            (new_utterance_id, start_time, end_time).

    Returns:
        None
    """
    # Paths to Kaldi files
    wav_scp_path = os.path.join(test_dir, "wav.scp")
    text_path = os.path.join(test_dir, "text")
    utt2spk_path = os.path.join(test_dir, "utt2spk")
    segment_path = os.path.join(test_dir, "segments")

    # Temporary storage for new data
    new_text = []
    new_utt2spk = []
    new_segment = []
    # Check existence of Kaldi files
    if (
        not os.path.exists(text_path)
        or not os.path.exists(utt2spk_path)
        or not os.path.exists(segment_path)
    ):
        print("Error: Kaldi files not found in the specified directory.")
        sys.exit(1)

    # update variables

    with open(text_path, "r") as text_file_input:
        for line in text_file_input:
            utter, _ = line.strip().split(maxsplit=1)
            if utter in utterance_splits:
                for (
                    new_utter,
                    _,
                    _,
                    new_txt,
                ) in utterance_splits[utter]:
                    new_text.append(f"{new_utter} {new_txt}\n")
            else:
                new_text.append(line)

    with open(utt2spk_path, "r") as utt2spk_file_input:
        for line in utt2spk_file_input:
            utter, spk = line.strip().split()
            if utter in utterance_splits:
                for (
                    new_utter,
                    _,
                    _,
                    _,
                ) in utterance_splits[utter]:
                    new_utt2spk.append(f"{new_utter} {spk}\n")
            else:
                new_utt2spk.append(line)

    with open(segment_path, "r") as segment_file_input:
        for line in segment_file_input:
            utter, id, start, end = line.strip().split()
            if utter in utterance_splits:
                for (
                    new_utter,
                    new_start,
                    new_end,
                    _,
                ) in utterance_splits[utter]:
                    new_segment.append(f"{new_utter} {id} {new_start} {new_end}\n")
            else:
                new_segment.append(line)

    # Write Kaldi files
    with open(text_path, "w") as text_file:
        text_file.writelines(new_text)

    with open(utt2spk_path, "w") as utt2spk_file:
        utt2spk_file.writelines(new_utt2spk)

    with open(segment_path, "w") as segment_file:
        segment_file.writelines(new_segment)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python truncate_test.py [test set directory]")
        sys.exit(1)

    test_dir = sys.argv[1]

    # Example input: specify utterance splits, below is an example,
    # the text should be filled according to the split,
    # and it's possible to add more utterances
    utterance_splits = {
        "EDACC-C34-A-EDACC-C34-000000008": [
            (
                "EDACC-C34-A-EDACC-C34-000000008_0",
                0.00,
                130.12,
                "xxxxxxxxxx",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000008_1",
                130.12,
                266.52,
                "xxxxxxxxx",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000008_2",
                266.62,
                419.24,
                "xxxxxxxx",
            ),
            (
                "EDACC-C34-A-EDACC-C34-000000008_3",
                419.24,
                535.96,
                "xxxxxxxxx",
            ),
        ]
    }

    truncate_test_set(test_dir, utterance_splits)
