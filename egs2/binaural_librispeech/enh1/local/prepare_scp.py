import argparse
import glob
import os


def prepare_scp(clean_dir, noisy_dir, output_clean_scp, output_noisy_scp):
    with open(output_clean_scp, "w") as clean_scp, open(
        output_noisy_scp, "w"
    ) as noisy_scp:
        for split in os.listdir(clean_dir):
            split_path_clean = os.path.join(clean_dir, split)
            split_path_noisy = os.path.join(noisy_dir, split)

            if os.path.isdir(split_path_clean) \
                and os.path.isdir(split_path_noisy):
                for speaker in os.listdir(split_path_clean):
                    speaker_path_clean = os.path.join(split_path_clean, speaker)
                    speaker_path_noisy = os.path.join(split_path_noisy, speaker)

                    if os.path.isdir(speaker_path_clean) and os.path.isdir(
                        speaker_path_noisy
                    ):
                        for chapter in os.listdir(speaker_path_clean):
                            chapter_path_clean = os.path.join(
                                speaker_path_clean, chapter
                            )
                            chapter_path_noisy = os.path.join(
                                speaker_path_noisy, chapter
                            )

                            if os.path.isdir(chapter_path_clean) \
                                and os.path.isdir(
                                chapter_path_noisy
                            ):
                                for utterance in glob.glob(
                                    os.path.join(chapter_path_clean, "*.flac")
                                ):
                                    utterance_id = os.path.basename(utterance)
                                    clean_file_path = os.path.join(
                                        chapter_path_clean, utterance_id
                                    )
                                    noisy_file_path = os.path.join(
                                        chapter_path_noisy, utterance_id
                                    )

                                    clean_scp.write(
                                        f"{utterance_id} {clean_file_path}\n"
                                    )
                                    noisy_scp.write(
                                        f"{utterance_id} {noisy_file_path}\n"
                                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare SCP files for the enh1 task in ESPnet."
    )
    parser.add_argument(
        "clean_data_directory", type=str, 
        help="Path to the clean data directory."
    )
    parser.add_argument(
        "noisy_data_directory", type=str, 
        help="Path to the noisy data directory."
    )
    parser.add_argument(
        "output_clean_scp", type=str, 
        help="Output path for the clean SCP file."
    )
    parser.add_argument(
        "output_noisy_scp", type=str, 
        help="Output path for the noisy SCP file."
    )

    args = parser.parse_args()

    prepare_scp(
        args.clean_data_directory,
        args.noisy_data_directory,
        args.output_clean_scp,
        args.output_noisy_scp,
    )
