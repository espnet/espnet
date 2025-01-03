import os
import random
import re
import sys


def generate_data_files(
    target_root,
    edacc_root,
    segmented_audio_path,
    utter_number,
    segment_length=1883,
):

    for x in ["dev", "test"]:
        # output file directory
        text_out = os.path.join(target_root, x, "text")
        utt2spk_out = os.path.join(target_root, x, "utt2spk")
        scp_out = os.path.join(target_root, x, "wav.scp")
        seg_out = os.path.join(target_root, x, "segments")

        # input file directory
        text = os.path.join(edacc_root, x, "text")
        segments = os.path.join(edacc_root, x, "segments")
        utt2spk = os.path.join(edacc_root, x, "utt2spk")

        os.makedirs(
            os.path.join(target_root, x),
            exist_ok=True,
        )

        # process utt2spk
        utter_spk_map = {}
        utt_list = []
        with open(utt2spk_out, "w") as utt2spk_out:
            if os.path.exists(utt2spk):
                with open(utt2spk, "r") as utt2spk:
                    for line in utt2spk:
                        utter, spk = line.strip().split()
                        # process utter for C30
                        if "C30" in utter and int(utter[-9:]) >= 387:
                            new_number = int(utter[-9:]) - 387
                            utter = "EDACC-C30_P2-" + f"{new_number:09d}"
                        elif "C30" in utter and int(utter[-9:]) < 387:
                            utter = utter.replace("C30", "C30_P1")
                        utter_spk_map[utter] = spk
                        utt2spk_out.write(f"{utter_spk_map[utter]}-{utter} {spk}\n")
                        utt_list.append(utter)

        if x == "dev":
            random.seed(42)
            random.shuffle(utt_list)
            # select some utterances for training
            train_utter_list = utt_list[:utter_number]
            train_utterlist_path = os.path.join(target_root, "train_utterlist")
            with open(train_utterlist_path, "w") as f:
                for utter in train_utter_list:
                    f.write(f"{utter_spk_map[utter]}-{utter}\n")

            # select rest of utterances for validation
            valid_utter_list = utt_list[utter_number:]
            valid_utterlist_path = os.path.join(target_root, "valid_utterlist")
            with open(valid_utterlist_path, "w") as f:
                for utter in valid_utter_list:
                    f.write(f"{utter_spk_map[utter]}-{utter}\n")

        # process text
        with open(text_out, "w") as text_out:
            if os.path.exists(text):
                with open(text, "r") as text:
                    for line in text:
                        (
                            utter,
                            txt,
                        ) = line.strip().split(maxsplit=1)
                        # process utter for C30
                        if "C30" in utter and int(utter[-9:]) >= 387:
                            new_number = int(utter[-9:]) - 387
                            utter = "EDACC-C30_P2-" + f"{new_number:09d}"
                        elif "C30" in utter and int(utter[-9:]) < 387:
                            utter = utter.replace("C30", "C30_P1")
                        text_out.write(f"{utter_spk_map[utter]}-{utter} {txt}\n")

        # process segments and wav.scp
        wavID_set = set()
        with open(seg_out, "w") as seg_out, open(scp_out, "w") as scp_out:
            if os.path.exists(segments):
                with open(segments, "r") as segments:
                    for line in segments:
                        (
                            utter,
                            wavID,
                            start,
                            end,
                        ) = line.strip().split()
                        # process utter for C30
                        if "C30" in utter and int(utter[-9:]) >= 387:
                            new_number = int(utter[-9:]) - 387
                            utter = "EDACC-C30_P2-" + f"{new_number:09d}"
                            wavID = wavID.replace("C30", "C30_P2")
                            start = f"{float(start) - segment_length:.2f}"
                            end = f"{float(end) - segment_length:.2f}"
                            audio_path = os.path.join(
                                segmented_audio_path,
                                f"{wavID}.wav",
                            )
                        elif "C30" in utter and int(utter[-9:]) < 387:
                            utter = utter.replace("C30", "C30_P1")
                            wavID = wavID.replace("C30", "C30_P1")
                            audio_path = os.path.join(
                                segmented_audio_path,
                                f"{wavID}.wav",
                            )
                        else:
                            audio_path = os.path.join(
                                edacc_root,
                                "data",
                                f"{wavID}.wav",
                            )
                        seg_out.write(
                            f"{utter_spk_map[utter]}-{utter} {wavID} {start} {end}\n"
                        )
                        if os.path.exists(audio_path) and wavID not in wavID_set:
                            scp_out.write(f"{wavID} {audio_path}\n")
                            wavID_set.add(wavID)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(
            "Usage: python data_prep.py [edacc download directory]"
            "[target directory] [large audio path]"
        )
        sys.exit(1)

    edacc_root = sys.argv[1]  # the dir should be "downloads/edacc_v1.0"
    target_root = sys.argv[2]  # the dir should be "data"
    segmented_audio_path = sys.argv[
        3
    ]  # should be "downloads/edacc_v1.0/data/segmentation"
    generate_data_files(
        target_root,
        edacc_root,
        segmented_audio_path,
        segment_length=1883,
        utter_number=5000,
    )
