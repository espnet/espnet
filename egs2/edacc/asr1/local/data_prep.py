import os
import re
import sys


def generate_data_files(target_root, edacc_root):

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

        os.makedirs(os.path.join(target_root, x), exist_ok=True)

        # process utt2spk
        utter_spk_map = {}
        with open(utt2spk_out, "w") as utt2spk_out:
            if os.path.exists(utt2spk):
                with open(utt2spk, "r") as utt2spk:
                    for line in utt2spk:
                        utter, spk = line.strip().split()
                        utter_spk_map[utter] = spk
                        utt2spk_out.write(f"{utter_spk_map[utter]}-{utter} {spk}\n")

        # process text
        with open(text_out, "w") as text_out:
            if os.path.exists(text):
                with open(text, "r") as text:
                    for line in text:
                        utter, txt = line.strip().split(maxsplit=1)
                        text_out.write(f"{utter_spk_map[utter]}-{utter} {txt}\n")

        # process segments and wav.scp
        wavID_set = set()
        with open(seg_out, "w") as seg_out, open(scp_out, "w") as scp_out:
            if os.path.exists(segments):
                with open(segments, "r") as segments:
                    for line in segments:
                        utter, wavID, others = line.strip().split(maxsplit=2)
                        seg_out.write(
                            f"{utter_spk_map[utter]}-{utter} {wavID} {others}\n"
                        )
                        audio_path = os.path.join(edacc_root, "data", f"{wavID}.wav")
                        if os.path.exists(audio_path) and wavID not in wavID_set:
                            scp_out.write(f"{wavID} {audio_path}\n")
                            wavID_set.add(wavID)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python data_prep.py [edacc download directory] [target directory]"
        )
        sys.exit(1)

    edacc_root = sys.argv[1]  # the dir should be "downloads/edacc_v1.0"
    target_root = sys.argv[2]  # the dir should be "data"

    generate_data_files(target_root, edacc_root)
