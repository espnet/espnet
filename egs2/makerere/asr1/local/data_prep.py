import csv
import glob
import os
import random
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python data_prep.py [root] [sph2pipe]")
        sys.exit(1)

    root = sys.argv[1]
    sph2pipe = sys.argv[2]

    all_audio_list = glob.glob(
        os.path.join(root, "makerere_radio_dataset/transcribed/dataset", "*.wav")
    )
    random.shuffle(all_audio_list)

    df = {}
    with open(
        "downloads/makerere_radio_dataset/transcribed/cleaned.csv", "r"
    ) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            df[row[0]] = row[2]

    for x in ["train", "test"]:
        if x == "train":
            audio_list = all_audio_list[0 : int(len(all_audio_list) * 0.8)]
        else:
            audio_list = all_audio_list[int(len(all_audio_list) * 0.8) :]
        # print('audio_list', len(audio_list))

        with open(os.path.join("data", x, "text"), "w") as text_f, open(
            os.path.join("data", x, "wav.scp"), "w"
        ) as wav_scp_f, open(os.path.join("data", x, "utt2spk"), "w") as utt2spk_f:
            i = 0
            for audio_path in audio_list:
                filename = os.path.basename(audio_path)
                speaker = filename.split(".")[0][8]
                if filename not in df or len(list(df[filename])) == 0:
                    continue
                transcript = df[filename]
                uttid = filename[:-4]  # "sk-o73a"
                wav_scp_f.write(f"{uttid} {audio_path}\n")
                text_f.write(f"{uttid} {transcript}\n")
                utt2spk_f.write(f"{uttid} {speaker}\n")
                i = i + 1
