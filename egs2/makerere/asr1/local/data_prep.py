import os
import glob
import sys
import csv
import random

if __name__ == "__main__":

    print('data prep ---- ____ -----')
    if len(sys.argv) != 3:
        print("Usage: python data_prep.py [root] [sph2pipe]")
        sys.exit(1)
    
    root = sys.argv[1]
    print('root', root)
    sph2pipe = sys.argv[2]

    all_audio_list = glob.glob(
        os.path.join(root, 'makerere_radio_dataset/transcribed/dataset', "*.wav")
    )
    random.shuffle(all_audio_list)

    df = {}
    with open('downloads/makerere_radio_dataset/transcribed/cleaned.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            df[row[0]] = row[2]  # Assuming first column is 'wav_filename' and second column is 'transcript'
    
    print({k: df[k] for k in list(df.keys())[:5]})

    for x in ["train", "test"]:
        if x=='train':
            audio_list = all_audio_list[0:int(len(all_audio_list)*0.8)]
        else:
            audio_list = all_audio_list[int(len(all_audio_list)*0.8):]
        # print('audio_list', len(audio_list))

        with open(os.path.join("data", x, "text"), "w") as text_f, open(
            os.path.join("data", x, "wav.scp"), "w"
        ) as wav_scp_f, open(
            os.path.join("data", x, "utt2spk"), "w"
        ) as utt2spk_f:
            i=0
            for audio_path in audio_list:
                print(i)
                print()
                print('audio_path', audio_path)
                filename = os.path.basename(audio_path)
                print('filename', filename)    # "o73a.wav" etc
                speaker = filename.split('.')[0][8]     # "lc", "sk", etc
                print('speaker', speaker) 
                # print('df[filename]', df[filename])
                if filename not in df or len(list(df[filename])) == 0:
                    continue
                transcript =  df[filename] #" ".join(list(filename[:-5]))  # "o73" -> "o 7 3"
                print('transcript', transcript)
                uttid = filename[:-4]    # "sk-o73a"
                print('uttid', uttid)
                wav_scp_f.write(f"{uttid} {audio_path}\n") 
                text_f.write(f"{uttid} {transcript}\n")
                utt2spk_f.write(f"{uttid} {speaker}\n")
                i=i+1

