import os
import glob
import sys
from pydub import AudioSegment

# https://github.com/espnet/espnet/blob/master/egs/commonvoice/asr1/local/data_prep.pl#L57
def load_mp3(root):
    all_audio_list = glob.glob(
        os.path.join(root, "*", "*.mp3")
    )
    for audio_path in all_audio_list:
        sound = AudioSegment.from_mp3(audio_path)
        sound.export(os.path.join(os.path.dirname(audio_path), os.path.basename(audio_path + ".wav")), format="wav", bitrate=16)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_prep.py [root] [sph2pipe]")
        sys.exit(1)
    root = sys.argv[1]
    sph2pipe = sys.argv[2]

    load_mp3(root)

    all_audio_list = glob.glob(
        os.path.join(root, "*", "*.wav")
    )

    # TODO: write above files in kaldi format using tag files
    # utt2spk(related to uttid)

def parseTsv(root):
  # Read the TSV file 
  tsvPath = os.path.join(root, "*", "*.tsv")
  # eg："raw_30s_test.tsv"
  with open(tsvPath) as fr:
    for line in fr.readlines()[1:]:
      l=line.split('\t')
      tags = " ".join(l[5:])
      #print(tags)
      #uttid: 'track_0000214-artist_000014-album_000031-14/214'
      uttid = "-".join(l[0:4])[:-4]
      #print(uttid)
      with open('tags.text', 'a+') as fw:
        fw.write(tags)

    for x in ["train", "test"]:
        with open(os.path.join("data", x, "text"), "w") as text_f, open(
            os.path.join("data", x, "wav.scp"), "w"
        ) as wav_scp_f, open(
            os.path.join("data", x, "utt2spk"), "w"
        ) as utt2spk_f:

            for audio_path in all_audio_list:
                filename = os.path.basename(audio_path)     # "o73a.wav" etc
                speaker = os.path.basename(os.path.dirname(
                    audio_path))     # "lc", "sk", etc

                transcript = " ".join(list(filename[:-5]))  # "o73" -> "o 7 3"
                uttid = f"{speaker}-{filename[:-4]}"    # "sk-o73a"

                wav_scp_f.write(
                    f"{uttid} {sph2pipe} -f wav -p -c 1 {audio_path} |\n"
                )

                # TODO: write the other files in the Kaldi format
