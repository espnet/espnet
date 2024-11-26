import argparse
import glob
import os

from tqdm import tqdm

cache_size = 10000

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--original-dir", type=str, help="the directory of the original myst dataset"
)
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--is-wav", action="store_true")
args = parser.parse_args()

original_partition_list = ["train", "development", "test"]
data_partition_list = ["train", "dev", "test"]

# Iterate over each partition
for original_partition_name, data_partition_name in zip(
    original_partition_list, data_partition_list
):
    print(f"processing {data_partition_name} set")
    original_partition = os.path.join(args.original_dir, original_partition_name)
    data_partition = os.path.join(args.data_dir, data_partition_name)

    # Create the partition directory in the Kaldi structure
    os.makedirs(data_partition, exist_ok=True)

    # Initialize the Kaldi-style files
    text_file = os.path.join(data_partition, "text")
    utt2spk_file = os.path.join(data_partition, "utt2spk")
    spk2utt_file = os.path.join(data_partition, "spk2utt")
    wav_scp_file = os.path.join(data_partition, "wav.scp")

    # Empty the files if they already exist
    open(text_file, "w").close()
    open(utt2spk_file, "w").close()
    open(spk2utt_file, "w").close()
    open(wav_scp_file, "w").close()

    text_data = ""
    utt2spk_data = ""
    wav_scp_data = ""

    # Iterate overall audio files
    count = 0
    if args.is_wav:
        audio_files = glob.glob(f"{original_partition}/**/*.wav", recursive=True)
    else:
        audio_files = glob.glob(f"{original_partition}/**/*.flac", recursive=True)
    for audio_file in tqdm(audio_files):
        student_id = audio_file.split("/")[-3]
        session_dir = os.path.dirname(audio_file)
        audio_base = os.path.splitext(os.path.basename(audio_file))[0]
        transcription_file = os.path.join(session_dir, audio_base + ".trn")

        if os.path.isfile(transcription_file):
            with open(transcription_file, "r") as trn:
                transcription = trn.read().strip()
            text_data += f"{audio_base} {transcription}\n"
            utt2spk_data += f"{audio_base} {student_id}\n"
            wav_scp_data += f"{audio_base} {audio_file}\n"

        count += 1
        if count == cache_size:
            with open(text_file, "a") as tf, open(utt2spk_file, "a") as uf, open(
                wav_scp_file, "a"
            ) as wf:
                tf.write(text_data)
                uf.write(utt2spk_data)
                wf.write(wav_scp_data)
            text_data = ""
            utt2spk_data = ""
            wav_scp_data = ""
            count = 0

    # clear cache
    if count > 0:
        with open(text_file, "a") as tf, open(utt2spk_file, "a") as uf, open(
            wav_scp_file, "a"
        ) as wf:
            tf.write(text_data)
            uf.write(utt2spk_data)
            wf.write(wav_scp_data)
        text_data = ""
        utt2spk_data = ""
        wav_scp_data = ""
