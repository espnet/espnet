# Prepare Kaldi-style data directory for AMI diarization task
import argparse
import os
import re
import yaml

def float2str(number, size=6):
    number = str(int(number * 100))  # convert to integer after multiplying by 100
    return number.zfill(size)  # pad with zeros to the left to ensure the string is of the desired size


def gen_utt_id(wav_id: str, utt_start_time: float, utt_end_time: float) -> str:
    return f"{wav_id}_{float2str(utt_start_time)}_{float2str(utt_end_time)}"


def prepare_kaldi_files(
    config_path: str, mic_type: str, if_mini: str, 
    sound_type: str, dataset_type: str, kaldi_files_base_dir: str
) -> None:
    # dataset_type: "train", "dev", "test"
    if if_mini == "true":
        if_mini = True
    elif if_mini == "false":
        if_mini = False
    else:
        raise ValueError("if_mini must be 'true' or 'false'")
    
    if sound_type == "word_and_vocalsounds":
        assert mic_type == "ihm", "Only data with ihm microphone type is available for word_and_vocalsounds sound type."
        assert not if_mini, "Only full dataset is available for word_and_vocalsounds sound type."
    if sound_type == "only_words":
        assert not if_mini, "Only full dataset is available for only_words sound type."

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset = "AMI-SDM" if mic_type == "sdm" else "AMI"
    task = "SpeakerDiarization"
    task_option = "mini" if if_mini else sound_type
    kaldi_files_dir = os.path.join(kaldi_files_base_dir, dataset_type)

    if not os.path.exists(kaldi_files_dir):
        os.makedirs(kaldi_files_dir)

    wavscp = open(os.path.join(kaldi_files_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(kaldi_files_dir, "utt2spk"), "w", encoding="utf-8")
    segments = open(os.path.join(kaldi_files_dir, "segments"), "w", encoding="utf-8")

    databases = config["Databases"]
    protocols = config["Protocols"]

    # wav_id_txt is the txt file contains a list of wav_ids, uri with the same meaning of wav_id
    wav_id_txt = protocols[dataset][task][task_option][dataset_type]["uri"] 

    # wav_path_template is like amicorpus/{uri}/audio/{uri}.Mix-Headset.wav
    wav_path_template = databases[dataset] 

    # rttm_path_template is like ami_diarization_setup/only_words/rttms/train/{uri}.rttm
    rttm_path_template = protocols[dataset][task][task_option][dataset_type]["annotation"] 

    # lab_path_template is like ami_diarization_setup/only_words/labs/train/{uri}.lab
    # lab file contain the utterance start and end time information
    # like: 5.57 6.01 speech, according to this file, we can get utt_id
    lab_path_template = protocols[dataset][task][task_option][dataset_type]["lab"]

    # we may not use uem file when creating kaldi-style data directory
    uem_path_template = protocols[dataset][task][task_option][dataset_type]["annotated"]

    wav_ids = []
    with open(wav_id_txt, "r") as f:
        for line in f:
            wav_ids.append(line.strip())

    for wav_id in wav_ids:
        ### ======== Prepare segments, utt2spk ======== ###
        rttm_path = rttm_path_template.format(uri=wav_id)
        lab_path = lab_path_template.format(uri=wav_id)

        # read rttm file, 
        # transform it to [{"spk_id": str, "spk_start_time": float, "spk_end_time": float}, ...]
        rttm_dict = []
        with open(rttm_path, "r") as f:
            for line_id, line in enumerate(f):
                sps = re.split(" +", line.rstrip())
                assert len(sps) == 10, f"Error in {rttm_path} at line {line_id + 1}"
                label_type, wav_id, channel, spk_start_time, spk_duration, _, _, spk_id, _, _ = sps
                assert label_type == "SPEAKER", f"Error in {rttm_path} at line {line_id + 1}"
                rttm_dict.append({
                    "spk_id": spk_id,
                    "spk_start_time": float(spk_start_time),
                    "spk_end_time": float(spk_start_time) + float(spk_duration)
                })
        
        # read lab file, 
        # transform it to [{"utt_id": str, "utt_start_time": float, "utt_end_time": float}, ...]
        lab_dict = []
        with open(lab_path, "r") as f:
            for line_id, line in enumerate(f):
                sps = re.split(" +", line.rstrip())
                assert len(sps) == 3, f"Error in {lab_path} at line {line_id + 1}"
                utt_start_time, utt_end_time, utt_label = sps
                assert utt_label == "speech", f"Error in {lab_path} at line {line_id + 1}"
                lab_dict.append({
                    "utt_id": gen_utt_id(wav_id, float(utt_start_time), float(utt_end_time)),
                    "utt_start_time": float(utt_start_time),
                    "utt_end_time": float(utt_end_time)
                })
        
        # match the utterance and speaker, write to segments, utt2spk
        # Note: only create utt2spk, spk2utt can be created through utt2spk by 
        #       `utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt``
        spk_count = 0
        for utt in lab_dict:
            utt_id = utt["utt_id"]
            spks_in_utt = []

            while rttm_dict[spk_count]["spk_start_time"] >= utt["utt_start_time"] and rttm_dict[spk_count]["spk_end_time"] <= utt["utt_end_time"]:
                spk_id = rttm_dict[spk_count]["spk_id"]
                spk_start_time = rttm_dict[spk_count]["spk_start_time"]
                spk_end_time = rttm_dict[spk_count]["spk_end_time"]

                segments.write(f"{utt_id} {wav_id} {spk_start_time} {spk_end_time}\n")
                spks_in_utt.append(spk_id)

                spk_count += 1
                if spk_count >= len(rttm_dict):
                    break

            # assert len(spks_in_utt) > 0, f"Error in {lab_path}, no speaker found for utterance {utt_id}"

            spks_in_utt_str = " ".join(spks_in_utt)
            utt2spk.write(f"{utt_id} {spks_in_utt_str}\n")

        ### ======== Prepare wav.scp ======== ###
        wav_path = wav_path_template.format(uri=wav_id)
        wavscp.write(f"{wav_id} {wav_path}\n")
    
    wavscp.close()
    utt2spk.close()
    segments.close()
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ami_diarization_config", type=str, required=True, 
    help="Path to the config file of AMI diarization, \
    typically located at ./ami_diarization_setup/pyannote/database.yml."
)
parser.add_argument(
    "--mic_type", type=str, required=True, 
    help="Microphone type, options: 'ihm', 'sdm'. (ihm: individual head mic, sdm: single distant mic)"
)
parser.add_argument(
    "--if_mini", type=str, required=True, 
    help="If true, use the subset of corresponding dataset."
)
parser.add_argument(
    "--sound_type", type=str, required=True, 
    help="Sound type, options: 'only_words', 'word_and_vocal'"
)
parser.add_argument(
    "--kaldi_files_base_dir", type=str, required=True, 
    help="Directory to store kaldi style data files, typically located at ./data, under this base dir, there are /train, /dev, /test."
)

args = parser.parse_args()

### prepare train
prepare_kaldi_files(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "train", args.kaldi_files_base_dir
)
### prepare dev
prepare_kaldi_files(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "dev", args.kaldi_files_base_dir
)
### prepare test
prepare_kaldi_files(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "test", args.kaldi_files_base_dir
)

print("Successfully complete Kaldi-style preparation")
