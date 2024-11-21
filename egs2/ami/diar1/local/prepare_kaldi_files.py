# Prepare Kaldi-style data directory for AMI diarization task
import argparse
import os
import re
import yaml

def float2str(number, size=6):
    number = str(int(number * 100))  # convert to integer after multiplying by 100
    return number.zfill(size)  # pad with zeros to the left to ensure the string is of the desired size


def gen_utt_id(wav_id: str, spk_id: str, utt_start_time: float, utt_end_time: float) -> str:
    return f"{spk_id}_{wav_id}_{float2str(utt_start_time)}_{float2str(utt_end_time)}"


def prepare_kaldi_files(
    config_path: str, mic_type: str, if_mini: str, 
    sound_type: str, dataset_type: str, kaldi_files_base_dir: str, 
    num_spk: str
) -> None:
    # dataset_type: "train", "dev", "test"
    if if_mini == "true":
        if_mini = True
    elif if_mini == "false":
        if_mini = False
    else:
        raise ValueError("if_mini must be 'true' or 'false'")
    
    assert num_spk == "4" or num_spk == "None", f"num_spk should be 4 or None, but get {num_spk}"
    
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

    segments_entries = []
    utt2spk_entries = []
    wavscp_entries = []

    for wav_id in wav_ids:
        rttm_path = rttm_path_template.format(uri=wav_id)
        lab_path = lab_path_template.format(uri=wav_id)
        
        ### ======== Detect wav files with more or less than 4 speakers ======== ###
        unique_speaker_set = set()
        with open(rttm_path, "r") as f:
            for line_id, line in enumerate(f):
                sps = re.split(" +", line.rstrip())
                assert len(sps) == 10, f"Error in {rttm_path} at line {line_id + 1}"

                label_type, wav_id, channel, spk_start_time, spk_duration, _, _, spk_id, _, _ = sps
                assert label_type == "SPEAKER", f"Error in {rttm_path} at line {line_id + 1}"

                unique_speaker_set.add(spk_id)
        
        if num_spk == "4" and len(unique_speaker_set) != 4:
            continue # For num_spk == 4, since there are several files in ami that with 
                     # 3 or 5 speakers, we choose to neglect these files. 

        ### ======== Prepare segments, utt2spk ======== ###
        with open(rttm_path, "r") as f:
            f.seek(0)
            for line_id, line in enumerate(f):
                sps = re.split(" +", line.rstrip())
                assert len(sps) == 10, f"Error in {rttm_path} at line {line_id + 1}"

                label_type, wav_id, channel, spk_start_time, spk_duration, _, _, spk_id, _, _ = sps
                assert label_type == "SPEAKER", f"Error in {rttm_path} at line {line_id + 1}"

                spk_start_time = float(spk_start_time)
                spk_end_time = spk_start_time + float(spk_duration)
                utt_id = gen_utt_id(wav_id, spk_id, spk_start_time, spk_end_time)

                segments_entries.append(f"{utt_id} {wav_id} {spk_start_time} {spk_end_time}\n")
                utt2spk_entries.append(f"{utt_id} {spk_id}\n")

        ### ======== Prepare wav.scp ======== ###
        wav_path = wav_path_template.format(uri=wav_id)
        wavscp_entries.append(f"{wav_id} {wav_path}\n")
    
    segments_entries.sort()
    utt2spk_entries.sort()
    wavscp_entries.sort()
    segments.writelines(segments_entries)
    utt2spk.writelines(utt2spk_entries)
    wavscp.writelines(wavscp_entries)
    
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

parser.add_argument(
    "--num_spk", type=str, required=True, 
    help="The number of speakers in a wav file. Only accept 4 or None. "
)

args = parser.parse_args()

### prepare train
prepare_kaldi_files(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "train", args.kaldi_files_base_dir, args.num_spk
)
### prepare dev
prepare_kaldi_files(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "dev", args.kaldi_files_base_dir, args.num_spk
)
### prepare test
prepare_kaldi_files(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "test", args.kaldi_files_base_dir, args.num_spk
)

print("Successfully complete Kaldi-style preparation")
