# Prepare Kaldi-style data directory for AMI diarization task
import argparse
import os
import re

def float2str(number, size=6):
    number = str(int(number * 100))  # convert to integer after multiplying by 100
    return number.zfill(size)  # pad with zeros to the left to ensure the string is of the desired size


def gen_utt_id(wav_id: str, spk_id: str, utt_start_time: float, utt_end_time: float) -> str:
    return f"{spk_id}_{wav_id}_{float2str(utt_start_time)}_{float2str(utt_end_time)}"


def prepare_kaldi_files(
    dataset_type: str, kaldi_files_base_dir: str, 
    num_spk: str, segmented_dataset_dir: str, ami_setup_base_dir: str
) -> None:
    # dataset_type: "train", "dev", "test"
    
    assert num_spk == "3" or num_spk == "4" or num_spk == "5", f"num_spk should be 3, 4, or 5, but get {num_spk}"
    num_spk = int(num_spk)
    kaldi_files_dir = os.path.join(kaldi_files_base_dir, dataset_type)

    if not os.path.exists(kaldi_files_dir):
        os.makedirs(kaldi_files_dir)

    wavscp = open(os.path.join(kaldi_files_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(kaldi_files_dir, "utt2spk"), "w", encoding="utf-8")
    segments = open(os.path.join(kaldi_files_dir, "segments"), "w", encoding="utf-8")

    # wav_id_txt is the txt file contains a list of wav_ids, uri with the same meaning of wav_id
    wav_id_txt = os.path.join(segmented_dataset_dir, dataset_type, "wav_ids.txt")

    wav_path_template = f"{segmented_dataset_dir}/{dataset_type}/wav/{{uri}}.wav" 
    rttm_path_template = f"{segmented_dataset_dir}/{dataset_type}/rttm/{{uri}}.rttm" 
    full_rttm_path_template = f"{ami_setup_base_dir}/only_words/rttms/{dataset_type}/{{uri_full}}.rttm"

    wav_ids = []
    with open(wav_id_txt, "r") as f:
        for line in f:
            wav_ids.append(line.strip())

    segments_entries = []
    utt2spk_entries = []
    wavscp_entries = []
    unique_speaker_all = dict() # key: wav_id (before split), value: set of unique speakers

    for wav_id in wav_ids:
        rttm_path = rttm_path_template.format(uri=wav_id)
        wav_id_full = wav_id.split("_")[0] # the corresponding full wav_id before split, e.g., wav_id ES2002a_002, full_wav_id ES2002a
        rttm_path_full = full_rttm_path_template.format(uri_full=wav_id_full)
        
        ### ======== Count the unique number of speakers in the original rttm (before split) ======== ###
        # Count the number of unique speakers in the rttm file (before split)
        if wav_id_full not in unique_speaker_all:
            unique_speaker_set = set()
            with open(rttm_path_full, "r") as f:
                # Count the number of unique speakers in the rttm file
                for line_id, line in enumerate(f):
                    sps = re.split(" +", line.rstrip())
                    assert len(sps) == 10, f"Error in {rttm_path} at line {line_id + 1}"

                    label_type, wav_id, channel, spk_start_time, spk_duration, _, _, spk_id, _, _ = sps
                    assert label_type == "SPEAKER", f"Error in {rttm_path} at line {line_id + 1}"

                    unique_speaker_set.add(spk_id)
            unique_speaker_all[wav_id_full] = unique_speaker_set
        
        ## Skip the wav files with more or less than num_spk speakers
        if len(unique_speaker_all[wav_id_full]) != num_spk:
            continue 

        ### ======================= Prepare segments, utt2spk ================== ###
        spk_in_segment = set()
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
                spk_in_segment.add(spk_id)

                segments_entries.append(f"{utt_id} {wav_id} {spk_start_time} {spk_end_time}\n")
                utt2spk_entries.append(f"{utt_id} {spk_id}\n")
            
            # Add placeholder for the missing speakers, to ensure that each wav file has 4 speakers
            if spk_in_segment != unique_speaker_all[wav_id_full]:
                for spk_id in unique_speaker_all[wav_id_full]:
                    if spk_id not in spk_in_segment:
                        spk_start_time = 0.0
                        spk_end_time = 1 / 16000  # 1 sample, the end time must be greater than start time, and set 1 sample will not affect the result, since the duration is too short
                        utt_id = gen_utt_id(wav_id, spk_id, spk_start_time, spk_end_time)

                        segments_entries.append(f"{utt_id} {wav_id} {spk_start_time} {spk_end_time}\n")
                        utt2spk_entries.append(f"{utt_id} {spk_id}\n")

        ### ====================== Prepare wav.scp ============================== ###
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
    "--kaldi_files_base_dir", type=str, required=True, 
    help="Directory to store kaldi style data files, typically located at ./data, under this base dir, there are /train, /dev, /test."
)
parser.add_argument(
    "--num_spk", type=str, required=True, 
    help="The number of speakers in a wav file. Only accept 3, 4, or 5. "
)
parser.add_argument(
    "--segmented_dataset_dir", type=str, required=True, 
    help="Directory to store the segmented wavs and rttms, typically located at ./segmented_dataset, under this base dir, there are /train, /dev, /test."
)
parser.add_argument(
    "--ami_setup_base_dir", type=str, required=True, 
    help="Directory to ami diarization setup files, typically located at ./ami_diarization_setup"
)

args = parser.parse_args()

### prepare train
prepare_kaldi_files(
    "train", args.kaldi_files_base_dir, args.num_spk, 
    args.segmented_dataset_dir, args.ami_setup_base_dir
)
### prepare dev
prepare_kaldi_files(
    "dev", args.kaldi_files_base_dir, args.num_spk, 
    args.segmented_dataset_dir, args.ami_setup_base_dir
)
### prepare test
prepare_kaldi_files(
    "test", args.kaldi_files_base_dir, args.num_spk, 
    args.segmented_dataset_dir, args.ami_setup_base_dir
)

print("Successfully complete Kaldi-style preparation")
