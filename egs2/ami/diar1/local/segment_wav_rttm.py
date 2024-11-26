import logging
import os
import argparse
import soundfile
import yaml
from tqdm import tqdm

logger = logging.getLogger('segment_wav_rttm.py')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("{asctime} ({name}:{lineno}:{levelname}) {message}", style='{')
handler.setFormatter(formatter)
logger.addHandler(handler)

def segment_wav_rttm(
    config_path: str, 
    mic_type: str,
    if_mini: str,
    sound_type: str, 
    dataset_type: str, 
    segment_output_dir: str,
    duration: float = 20.0 
) -> None:
    """
    Segment the WAV files and corresponding RTTM files in the dataset.

    Args:
        config_path (str): 
            Path to the config file for AMI diarization, 
            typically located at `./ami_diarization_setup/pyannote/database.yml`.
        
        mic_type (str): 
            Type of microphone. Options are:
            - 'ihm': Individual Head Mic
            - 'sdm': Single Distant Mic
        
        if_mini (str): 
            Use a subset of the dataset if set to 'true'.
            NOTE: to supply the shell script, the 'true'
            and 'false' should be in string format. 
        
        sound_type (str): 
            Type of sound to segment. Options are:
            - 'only_words': Segment words only
            - 'word_and_vocal': Segment both words and vocals
        
        dataset_type (str): 
            Type of dataset to process. Options are:
            - 'train'
            - 'dev'
            - 'test'
        
        segment_output_dir (str): 
            Directory to store the segmented WAVs and RTTMs.
            Typically located at `./segmented_dataset`, with subdirectories for 
            each dataset type (`/train`, `/dev`, `/test`).
        
        duration (float):
            The threshold duration to segment the WAV files.
            The segmented wav files will have a duration greater than or equal to 
            this value.
    """
    ### ======================= Load config file ======================= ###
    logger.info(f"Start segmenting {dataset_type} dataset")

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
        assert not if_mini, "Only full dataset is available for only_words sound type, please set sound_type to None."

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset = "AMI-SDM" if mic_type == "sdm" else "AMI"
    task = "SpeakerDiarization"
    task_option = "mini" if if_mini else sound_type

    databases = config["Databases"]
    protocols = config["Protocols"]

    # wav_id_txt is the txt file contains a list of wav_ids, uri with the same meaning of wav_id
    wav_id_txt = protocols[dataset][task][task_option][dataset_type]["uri"] 

    # wav_path_template is like amicorpus/{uri}/audio/{uri}.Mix-Headset.wav
    wav_path_template = databases[dataset] 

    # rttm_path_template is like ami_diarization_setup/only_words/rttms/train/{uri}.rttm
    rttm_path_template = protocols[dataset][task][task_option][dataset_type]["annotation"] 

    output_segment_wavs_dir = os.path.join(segment_output_dir, dataset_type, "wav")
    output_segment_rttms_dir = os.path.join(segment_output_dir, dataset_type, "rttm")
    
    if not os.path.exists(output_segment_wavs_dir):
        os.makedirs(output_segment_wavs_dir)
    if not os.path.exists(output_segment_rttms_dir):
        os.makedirs(output_segment_rttms_dir)
    
    wav_ids = []
    segmented_wav_ids = []

    with open(wav_id_txt, "r") as f:
        for line in f:
            wav_ids.append(line.strip())

    ### =============== Segment each wav and corresponding rttm file =============== ###
    for wav_id in tqdm(wav_ids, desc=f"{dataset_type} dataset"):

        wav_path = wav_path_template.format(uri=wav_id)
        rttm_path = rttm_path_template.format(uri=wav_id)
        
        wav, sr = soundfile.read(wav_path)
        assert len(wav.shape) == 1, f"Error in {wav_path}, the wav file should be mono channel"

        segment_id = 0
        last_segment_end_time = 0.0
        curr_spk_start_time = 0.0
        curr_spk_end_time = 0.0
        segment_rttm_entries = [] # The segmented rttms for each duration, when new segment, clear the list first

        with open(rttm_path, "r") as f:
            lines = f.readlines()
            num_lines = len(lines)
            line_id = 0
            while line_id < num_lines:
                line = lines[line_id]
                sps = line.strip().split()
                assert len(sps) == 10, f"Error in {rttm_path} at line {line_id + 1}"

                label_type, wav_id, channel, spk_start_time, spk_duration, _, _, spk_id, _, _ = sps
                assert label_type == "SPEAKER", f"Error in {rttm_path} at line {line_id + 1}"

                relative_spk_start_time = round(float(spk_start_time) - last_segment_end_time, 2)
                relative_line = f"{label_type} {wav_id}_{segment_id} {channel} {str(relative_spk_start_time)} {spk_duration} <NA> <NA> {spk_id} <NA> <NA>\n"
                segment_rttm_entries.append(relative_line)

                curr_spk_start_time = float(spk_start_time)
                curr_spk_end_time = float(spk_start_time) + float(spk_duration)

                if curr_spk_end_time - last_segment_end_time >= duration:

                    ### Detect whether there are overlapped speakers (fully and partially)
                    for next_line_id in range(line_id + 1, num_lines):
                        sps_next = lines[next_line_id].strip().split()
                        assert len(sps_next) == 10, f"Error in {rttm_path} at line {next_line_id + 1}"
                        label_type_next, wav_id_next, channel_next, spk_start_time_next, spk_duration_next, _, _, spk_id_next, _, _ = sps_next

                        assert label_type_next == "SPEAKER", f"Error in {rttm_path} at line {next_line_id + 1}"
                        relative_spk_start_time_next = round(float(spk_start_time_next) - last_segment_end_time, 2)
                        relative_line_next = f"{label_type_next} {wav_id_next}_{segment_id} {channel_next} {str(relative_spk_start_time_next)} {spk_duration_next} <NA> <NA> {spk_id_next} <NA> <NA>\n"

                        next_spk_start_time = float(spk_start_time_next)
                        next_spk_end_time = float(spk_start_time_next) + float(spk_duration_next)

                        if next_spk_start_time >= curr_spk_end_time:
                            # No overlap
                            break
                        if next_spk_start_time >= curr_spk_start_time and next_spk_end_time <= curr_spk_end_time:
                            # Fully overlap
                            segment_rttm_entries.append(relative_line_next)
                            line_id += 1 # read and append, then line_id += 1
                            continue
                        if next_spk_start_time >= curr_spk_start_time and next_spk_end_time > curr_spk_end_time:
                            # Partially overlap
                            curr_spk_start_time = next_spk_start_time
                            curr_spk_end_time = next_spk_end_time
                            segment_rttm_entries.append(relative_line_next)
                            line_id += 1
                            continue

                    if len(segment_rttm_entries) == 0:
                        # No speaker in this duration, skip
                        last_segment_end_time = curr_spk_end_time
                        continue

                    segment_wav_path = os.path.join(output_segment_wavs_dir, f"{wav_id}_{segment_id}.wav")
                    segment_rttm_path = os.path.join(output_segment_rttms_dir, f"{wav_id}_{segment_id}.rttm")

                    # Check if correct duration
                    line_last = segment_rttm_entries[-1].strip().split()
                    assert float(line_last[3]) + float(line_last[4]) - (curr_spk_end_time - last_segment_end_time) < 1e-3, (
                        f"Error in {line_last}, the duration of segment rttm must equal to the duration of the segment wav.\n"
                        f"rttm duration: {float(line_last[3]) + float(line_last[4])}\n"
                        f"wav duration: {curr_spk_end_time - last_segment_end_time}"
                    )

                    soundfile.write(segment_wav_path, wav[int(last_segment_end_time * sr):int(curr_spk_end_time * sr)], sr)
                    with open(segment_rttm_path, "w") as f:
                        f.writelines(segment_rttm_entries)
                    
                    # Clear the segment_rttm_entries
                    segment_rttm_entries = []
                    segmented_wav_ids.append(f"{wav_id}_{segment_id}")
                    segment_id += 1
                    last_segment_end_time = curr_spk_end_time
                
                line_id += 1

            if curr_spk_end_time - last_segment_end_time > 0:
                # Process the last segment, which is less than duration
                if len(segment_rttm_entries) == 0:
                    # No speaker in this duration, skip
                    continue

                segment_wav_path = os.path.join(output_segment_wavs_dir, f"{wav_id}_{segment_id}.wav")
                segment_rttm_path = os.path.join(output_segment_rttms_dir, f"{wav_id}_{segment_id}.rttm")

                soundfile.write(segment_wav_path, wav[int(last_segment_end_time * sr):], sr)
                with open(segment_rttm_path, "w") as f:
                    f.writelines(segment_rttm_entries)
                
                segmented_wav_ids.append(f"{wav_id}_{segment_id}")
                segment_id += 1


    # Write segmented wav file list to a txt file. 
    segmented_wav_ids.sort()
    with open(os.path.join(segment_output_dir, dataset_type, "wav_ids.txt"), "w") as f:
        f.write("\n".join(segmented_wav_ids) + "\n")
    
    logger.info(f"Complete segmenting {dataset_type} dataset, {len(segmented_wav_ids)} segments are generated in total.")

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
    "--segment_output_dir", type=str, required=True, 
    help="Directory to store the output segmented wavs and rttms, typically located at ./segmented_dataset, under this base dir, there are /train, /dev, /test."
)
parser.add_argument(
    "--duration", type=str, required=True, 
    help="The estimated duration of the segmented wav files, which is not the exact duration of each segment, but the lower bound of the duration of each segment."
)

args = parser.parse_args()

### segment train
segment_wav_rttm(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "train", args.segment_output_dir, 
    float(args.duration)
)
### segment dev
segment_wav_rttm(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "dev", args.segment_output_dir, 
    float(args.duration)
)
### segment test
segment_wav_rttm(
    args.ami_diarization_config, args.mic_type, args.if_mini, 
    args.sound_type, "test", args.segment_output_dir, 
    float(args.duration)
)
