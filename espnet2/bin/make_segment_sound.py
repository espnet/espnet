# with a wav.scp file like this

# (base) [sbharadwaj@dt-login03 sbharadwaj]$ head icme_challenge/dump/raw/GigaSpeech/wav.scp
# GigaSpeech_AUD0000000003_000031540_000054400_en_asr /scratch/bbjs/shared/owsm_data/datasets/GigaSpeech/train/data/wav.1.ark:52
# GigaSpeech_AUD0000000003_000054580_000083880_en_asr /scratch/bbjs/shared/owsm_data/datasets/GigaSpeech/train/data/wav.1.ark:381495
# GigaSpeech_AUD0000000003_000084030_000112780_en_asr /scratch/bbjs/shared/owsm_data/datasets/GigaSpeech/train/data/wav.1.ark:873902

# give me code to make a segments file of the format

# <segment-id> <recording-id> <start-time> <end-time>

# Note that in the given file the record ids have the time period - GigaSpeech_AUD0000000003_000191710_000212660_en_asr

# For example this one has 191710 to 212660 milliseconds. 20 seconds approx. Extract these by splitting on underscore.

# In the segments file that we make we should create all utterances of 10 seconds. Therefore start from the start time in seconds and create multiple utterances for each record. Ignore the last utterance if it falls out of range.

# from pathlib import Path
# from tqdm import tqdm

# wav_scp_path = "/work/nvme/bbjs/sbharadwaj/icme_challenge/dump/raw/GigaSpeech/wav.scp"
# segments_path = "/work/nvme/bbjs/sbharadwaj/icme_challenge/dump/raw/GigaSpeech/segments"
# SEGMENT_LENGTH = 10  # seconds

# with open(wav_scp_path) as f:
#     lines = f.readlines()

# missed = 0
# with open(segments_path, "w") as out:
#     for line in tqdm(lines, desc="Generating segments"):
#         utt_id, _ = line.strip().split(maxsplit=1)
#         parts = utt_id.split("_")
#         if len(parts) < 4:
#             missed += 1
#             continue

#         start_ms = int(parts[2])
#         end_ms = int(parts[3])
#         start_sec = start_ms / 1000
#         end_sec = end_ms / 1000
#         rec_id = "_".join(parts)

#         seg_num = 0
#         t_in_audio = 0
#         t = start_sec
#         while t + SEGMENT_LENGTH <= end_sec:
#             segment_id = f"{utt_id}_{seg_num}"
#             out.write(
#                 f"{segment_id} {utt_id} {t_in_audio:.2f} {t_in_audio + SEGMENT_LENGTH:.2f}\n"
#             )
#             t += SEGMENT_LENGTH
#             t_in_audio += SEGMENT_LENGTH
#             seg_num += 1
#         # break

# print(f"Missed {missed} lines due to incorrect format.")


# with a wav.scp file like this

# (base) [sbharadwaj@dt-login03 sbharadwaj]$ head icme_challenge/dump/raw/GigaSpeech/wav.scp
# GigaSpeech_AUD0000000003_000031540_000054400_en_asr /scratch/bbjs/shared/owsm_data/datasets/GigaSpeech/train/data/wav.1.ark:52
# GigaSpeech_AUD0000000003_000054580_000083880_en_asr /scratch/bbjs/shared/owsm_data/datasets/GigaSpeech/train/data/wav.1.ark:381495
# GigaSpeech_AUD0000000003_000084030_000112780_en_asr /scratch/bbjs/shared/owsm_data/datasets/GigaSpeech/train/data/wav.1.ark:873902

# give me code to make a segments file of the format

# <segment-id> <recording-id> <start-time> <end-time>

# Note that in the given file the record ids have the time period - GigaSpeech_AUD0000000003_000191710_000212660_en_asr

# For example this one has 191710 to 212660 milliseconds. 20 seconds approx. Extract these by splitting on underscore.

# In the segments file that we make we should create all utterances of 10 seconds. Therefore start from the start time in seconds and create multiple utterances for each record. Ignore the last utterance if it falls out of range.

# from pathlib import Path
# from tqdm import tqdm

# wav_scp_path = "/work/nvme/bbjs/sbharadwaj/icme_challenge/dump/raw/commonvoice/wav.scp"
# segments_path = (
#     "/work/nvme/bbjs/sbharadwaj/icme_challenge/dump/raw/commonvoice/segments"
# )
# SEGMENT_LENGTH = 10  # seconds

# with open(wav_scp_path) as f:
#     lines = f.readlines()

# missed = 0
# with open(segments_path, "w") as out:
#     for line in tqdm(lines, desc="Generating segments"):
#         utt_id, _ = line.strip().split(maxsplit=1)
#         parts = utt_id.split("_")
#         parts = utt_id.split(".mp3")[1].split("_")[1:]
#         if len(parts) < 2:
#             missed += 1
#             continue

#         start_ms = int(parts[0])
#         end_ms = int(parts[1])
#         start_sec = start_ms / 1000
#         end_sec = end_ms / 1000
#         rec_id = utt_id

#         seg_num = 0
#         t_in_audio = 0
#         t = start_sec
#         while t < end_sec - 1:
#             segment_id = f"{utt_id}_{seg_num}"
#             endtime = min(t + SEGMENT_LENGTH, end_sec)
#             out.write(f"{segment_id} {utt_id} {t_in_audio:.2f} {endtime:.2f}\n")
#             t += SEGMENT_LENGTH
#             t_in_audio += SEGMENT_LENGTH
#             seg_num += 1
#         # break

# print(f"Missed {missed} lines due to incorrect format.")


##### MTG JAM #####
# give me code to create a segments file from the wav.scp file. use torch audio to get the duration of each utterance mp3. use tqdm. create segments file with kaldi like format

# import torchaudio
# from tqdm import tqdm

# wavscp_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/mtg_jamendo/wav.scp"
# segments_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/mtg_jamendo/segments"
# SEGMENT_LENGTH = 10

# with open(wavscp_path, "r") as f:
#     lines = [line.strip().split(maxsplit=1) for line in f]

# with open(segments_path, "w") as out_f:
#     for utt_id, wav_path in tqdm(lines, desc="Creating segments"):
#         try:
#             info = torchaudio.info(wav_path)
#             duration = info.num_frames / info.sample_rate
#             seg_num = 0
#             start = 0.0
#             while start + SEGMENT_LENGTH <= duration:
#                 segment_id = f"{utt_id}_segment{seg_num}"
#                 end = start + SEGMENT_LENGTH
#                 out_f.write(f"{segment_id} {utt_id} {start:.2f} {end:.2f}\n")
#                 start += SEGMENT_LENGTH
#                 seg_num += 1
#         except Exception as e:
#             print(f"Failed to process {utt_id} - {wav_path}: {e}")


# ##### EARS #####

# import torchaudio
# from tqdm import tqdm

# wavscp_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/ears/wav.scp"
# segments_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/ears/segments"
# SEGMENT_LENGTH = 10

# with open(wavscp_path, "r") as f:
#     lines = [line.strip().split(maxsplit=1) for line in f]

# with open(segments_path, "w") as out_f:
#     for utt_id, wav_path in tqdm(lines, desc="Creating segments"):
#         try:
#             info = torchaudio.info(wav_path)
#             duration = info.num_frames / info.sample_rate
#             seg_num = 0
#             start = 0.0
#             while start + SEGMENT_LENGTH <= duration:
#                 segment_id = f"{utt_id}_segment{seg_num}"
#                 end = start + SEGMENT_LENGTH
#                 out_f.write(f"{segment_id} {utt_id} {start:.2f} {end:.2f}\n")
#                 start += SEGMENT_LENGTH
#                 seg_num += 1
#         except Exception as e:
#             print(f"Failed to process {utt_id} - {wav_path}: {e}")

# (espnet) [sbharadwaj@dt-login03 ssl1]$ head  /work/nvme/bbjs/peng6/DeltaAI/espnet-owsm-ft/egs2/owsm_v4/s2t1/dump/raw/yodas0.10/wav.scp
# ---9WmvOo24_000000784_000030373_vie_asr dump/raw/org/yodas0.00/data/format.1/data_wav.ark:40
# ---9WmvOo24_000030373_000059343_vie_asr dump/raw/org/yodas0.00/data/format.1/data_wav.ark:496437

##### YODAS-Speech #####

import torchaudio
from tqdm import tqdm
import random

wavscp_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/yodas_speech/wav.scp"
segments_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/yodas_speech/segments"
path_prefix = "/work/nvme/bbjs/peng6/DeltaAI/espnet-owsm-ft/egs2/owsm_v4/s2t1/"
new_wav_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/yodas_speech/wav.new.scp"
SEGMENT_LENGTH = 10

with open(wavscp_path, "r") as f:
    lines = [line.strip().split(maxsplit=1) for line in f]

with open(segments_path, "w") as out_f, open(new_wav_path, "w") as new_wav_f:
    for utt_id, wav_path in tqdm(lines, desc="Creating segments"):
        try:
            start, end = utt_id.split("_")[-4:-2]
            start = int(start) / 1000.0
            end = int(end) / 1000.0
            duration = end - start
            segment_id = f"{utt_id}_segment0"
            end = min(duration, SEGMENT_LENGTH)
            out_f.write(f"{segment_id} {utt_id} 0.0 {end:.2f}\n")
            new_wav_f.write(f"{utt_id} {path_prefix}{wav_path}\n")
        except Exception as e:
            print(f"Failed to process {utt_id} - {wav_path}: {e}")
