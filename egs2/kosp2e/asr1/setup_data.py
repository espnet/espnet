import os
from tqdm import tqdm
import pandas as pd

data_types = ['covid', 'kss', 'stylekqc', 'zeroth']
wavs_path = 'egs2/kosp2e/asr1/wavs'
data_path = 'egs2/kosp2e/asr1/data'
metadata_path = 'egs2/kosp2e/asr1/downloads/metadata'

assert os.path.exists(wavs_path), "WAVS path does not exist. See https://github.com/warnikchow/kosp2e/tree/main"
assert os.path.exists(data_path), "DATA path does not exist. See https://github.com/warnikchow/kosp2e/tree/main"
assert os.path.exists(metadata_path), "METADATA path does not exist. See https://github.com/warnikchow/kosp2e/tree/main"

meta_data_train = {}
meta_data_val = {}
meta_data_test = {}

def check_data_train_val_test(data_type, wav_name):
    wav_name = wav_name.split('.')[0]
    if wav_name in meta_data_test[data_type]['id'].values:
        return 'test'
    elif wav_name in meta_data_val[data_type]['id'].values:
        return 'val'
    elif wav_name in meta_data_train[data_type]['id'].values:
        return 'train'
    else:
        return None

for data_type in tqdm(data_types, desc="Loading metadata"):
    meta_data_train[data_type] = pd.read_excel(f"{metadata_path}/{data_type}_train.xlsx")
    meta_data_val[data_type] = pd.read_excel(f"{metadata_path}/{data_type}_dev.xlsx")
    meta_data_test[data_type] = pd.read_excel(f"{metadata_path}/{data_type}_test.xlsx")

# Create 'text' file
# Form : uttidA &lt;transcription&gt;

for data_type in tqdm(data_types, desc="Creating 'text' file"):
    print(f"Processing {data_type}")
    for wav_file in tqdm(sorted(os.listdir(f"{wavs_path}/{data_type}")), desc=f"Processing {data_type}"):
        wav_name = wav_file.split('.')[0]
        train_val_test = check_data_train_val_test(data_type, wav_file)
        if train_val_test is None:
            continue
        elif train_val_test == 'train':
            transcription = meta_data_train[data_type][meta_data_train[data_type]['id'] == wav_name]['sentence'].values[0]
            with open(f"{data_path}/train/text", "a") as f:
                f.write(f"{wav_name} {transcription}\n")
        elif train_val_test == 'val':
            transcription = meta_data_val[data_type][meta_data_val[data_type]['id'] == wav_name]['sentence'].values[0]
            with open(f"{data_path}/val/text", "a") as f:
                f.write(f"{wav_name} {transcription}\n")
        elif train_val_test == 'test':
            transcription = meta_data_test[data_type][meta_data_test[data_type]['id'] == wav_name]['sentence'].values[0]
            with open(f"{data_path}/test/text", "a") as f:
                f.write(f"{wav_name} {transcription}\n")


# Create 'wav.scp' file
# Form : uttidA wav-path

for data_type in tqdm(data_types, desc="Creating 'wav.scp' file"):
    print(f"Processing {data_type}")
    for wav_file in tqdm(sorted(os.listdir(f"{wavs_path}/{data_type}")), desc=f"Processing {data_type}"):
        wav_name = wav_file.split('.')[0]
        wav_path = f"wavs/{data_type}/{wav_file}"
        train_val_test = check_data_train_val_test(data_type, wav_file)
        if train_val_test is None:
            continue
        elif train_val_test == 'train':
            with open(f"{data_path}/train/wav.scp", "a") as f:
                f.write(f"{wav_name} {wav_path}\n")
        elif train_val_test == 'val':
            with open(f"{data_path}/val/wav.scp", "a") as f:
                f.write(f"{wav_name} {wav_path}\n")
        elif train_val_test == 'test':
            with open(f"{data_path}/test/wav.scp", "a") as f:
                f.write(f"{wav_name} {wav_path}\n")

# Create 'utt2spk' file
# Form : uttidA spk_id

for data_type in tqdm(data_types, desc="Creating 'utt2spk' file"):
    print(f"Processing {data_type}")
    for wav_file in tqdm(sorted(os.listdir(f"{wavs_path}/{data_type}")), desc=f"Processing {data_type}"):
        wav_name = wav_file.split('.')[0]
        train_val_test = check_data_train_val_test(data_type, wav_file)
        if train_val_test is None:
            continue
        elif train_val_test == 'train':
            spk_id = meta_data_train[data_type][meta_data_train[data_type]['id'] == wav_name]['id'].values[0]
            with open(f"{data_path}/train/utt2spk", "a") as f:
                f.write(f"{wav_name} {spk_id}\n")
        elif train_val_test == 'val':
            spk_id = meta_data_val[data_type][meta_data_val[data_type]['id'] == wav_name]['id'].values[0]
            with open(f"{data_path}/val/utt2spk", "a") as f:
                f.write(f"{wav_name} {spk_id}\n")
        elif train_val_test == 'test':
            spk_id = meta_data_test[data_type][meta_data_test[data_type]['id'] == wav_name]['id'].values[0]
            with open(f"{data_path}/test/utt2spk", "a") as f:
                f.write(f"{wav_name} {spk_id}\n")

# Create 'spk2utt' file
# Form : spk_id utt_ids

for data_type in tqdm(data_types, desc="Creating 'spk2utt' file"):
    print(f"Processing {data_type}")
    for wav_file in tqdm(sorted(os.listdir(f"{wavs_path}/{data_type}")), desc=f"Processing {data_type}"):
        wav_name = wav_file.split('.')[0]
        train_val_test = check_data_train_val_test(data_type, wav_file)
        if train_val_test is None:
            continue
        elif train_val_test == 'train':
            spk_id = meta_data_train[data_type][meta_data_train[data_type]['id'] == wav_name]['id'].values[0]
            with open(f"{data_path}/train/spk2utt", "a") as f:
                f.write(f"{spk_id} {wav_name}\n")
        elif train_val_test == 'val':
            spk_id = meta_data_val[data_type][meta_data_val[data_type]['id'] == wav_name]['id'].values[0]
            with open(f"{data_path}/val/spk2utt", "a") as f:
                f.write(f"{spk_id} {wav_name}\n")
        elif train_val_test == 'test':
            spk_id = meta_data_test[data_type][meta_data_test[data_type]['id'] == wav_name]['id'].values[0]
            with open(f"{data_path}/test/spk2utt", "a") as f:
                f.write(f"{spk_id} {wav_name}\n")

# After all files are written, sort each file by uttid
for split in ['train', 'val', 'test']:
    for filename in ['text', 'wav.scp', 'utt2spk', 'spk2utt']:
        file_path = f"{data_path}/{split}/{filename}"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
            lines = sorted(lines, key=lambda x: x.split()[0])
            with open(file_path, "w") as f:
                f.writelines(lines)
            print(f"Sorted {file_path}")