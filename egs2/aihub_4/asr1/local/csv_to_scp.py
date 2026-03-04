# csv: file_name, labels
# scp: utt2wav, utt2txt
# utt id is assigned by the index of the csv file


csv_path = "/mnt/ssd/jieun/datasets/fss/조직원only.csv"

import pandas as pd
import subprocess
df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    wav = row['file_name']
    txt = row['labels']
    utt = f'utt{index:05d}' # 5자리 숫자로 만들기

    with open('/mnt/ssd/jieun/datasets/fss/조직원_utt2wav.scp', 'a') as f:
        f.write(f'{utt}\t{wav}\n')
    with open('/mnt/ssd/jieun/datasets/fss/조직원_utt2txt.scp', 'a') as f:
        f.write(f'{utt}\t{txt}\n')

# bash 명령어 실행

cmd = """
cd /ESPnet/espnet/egs2/aihub_4/asr1
mkdir -p data/eval2
cp /mnt/ssd/jieun/datasets/fss/조직원_utt2wav.scp data/eval2/wav.scp
cp /mnt/ssd/jieun/datasets/fss/조직원_utt2txt.scp data/eval2/text
awk '{print $1, $1}' data/eval2/wav.scp > data/eval2/utt2spk
utils/utt2spk_to_spk2utt.pl data/eval2/utt2spk > data/eval2/spk2utt
utils/fix_data_dir.sh data/eval2
"""
subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)

