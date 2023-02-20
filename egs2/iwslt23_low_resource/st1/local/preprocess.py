import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', required=True)

def main(in_path, split):

    # collect wavs
    wav_path = f'{in_path}/{split}/wav/'
    wav_src = os.listdir(wav_path)

    # construct text
    que_src = open(f'{in_path}/{split}/txt/{split}.que', 'r', encoding='utf-8').readlines()
    esp_src = open(f'{in_path}/{split}/txt/{split}.spa.tc', 'r', encoding='utf-8').readlines()
    parallel_src = zip(que_src, esp_src, wav_src)

    que_text = open(f'data/{split}/text.que', 'w', encoding='utf-8')
    esp_text = open(f'data/{split}/text.spa', 'w', encoding='utf-8')
    wav_scp = open(f'data/{split}/wav.scp', 'w', encoding='utf-8')
    utt2spk = open(f'data/{split}/utt2spk', 'w', encoding='utf-8')
    spk2utt = open(f'data/{split}/spk2utt', 'w', encoding='utf-8')


    for i, line in enumerate(parallel_src):
        utt_id = line[2].split('.')[0]

        que_text.write(f'{utt_id} {line[0].strip()}\n')
        esp_text.write(f'{utt_id} {line[1].strip()}\n')
        wav_scp.write(f'{utt_id} {wav_path}{utt_id}.wav\n')
        utt2spk.write(f'{utt_id} {utt_id}\n')
        spk2utt.write(f'{utt_id} {utt_id}\n')

    que_text.close()
    esp_text.close()
    wav_scp.close()
    utt2spk.close()
    spk2utt.close()

if __name__ == "__main__":

    args = parser.parse_args()

    splits = ['train', 'valid', 'test']
    for split in splits:
        main(args.in_path, split)