import os
import pandas as pd
import numpy as np

# work_dir = espnet/egs/dcase19t4/sed1
DATA_DIR = './DCASE2019_task4/dataset/metadata'
AUDIO_DIR = os.path.join(os.getcwd(), 'DCASE2019_task4/dataset/audio')

def make_scps(scp_dir: str, filenames: np.ndarray, dataset: str, att: str) -> None:
    with open(os.path.join(scp_dir, 'text'), 'a') as text_f, \
         open(os.path.join(scp_dir, 'wav.scp'), 'a') as wav_scp_f, \
         open(os.path.join(scp_dir, 'utt2spk'), 'a') as utt2spk_f:

        for filename in filenames:
            if not os.path.exists(os.path.join(AUDIO_DIR, dataset, att, filename)):
                continue
            wav_id = os.path.splitext(filename)[0]
            wav_scp_f.write(f'{att}-{wav_id} {os.path.join(AUDIO_DIR, dataset, att, filename)}\n')
            utt2spk_f.write(f'{att}-{wav_id} {att}\n')
            text_f.write(f'{att}-{wav_id} {wav_id}.json\n')

def main():

    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/validation', exist_ok=True)

    for x in ['train', 'validation']:
        with open(os.path.join('data', x, 'text'), 'w') as text_f, \
             open(os.path.join('data', x, 'wav.scp'), 'w') as wav_scp_f, \
             open(os.path.join('data', x, 'utt2spk'), 'w') as utt2spk_f:

            text_f.truncate()
            wav_scp_f.truncate()
            utt2spk_f.truncate()

        if x == 'train':

            df_synthetic = pd.read_csv(os.path.join(DATA_DIR, x, 'synthetic.csv'), delimiter='\t')
            df_unlabel = pd.read_csv(os.path.join(DATA_DIR, x, 'unlabel_in_domain.csv'), delimiter='\t')
            df_weak = pd.read_csv(os.path.join(DATA_DIR, x, 'weak.csv'), delimiter='\t')

            make_scps(os.path.join('data', x), df_synthetic.dropna()['filename'].unique(), x, 'synthetic')
            make_scps(os.path.join('data', x), df_unlabel.dropna()['filename'].unique(), x, 'unlabel_in_domain')
            make_scps(os.path.join('data', x), df_weak.dropna()['filename'].unique(), x, 'weak')

        # elif x == 'validation':
        #     df_validation = pd.read_csv(os.path.join(DATA_DIR, x, 'validation.csv'), delimiter='\t')
        #     df_eval = pd.read_csv(os.path.join(DATA_DIR, x, 'eval_dcase2018.csv'), delimiter='\t')
        #     df_test = pd.read_csv(os.path.join(DATA_DIR, x, 'test_dcase2018.csv'), delimiter='\t')
        #
        #     make_scps(os.path.join('data', x), df_validation.dropna()['filename'].unique(), x, 'validation')
        #     make_scps(os.path.join('data', x), df_eval.dropna()['filename'].unique(), x, 'eval2018')
        #     make_scps(os.path.join('data', x), df_test.dropna()['filename'].unique(), x, 'test2018')


if __name__ == '__main__':
    main()
