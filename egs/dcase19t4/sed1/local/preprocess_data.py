import os

# work_dir = espnet/egs/dcase19t4/sed1
DATA_DIR = './DCASE2019_task4/dataset/metadata'


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
                with open(os.path.join(DATA_DIR, x, 'synthetic.csv')) as synthetic, \
                     open(os.path.join(DATA_DIR, x, 'unlabel_in_domain.csv')) as unlabel, \
                     open(os.path.join(DATA_DIR, x, 'weak.csv')) as weak:

                    synthetic.readline()
                    unlabel.readline()
                    weak.readline()

                    for line in synthetic.readlines():
                        filename, onset, offset, label = line.strip().split('\t')
                        wav_id = os.path.splitext(filename)[0]
                        wav_scp_f.write(wav_id + ' ' + filename + '\n')
                        utt2spk_f.write(wav_id + ' ' + 'synthetic\n')
                        text_f.write(wav_id + ' ' + wav_id + '.json\n')

                    for line in unlabel.readlines():
                        filename = line.strip()
                        wav_id = os.path.splitext(filename)[0]
                        wav_scp_f.write(wav_id + ' ' + filename + '\n')
                        utt2spk_f.write(wav_id + ' ' + 'unlabel\n')
                        text_f.write(wav_id + ' ' + wav_id + '.json\n')

                    for line in weak.readlines():
                        filename, label = line.strip().split('\t')
                        wav_id = os.path.splitext(filename)[0]
                        wav_scp_f.write(wav_id + ' ' + filename + '\n')
                        utt2spk_f.write(wav_id + ' ' + 'weak\n')
                        text_f.write(wav_id + ' ' + wav_id + '.json\n')

            elif x == 'validation':
                with open(os.path.join(DATA_DIR, x, 'validation.csv')) as validation, \
                     open(os.path.join(DATA_DIR, x, 'eval_dcase2018.csv')) as eval, \
                     open(os.path.join(DATA_DIR, x, 'test_dcase2018.csv')) as test:

                    validation.readline()
                    eval.readline()
                    test.readline()

                    for line in validation.readlines():
                        if len(line.strip().split('\t')) != 4:
                            continue
                        filename, onset, offset, label = line.strip().split('\t')
                        wav_id = os.path.splitext(filename)[0]
                        wav_scp_f.write(wav_id + ' ' + filename + '\n')
                        utt2spk_f.write(wav_id + ' ' + 'validation\n')
                        text_f.write(wav_id + ' ' + wav_id + '.json\n')

                    for line in eval.readlines():
                        if len(line.strip().split('\t')) != 4:
                            continue
                        filename, onset, offset, label = line.strip().split('\t')
                        wav_id = os.path.splitext(filename)[0]
                        wav_scp_f.write(wav_id + ' ' + filename + '\n')
                        utt2spk_f.write(wav_id + ' ' + 'eval2018\n')
                        text_f.write(wav_id + ' ' + wav_id + '.json\n')

                    for line in test.readlines():
                        if len(line.strip().split('\t')) != 4:
                            continue
                        filename, onset, offset, label = line.strip().split('\t')
                        wav_id = os.path.splitext(filename)[0]
                        wav_scp_f.write(wav_id + ' ' + filename + '\n')
                        utt2spk_f.write(wav_id + ' ' + 'test2018\n')
                        text_f.write(wav_id + ' ' + wav_id + '.json\n')


if __name__ == '__main__':
    main()
