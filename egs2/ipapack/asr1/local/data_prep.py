import argparse
import glob
from pathlib import Path

import webdataset as wds
from scipy.io import wavfile
import pandas as pd
from tarfile import ReadError
from phonepiece.ipa import read_ipa

from tqdm import tqdm

# adapted from https://github.com/juice500ml/espnet/blob/wav2gloss/egs2/
#       wav2gloss/asr1/local/data_prep.py


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert downloaded data to Kaldi format"
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("downloads"),
        required=True
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=Path("data"),
        required=True
    )
    parser.add_argument(
        "--min_wav_length",
        type=float,
        default=0.5,
    )
    return parser


def get_original_split(original_dataset, split, utt_id, doreco_splits):
    # DoReCo: language code (ex: ana1239)
        # use their splits (Table 11) - train/test
        # they use glottocode https://doreco.huma-num.fr/languages
    # MSWC: split + batch? (ex: mswc-dev-000001)
        # use their splits
    # FLEURS: language code + split (ex: af_za-test)
    if original_dataset == 'doreco':
        # 0148_DoReCo_doreco_anal1239_anm_20152111_Ngahring_PO_56_283.wav
        return (doreco_splits[doreco_splits['glottocode_or_prefix'] == split]
                .iloc[0]['split'])
    elif original_dataset == 'mswc':
        return split.split('-')[1]
    elif original_dataset == 'fleurs':
        return split.split('-')[1]


def generate_train_dev_test_splits(original_dataset, split, utt_id,
                                   doreco_splits):
    split = get_original_split(original_dataset, split, utt_id, doreco_splits)
    assert split in ['train', 'dev', 'test']

    if split == 'test':
        return f'test_{original_dataset}'
    return split


def write_dir(source_dir, target_dir, transcripts):
    wavscp = open(target_dir / "wav.scp", "w", encoding="utf-8")
    text = open(target_dir / "text", "w", encoding="utf-8")
    utt2spk = open(target_dir / "utt2spk", "w", encoding="utf-8")
    utt_id_mapping = open(source_dir / "uttid_map", "w", encoding="utf-8")

    count = 0
    for _, row in transcripts.iterrows():
        utt_id, path, ipa, orig_split = (row['utt_id'], row['path'],
                                             row['ipa'],
                                             row['orig_split'])

        # generate a new utterance id
        new_utt_id = f"aaaaa_{dataset}_{orig_split}_{count:020d}"
        # map original utt_id to nnew_utt_id
        utt_id_mapping.write(f"{utt_id} {new_utt_id}\n")

        wavscp.write(f"{new_utt_id} {path}\n")
        text.write(f"{new_utt_id} {ipa}\n")
        # ESPnet does not use speaker info for ASR anymore
        utt2spk.write(f"{new_utt_id} aaaaa\n")

        count += 1

    wavscp.close()
    text.close()
    utt2spk.close()
    utt_id_mapping.close()

    print(f"{target_dir}: {count} lines written.")


def normalize_text(ipa, ipa_tokenizer):
    # remove whitespace
    ipa = "".join(ipa.split())
    # phone(me) tokenizer (phonepiece)
    ipa_tokens = ipa_tokenizer.tokenize(ipa)
    # re-introduce spaces (like TIMIT)
    return " ".join(ipa_tokens)


def get_dataset_name(tar_path):
    # get original dataset name
    if tar_path.name.startswith('mswc'):
        # MSWC: mswc-{train,dev,test}
        return 'mswc'
    elif tar_path.stem.endswith(('train','dev','test')):
        # FLEURS: -{train,dev,test}.tar
        return 'fleurs'
    else:
        return 'doreco'


if __name__ == "__main__":
    SAMPLING_RATE = 16000

    parser = get_parser()
    args = parser.parse_args()
    min_wav_length = args.min_wav_length

    for dataset in ['mswc', 'fleurs', 'doreco']:
        print("making directory", args.source_dir.joinpath(dataset))
        args.source_dir.joinpath(dataset).mkdir(parents=True, exist_ok=True)
    print("making directory", args.target_dir)
    args.target_dir.mkdir(parents=True, exist_ok=True)

    ipa_tokenizer = read_ipa()
    doreco_splits = pd.read_csv('local/doreco_splits.csv')

    rows = []
    # glob is non-deterministic -> sort after globbing
    for i, path in tqdm(enumerate(sorted(args.source_dir.glob("*.tar")))):
        split = path.stem
        original_dataset = get_dataset_name(path)

        path = str(path)
        ds = wds.WebDataset(path).decode()
        ds = ds.to_tuple("__key__", "__url__", "npy", "__local_path__", "txt")

        split_rows = []
        try:
            for utt_id, _, audio, _, ipa in ds:
                new_path = (f'{args.source_dir}/{original_dataset}/'
                            f'{split}/{utt_id}.wav')
                Path(new_path).parent.mkdir(parents=True, exist_ok=True)

                # audio is just a list of samples.
                # SAMPLING_RATE determines how many samples per sec.
                # enforce min_wav_length
                if len(audio) / SAMPLING_RATE < min_wav_length:
                    continue
                wavfile.write(new_path, SAMPLING_RATE, audio)

                ipa = normalize_text(ipa, ipa_tokenizer)
                split_rows.append((utt_id, split, original_dataset, new_path,
                                   ipa))
        except ReadError as e:
            # currently, only yuca1254 has problems
            print('failed to untar', path, e)
            # do not add any rows related to this language
            continue
        print('\nfinished', path)
        rows += split_rows

    df = pd.DataFrame(rows, columns=['utt_id', 'orig_split', 'dataset',
                                     'path', 'ipa'])
    # train/dev/test splits
    df['split'] = df.apply(lambda row: generate_train_dev_test_splits(
                            row['dataset'], row['orig_split'], row['utt_id'],
                            doreco_splits), axis=1)
    df.to_csv(args.source_dir / 'transcript.csv',
              index=False)

    # kaldi format
    for split, split_df in df.groupby('split'):
        split_dir = args.target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        write_dir(args.source_dir, split_dir, split_df)
