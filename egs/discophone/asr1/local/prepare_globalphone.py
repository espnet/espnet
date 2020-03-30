#!/usr/bin/env python

import re
from pathlib import Path
from typing import NamedTuple, Dict, Optional

from tqdm import tqdm

SHN_DECODE_DECORATOR = '{shorten_path} -x {shn_file} - | sox -t raw -r 16000 -b 16 -e signed-integer - -t wav - |'

CODE2LANG = {
    'S0192': 'Arabic',
    'S0196': 'Czech',
    'S0197': 'French',
    'S0200': 'Korean',
    'S0193': 'Mandarin',
    'S0203': 'Spanish',
    'S0321': 'Thai',
}

LANG2SPLIT = {
    'Arabic': {
        'dev': [5, 36, 107, 164],  # TODO: +6 TBA ?
        'eval': [27, 39, 108, 137],  # TODO: + 6 TBA ?
    },
    'Czech': {  # TODO: TBA ? ; (PZ) I put just one speaker to test the recipe
        'dev': [1],
        'eval': [2],
    },
    'French': {
        'dev': [1],  # TODO: no dev? I put one spk here
        'eval': list(range(91, 99)),  # 91-98
    },
    'Korean': {
        'dev': [6, 12, 25, 40, 45, 61, 84, 86, 91, 98],
        'eval': [19, 29, 32, 42, 51, 64, 69, 80, 82, 88],
    },
    'Mandarin': {
        'dev': list(range(28, 33)) + list(range(39, 45)),  # 28-32, 39-44
        'eval': list(range(80, 90))  # 80-89
    },
    'Spanish': {
        'dev': list(range(1, 11)),  # 1-10
        'eval': list(range(11, 19)),  # 11-18
    },
    'Thai': {
        'dev': [23, 25, 28, 37, 45, 61, 73, 85],
        'eval': list(range(101, 109)),  # 101-108
    },
}

LANG2CODE = {l: c for c, l in CODE2LANG.items()}


class Segment(NamedTuple):
    recording_id: str
    start_seconds: float
    end_seconds: float


class DataDir(NamedTuple):
    wav_scp: Dict[bytes, bytes]
    # segments: Dict[str, Segment]
    text: Dict[bytes, bytes]
    utt2spk: Dict[bytes, bytes]

    def write(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / 'wav.scp', 'wb') as f:
            for k, v in self.wav_scp.items():
                f.write(k + b' ' + v + b'\n')
        with open(data_dir / 'text', 'wb') as f:
            for k, v in self.text.items():
                f.write(k + b' ' + v + b'\n')
        with open(data_dir / 'utt2spk', 'wb') as f:
            for k, v in self.utt2spk.items():
                f.write(k + b' ' + v + b'\n')


class GpDataset(NamedTuple):
    train: DataDir
    dev: DataDir
    eval: DataDir
    lang: str

    def write(self, data_root: Path):
        self.train.write(data_root / f'gp_{self.lang}_train')
        self.dev.write(data_root / f'gp_{self.lang}_dev')
        self.eval.write(data_root / f'gp_{self.lang}_eval')


def main():
    shorten_path = Path('/home/pzelasko/shorten-3.6.1/src/shorten')
    gp_path = Path('/export/corpora5/GlobalPhone/')
    data_dir = Path('data')
    train_languages = 'Arabic Czech French Korean Mandarin Spanish Thai'.split()
    romanized = True

    assert shorten_path.exists()
    assert gp_path.exists()

    for lang in tqdm(train_languages, desc='Preparing per-language data dirs'):
        # We need the following files
        # segments  spk2utt  text  utt2spk  wav.scp
        # TODO: segments
        # TODO: spk2utt
        dataset = parse_gp(gp_path / LANG2CODE[lang] / lang, romanized=romanized, shorten_path=shorten_path, lang=lang)
        dataset.write(data_dir)


def parse_gp(path: Path, shorten_path: Path, lang: str, romanized=True):
    def make_id(path: Path) -> str:
        try:
            stem = path.stem
            if stem.endswith('adc.'):
                stem = stem[:-4] + '.adc'
            id = stem.split('.')[0]
            parts = id.split('_')
            return f'{parts[0]}_UTT{int(parts[1]):03d}'
        except:
            print(path)
            raise

    audio_paths = list((path / 'adc').rglob('*.shn'))
    wav_scp = {
        make_id(p).encode('utf-8'): decompressed(p, shorten_path=shorten_path).encode('utf-8')
        for p in sorted(audio_paths)
    }

    lang_short = next(iter(wav_scp.keys()))[:2]

    tr_sfx = ('rmn' if romanized else 'trl')
    transcript_paths = list((path / tr_sfx).rglob(f'*.{tr_sfx}'))
    text = {}
    utt2spk = {}

    # NOTE: We are using bytes instead of str because some GP transcripts have non-Unicode symbols which fail to parse

    # easy to find parsing errors, as these values should never be used
    utt_id: Optional[bytes] = None  # e.g. AR059_UTT003
    spk_id: Optional[bytes] = None  # e.g. AR059
    for p in sorted(transcript_paths):
        with p.open('rb') as f:
            for line in map(bytes.strip, f):
                m = re.match(rb';SprecherID .*(\d+)', line)
                if m is not None:
                    spk_id = f'{lang_short.decode()}{int(m.group(1).decode()):03d}'.encode('utf-8')
                    continue
                m = re.match(rb'; (\d+):', line)
                if m is not None:
                    utt_id = f'{spk_id.decode()}_UTT{int(m.group(1).decode()):03d}'.encode('utf-8')
                    continue
                assert spk_id is not None
                assert utt_id is not None
                text[utt_id] = line
                utt2spk[utt_id] = spk_id

    def number_of(utt_id):
        try:
            return int(utt_id[2:5])
        except:
            print(utt_id)
            raise

    def select(table, split):
        if split == 'train':
            selected_ids = {
                utt_id for utt_id in text
                if all(
                    number_of(utt_id) not in LANG2SPLIT[lang][split_]
                    for split_ in ('dev', 'eval')
                )
            }
        else:
            selected_ids = {
                utt_id for utt_id in text
                if number_of(utt_id) in LANG2SPLIT[lang][split]
            }
        return {k: v for k, v in table.items() if k in selected_ids}

    return GpDataset(
        train=DataDir(
            wav_scp=select(wav_scp, 'train'),
            utt2spk=select(utt2spk, 'train'),
            text=select(text, 'train')
        ),
        dev=DataDir(
            wav_scp=select(wav_scp, 'dev'),
            utt2spk=select(utt2spk, 'dev'),
            text=select(text, 'dev')
        ),
        eval=DataDir(
            wav_scp=select(wav_scp, 'eval'),
            utt2spk=select(utt2spk, 'eval'),
            text=select(text, 'eval')
        ),
        lang=lang
    )


def decompressed(path: Path, shorten_path: Path) -> str:
    return SHN_DECODE_DECORATOR.format(shorten_path=shorten_path, shn_file=path)


if __name__ == '__main__':
    main()
