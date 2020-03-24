#!/usr/bin/env python

import re
from itertools import chain, repeat
from pathlib import Path
from typing import NamedTuple, Dict

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

LANG2CODE = {l: c for c, l in CODE2LANG.items()}


class Segment(NamedTuple):
    recording_id: str
    start_seconds: float
    end_seconds: float


class GpDataset(NamedTuple):
    wav_scp: Dict[str, str]
    # segments: Dict[str, Segment]
    text: Dict[bytes, bytes]
    utt2spk: Dict[bytes, bytes]

    def write(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / 'wav.scp', 'w') as f:
            for k, v in self.wav_scp.items():
                print(k, v, file=f)

        with open(data_dir / 'text', 'wb') as f:
            for k, v in self.text.items():
                f.write(k + b' ' + v + b'\n')

        with open(data_dir / 'utt2spk', 'wb') as f:
            for k, v in self.utt2spk.items():
                f.write(k + b' ' + v + b'\n')


def main():
    shorten_path = Path('/home/pzelasko/shorten-3.6.1/src/shorten')
    gp_path = Path('/export/corpora5/GlobalPhone/')
    data_dir = Path('data')
    train_languages = 'Arabic Czech French Korean Mandarin Spanish Thai'.split()
    test_languages = ''.split()
    romanized = True

    assert shorten_path.exists()
    assert gp_path.exists()

    for lang, is_eval in chain(
            zip(train_languages, repeat(False)),
            zip(test_languages, repeat(True))
    ):
        # e.g. Arabic, eval=False -> data/gp_Arabic
        #      Arabic, eval=True  -> data/eval_gp_Arabic
        corpus_dir = data_dir / f'{"eval_" if is_eval else ""}gp_{lang}'

        # We need the following files
        # segments  spk2utt  text  utt2spk  wav.scp
        dataset = parse_gp(gp_path / LANG2CODE[lang] / lang, romanized=romanized, shorten_path=shorten_path)

        # TODO: test/dev/eval split

        dataset.write(corpus_dir)


def parse_gp(path: Path, shorten_path: Path, romanized=True):
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
    wav_scp = {make_id(p): decompressed(p, shorten_path=shorten_path) for p in sorted(audio_paths)}

    lang_short = next(iter(wav_scp.keys()))[:2]

    tr_sfx = ('rmn' if romanized else 'trl')
    transcript_paths = list((path / tr_sfx).rglob(f'*.{tr_sfx}'))
    text = {}
    utt2spk = {}

    # NOTE: We are using bytes instead of str because some GP transcripts have non-Unicode symbols which fail to parse

    # easy to find parsing errors, as these values should never be used
    utt_id: bytes = b'PARSING_ERROR'
    spk_id: bytes = b'PARSING_ERROR'
    for p in sorted(transcript_paths):
        with p.open('rb') as f:
            for line in map(bytes.strip, f):
                m = re.match(rb';SprecherID (\d+)', line)
                if m is not None:
                    spk_id = f'{lang_short}{int(m.group(1).decode()):03d}'.encode('utf-8')
                    continue
                m = re.match(rb'; (\d+):', line)
                if m is not None:
                    utt_id = f'{spk_id.decode()}_UTT{int(m.group(1).decode()):03d}'.encode('utf-8')
                    continue
                text[utt_id] = line
                utt2spk[utt_id] = spk_id

    return GpDataset(wav_scp=wav_scp, utt2spk=utt2spk, text=text)


def decompressed(path: Path, shorten_path: Path) -> str:
    return SHN_DECODE_DECORATOR.format(shorten_path=shorten_path, shn_file=path)


if __name__ == '__main__':
    main()
