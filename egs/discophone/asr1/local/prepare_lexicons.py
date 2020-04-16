#!/usr/bin/env python
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from subprocess import run, PIPE, DEVNULL
from tempfile import NamedTemporaryFile

from typing import List, Dict

BABELCODE2LANG = {
    '107': 'Vietnamese',
    '201': 'Haitian',
    '202': 'Swahili',
    '203': 'Lao',
    '204': 'Tamil',
    '205': 'Kurmanji',
    '206': 'Zulu',
    '207': 'Tok-Pisin',
    '301': 'Cebuano',
    '302': 'Kazakh',
    '303': 'Telugu',
    '304': 'Lithuanian',
    '305': 'Guarani',
    '306': 'Igbo',
    '307': 'Amharic',
    '401': 'Mongolian',
    '402': 'Javanese',
    '403': 'Dholuo',
    '404': 'Georgian'
}


def main():
    # noinspection PyTypeChecker
    parser = ArgumentParser(
        description='Prepare LanguageNet IPA phonetic transcripts for a given language and data directory.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--lang', help='Selected language.')
    parser.add_argument('-d', '--data-dir', help='Path to Kaldi data directory with text file.')
    parser.add_argument('-g', '--g2p-models-dir', default='g2ps/models',
                        help='Directory with phonetisaurus g2p FST models for all languages.')
    parser.add_argument('-s', '--substitute-text', action='store_true',
                        help='Will save original text in text.bkp and save the IPA transcript to text.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    g2p_models_dir = Path(args.g2p_models_dir)
    lang = BABELCODE2LANG.get(args.lang, args.lang).lower()
    lang2fst = G2PModelProvider(g2p_models_dir)
    model = lang2fst.get(lang)
    if not model:
        raise ValueError(f'No G2P FST model for language {lang} in {g2p_models_dir}')

    text = data_dir / 'text'
    if not text.exists():
        raise ValueError(f'No such file: {text}')

    lexicon_path = data_dir / 'lexicon_ipa.txt'
    with NamedTemporaryFile('w+') as f:
        uniq_words = run(
            f"cut -f2- -d' ' {text} | tr ' ' '\n' | sort | uniq",
            text=True,
            check=True,
            shell=True,
            stdout=PIPE
        ).stdout
        f.write(uniq_words)
        f.flush()
        lexicon_path.write_text(
            run(
                ['phonetisaurus-g2pfst', f'--model={g2p_models_dir / model}', f'--wordlist={f.name}'],
                check=True,
                text=True,
                stdout=PIPE,
                stderr=DEVNULL
            ).stdout
        )

        lexicon = load_lexicon(lexicon_path)

        text_bkp = text.with_suffix('.bkp')
        shutil.copyfile(text, text_bkp)

        text_ipa = text.with_suffix('.ipa')

        with text_bkp.open() as fin, text_ipa.open('w') as fout:
            for line in fin:
                utt_id, *words = line.strip().split()
                phonetic = [''.join(p for p in map(str.strip, lexicon[w]) if p) for w in words]
                print(utt_id, *[w for w in map(str.strip, phonetic) if w], file=fout)

        if args.substitute_text:
            shutil.copyfile(text_ipa, text)


class G2PModelProvider:
    def __init__(self, g2p_models_dir: Path):
        self.lang2fst = {
            line.split('_')[0]: line
            for line in run(['ls', g2p_models_dir], text=True, check=True, stdout=PIPE).stdout.split()
        }

    def get(self, lang: str) -> str:
        if lang == 'arabic':
            lang = 'gulf-arabic'  # TODO: confirm that GlobalPhone has Gulf Arabic
        return self.lang2fst[lang]


def load_lexicon(p: Path) -> Dict[str, List[str]]:
    lexicon = {}
    with p.open() as f:
        for line in f:
            word, score, *phones = line.strip().split()
            lexicon[word] = phones
    return lexicon


if __name__ == '__main__':
    main()
