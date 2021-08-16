#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Convert kaldi-style text into phonemized sentences."""

import argparse
import codecs

from joblib import delayed
from joblib import Parallel

from espnet2.text.cleaner import TextCleaner
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer


def main():
    """Run phoneme conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2p", type=str, required=True, help="G2P type.")
    parser.add_argument("--cleaner", type=str, default=None, help="Cleaner type.")
    parser.add_argument("--nj", type=int, default=4, help="Number of parallel jobs.")
    parser.add_argument("in_text", type=str, help="Input kaldi-style text.")
    parser.add_argument("out_text", type=str, help="Output kaldi-style text.")
    args = parser.parse_args()

    phoneme_tokenizer = PhonemeTokenizer(args.g2p)
    cleaner = None
    if args.cleaner is not None:
        cleaner = TextCleaner(args.cleaner)
    with codecs.open(args.in_text, encoding="utf8") as f:
        lines = [line.strip() for line in f.readlines()]
    text = {line.split()[0]: " ".join(line.split()[1:]) for line in lines}
    if cleaner is not None:
        text = {k: cleaner(v) for k, v in text.items()}
    phns_list = Parallel(n_jobs=args.nj)(
        [delayed(phoneme_tokenizer.text2tokens)(sentence) for sentence in text.values()]
    )
    with codecs.open(args.out_text, "w", encoding="utf8") as g:
        for utt_id, phns in zip(text.keys(), phns_list):
            g.write(f"{utt_id} " + " ".join(phns) + "\n")


if __name__ == "__main__":
    main()
