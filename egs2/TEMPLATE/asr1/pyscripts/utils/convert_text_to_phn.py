#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Convert kaldi-style text into phonemized sentences."""

import argparse
import codecs
from tqdm import tqdm
import contextlib

from joblib import delayed
from joblib import Parallel, parallel

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
    with tqdm_joblib(
        tqdm(total=len(text.values()), desc="Phonemizing")
    ) as progress_bar:
        phns_list = Parallel(n_jobs=args.nj)(
            [
                delayed(phoneme_tokenizer.text2tokens)(sentence)
                for sentence in text.values()
            ]
        )
    with codecs.open(args.out_text, "w", encoding="utf8") as g:
        for utt_id, phns in zip(text.keys(), phns_list):
            g.write(f"{utt_id} " + " ".join(phns) + "\n")


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report
    into tqdm progress bar given as argument
    """

    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


if __name__ == "__main__":
    main()
