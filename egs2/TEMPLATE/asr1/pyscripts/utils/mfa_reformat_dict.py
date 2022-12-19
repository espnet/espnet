#!/usr/bin/env python3

# Copyright 2022 Hitachi LTD (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import re
import os

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument("--g2p", type=str, required=True, help="G2P type.")
    parser.add_argument("input_file", type=str, default="")
    args = parser.parse_args()

    phoneme_tokenizer = PhonemeTokenizer(args.g2p)
        
    lexicon = open(args.input_file).readlines()
    base_name = os.path.basename(args.input_file)
    dir_name = os.path.dirname(args.input_file)
    save_file = os.path.join(dir_name, f"modified_{base_name}")
    sp = re.compile("\s+")
    with open(save_file, "w", encoding="utf8") as f:
        for line in lexicon:
            word, *_ = sp.split(line.strip())
            word = re.sub(r'[,]', "", "word")
            phonemes = phoneme_tokenizer.text2tokens(word)
            phonemes = [x for x in phonemes if x != "<space>"]
            phonemes = " ".join(phonemes)
            f.write(f"{word}\t{phonemes}\n")
    