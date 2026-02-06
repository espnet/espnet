"""
This script filters out utterances with mismatched language labels
(audio, text, and original label).
Reference: Section 2.1.2 in the paper (https://arxiv.org/pdf/2506.00338)
"""

import json
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from utils import TO_ISO_LANGUAGE_CODE


def norm_speech_lang(lang):
    lang_mappings = {
        "iw": "he",
        "jw": "jv",
    }
    if lang in lang_mappings:
        lang = lang_mappings[lang]

    if lang in TO_ISO_LANGUAGE_CODE:
        return TO_ISO_LANGUAGE_CODE[lang]

    return lang


def norm_text_lang(lang):
    if len(lang) == 2 and lang in TO_ISO_LANGUAGE_CODE:
        return TO_ISO_LANGUAGE_CODE[lang]

    return lang


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True, help="Input file path")
    parser.add_argument("--out_file", type=str, required=True, help="Output file path")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    in_file = args.in_file
    out_file = args.out_file

    with open(in_file, "r") as fin, open(out_file, "w") as fout:
        for line in tqdm(fin):
            sample = json.loads(line.strip())
            lang = sample["lang"]
            speech_lang = norm_speech_lang(sample["speech_pred"])
            text_lang = norm_text_lang(sample["text_pred"])
            prevtext_lang = norm_text_lang(sample["prev_text_pred"])

            if prevtext_lang == "<na>":
                prevtext_lang = text_lang

            if lang == text_lang and lang == prevtext_lang and lang == speech_lang:
                fout.write(sample["utt_id"] + "\n")
