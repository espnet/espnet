#!/usr/bin/env python3

"""Convert files to/from MFA format for use in ESPnet TTS.

If you wish to add functions to create .lab files for MFA, add them like this:

def make_labs_[dataset]:
     ...
"""

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

from praatio import textgrid
from tacotron_cleaner import cleaners

ROOT_DIR = os.getcwd()

TEXTGRID_DIR = f"{ROOT_DIR}/textgrids"
TRAIN_TEXT_PATH = f"{ROOT_DIR}/data/train/text"
DURATIONS_PATH = f"{ROOT_DIR}/data/train/durations"

punctuation = "!',.?" + '"'

fs = 22050
hop_length = 256


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Utilities to format from MFA to ESPnet.\n"
            "Usage: python scripts/utils/mfa_format.py ACTION [dataset] [options]\n"
            "python scripts/utils/mfa_format.py labs ljspeech\n"
            "python scripts/utils/mfa_format.py validate\n"
            "python scripts/utils/mfa_format.py durations\n"
        )
    )

    parser.add_argument(
        "type",
        choices=["labs", "validate", "durations"],
        help='Must be "labs, "validate" or "durations',
    )
    parser.add_argument(
        "--dataset",
        choices=["ljspeech"],
        help='Currently only supports "ljspeech"',
    )
    parser.add_argument(
        "--wavs_dir",
        help="Path to wavs dir, lab files will be added here",
    )
    parser.add_argument(
        "--textgrid_dir",
        default=TEXTGRID_DIR,
        help="Path to output MFA .TextGrid files",
    )
    parser.add_argument(
        "--train_text_path",
        default=TRAIN_TEXT_PATH,
        help="Path to output list of utterances to phonemes",
    )
    parser.add_argument(
        "--durations_path",
        default=DURATIONS_PATH,
        help="Path to output durations file",
    )

    return parser


def make_labs_ljspeech(args):
    """Make lab file for ljspeech dataset."""
    wavs_dir = Path(args.wavs_dir)
    with open(wavs_dir / "../metadata.csv") as f:
        for line in f:
            items = line.rstrip().split("|")
            filename = items[0]
            text = items[-1]
            with open(wavs_dir / f"{filename}.lab", "w") as lab_f:
                # Same as custom_english_cleaners but without uppercasing.
                text = cleaners.convert_to_ascii(text)
                text = cleaners.lowercase(text)
                text = cleaners.expand_numbers(text)
                text = cleaners.expand_abbreviations(text)
                text = cleaners.expand_symbols(text)
                text = cleaners.remove_unnecessary_symbols(text)
                text = cleaners.collapse_whitespace(text)
                # Convert single quotes into double quotes
                #   so that MFA doesn't confuse them with clitics.
                # Find ' not preceded by a letter to the last ' not followed by a letter
                text = re.sub(r"(\W|^)'(\w[\w .,!?']*)'(\W|$)", r'\1"\2"\3', text)
                lab_f.write(text)
                print(filename)
    print("Finished writing .lab files")


def get_phoneme_durations(tg: textgrid.Textgrid, original_text: str):
    """Get phohene durations."""
    orig_text = original_text.replace(" ", "").rstrip()
    text_pos = 0

    words = tg.tierDict["words"].entryList
    phones = tg.tierDict["phones"].entryList

    prev_end = 0.0
    time2punc = {}
    phrase_words = []
    word = None
    for word in words:
        start = word.start
        if start == prev_end:
            phrase_words.append(word.label)
        else:
            # find punctuation at end of previous phrase
            phrase = "".join(phrase_words)
            for letter in phrase:
                char = orig_text[text_pos]
                while char != letter:
                    text_pos += 1
                    char = orig_text[text_pos]
                text_pos += 1

            timing = (prev_end, start)
            puncs = []
            while text_pos < len(orig_text):
                char = orig_text[text_pos]
                if char.isalpha():
                    break
                else:
                    puncs.append(char)
                    text_pos += 1
            time2punc[timing] = puncs if puncs else ["sil"]

            phrase_words = [word.label]
        prev_end = word.end

    # We preserve the start/end timings and not interval lengths
    #   due to rounding errors when converted to frames.
    timings = [0.0]
    new_phones = []
    prev_end = 0.0
    for phone in phones:
        start = phone.start
        if start > prev_end:
            # insert punctuation
            try:
                puncs = time2punc[(prev_end, start)]
            except KeyError:
                # In some cases MFA word segmentation fails
                #   and there is a pause inside the word.
                puncs = ["sil"]
            new_phones.extend(puncs)
            num_puncs = len(puncs)
            pause_time = (start - prev_end) / num_puncs
            for i in range(1, len(puncs) + 1):
                timings.append(prev_end + pause_time * i)

        new_phones.append(phone.label)
        timings.append(phone.end)
        prev_end = phone.end

    # Add end-of-utterance punctuations
    if word.end < tg.maxTimestamp:
        text_pos = len(orig_text) - 1
        while orig_text[text_pos] in punctuation:
            text_pos -= 1
        puncs = orig_text[text_pos + 1 :]
        if not puncs:
            puncs = ["sil"]
        new_phones.extend(puncs)
        num_puncs = len(puncs)
        pause_time = (tg.maxTimestamp - word.end) / num_puncs
        for i in range(1, len(puncs) + 1):
            timings.append(prev_end + pause_time * i)

    assert len(new_phones) + 1 == len(timings)

    total_durations = int(tg.maxTimestamp * fs) // hop_length + 1
    timing_frames = [int(timing * fs / hop_length + 0.5) for timing in timings]
    durations = [
        timing_frames[i + 1] - timing_frames[i] for i in range(len(timing_frames) - 1)
    ]

    assert sum(durations) == total_durations

    return new_phones, durations


def validate(args):
    """Validate arguments."""
    valid = True
    tg_paths = sorted(Path(args.textgrid_dir).glob("*.TextGrid"))
    for tg_path in tg_paths:
        filename = tg_path.stem
        tg = textgrid.openTextgrid(tg_path, False)
        phones = tg.tierDict["phones"].entryList
        for phone in phones:
            if phone.label == "spn":
                with open(Path(args.wavs_dir) / f"{filename}.lab") as f:
                    original_text = f.read()
                    print(f"{filename} contains spn. Text: {original_text}")
                valid = False
    assert valid


def make_durations(args):
    """Make durations file."""
    wavs_dir = Path(args.wavs_dir)
    textgrid_dir = args.textgrid_dir
    train_text_path = args.train_text_path
    durations_path = args.durations_path

    os.makedirs(os.path.dirname(train_text_path), exist_ok=True)
    with open(train_text_path, "w") as text_file:
        with open(durations_path, "w") as durations_file:
            lab_paths = sorted(wavs_dir.glob("*.lab"))
            for lab_path in lab_paths:
                filename = lab_path.stem
                with open(lab_path) as lab_file:
                    original_text = lab_file.read()
                tg_path = os.path.join(textgrid_dir, f"{filename}.TextGrid")
                tg = textgrid.openTextgrid(tg_path, False)
                new_phones, durations = get_phoneme_durations(tg, original_text)
                text_file.write(f'{filename} {" ".join(new_phones)}\n')
                durations = " ".join(str(d) for d in durations)
                durations_file.write(f"{filename} {durations} 0\n")


if __name__ == "__main__":
    args = get_parser().parse_args()
    try:
        if args.type == "labs":
            func = globals()[f"make_labs_{args.dataset}"]
            func(args)
        elif args.type == "validate":
            validate(args)
        elif args.type == "durations":
            make_durations(args)
        else:
            raise NotImplementedError
    except Exception:
        traceback.print_exc()
        sys.exit(1)
