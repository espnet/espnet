#!/usr/bin/env python3

"""Convert files to/from MFA format for use in ESPnet TTS.

If you wish to add functions to create .lab files for MFA, add them like this:

def make_labs_[dataset]:
     ...
"""

import argparse
import codecs
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict

import kaldiio
import soundfile as sf
from pyopenjtalk import run_frontend

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

# To Generate Phonemes from words:
# from montreal_forced_aligner.g2p.generator import PyniniValidator
# from montreal_forced_aligner.models import (
#     G2PModel,
#     ModelManager,
# )
# language = "english_us_mfa"
# manager = ModelManager()
# manager.download_model("g2p", language)
# model_path = G2PModel.get_pretrained_path(language)
# g2p = PyniniValidator(g2p_model_path=model_path, num_pronunciations=1, quiet=True)
# g2p.word_list = "my word list".split(" ")
# phones = g2p.generate_pronunciations()

ROOT_DIR = os.getcwd()

WORK_DIR = os.path.join(ROOT_DIR, "data", "local", "mfa")
TEXTGRID_DIR = os.path.join(WORK_DIR, "alignments")
TRAIN_TEXT_PATH = os.path.join(WORK_DIR, "text")
DURATIONS_PATH = os.path.join(WORK_DIR, "durations")
DICTIONARY_PATH = os.path.join(WORK_DIR, "train_dict.txt")

punctuation = '!,.?"'

# JP_DICT_URL =
# "https://raw.githubusercontent.com/r9y9/open_jtalk/1.11/src/mecab-naist-jdic/unidic-csj.csv"


def get_path(s, sep=os.sep):
    for x in s:
        if len(x.split(sep)) > 1:
            return x
    return ""


def get_jp_text(text):
    new_text = run_frontend(text)
    return " ".join([token["string"] for token in new_text])


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Utilities to format from MFA to ESPnet.\n"
            "Usage: python scripts/utils/mfa_format.py TASK [--options]\n"
            "python scripts/utils/mfa_format.py labs\n"
            "python scripts/utils/mfa_format.py validate\n"
            "python scripts/utils/mfa_format.py durations\n"
        )
    )

    parser.add_argument(
        "task",
        choices=["labs", "validate", "durations", "dictionary"],
        help='Must be "labs, "validate" or "durations',
    )
    parser.add_argument(
        "--data_sets",
        help="""List of the data sets (train, dev, eval)
        employed for generating the wavs/labs.""",
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        help="Path to save the corpus in wav/lab format",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=22050,
        help="Sampling rate of the audio files",
    )
    parser.add_argument(
        "--g2p_model",
        type=str,
        help="Name of the grapheme-to-phoneme model from ESPnet.",
    )
    parser.add_argument(
        "--text_cleaner",
        type=str,
        default=None,
        help="Name of the text cleaner from ESPnet.",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        default=256,
        help="The number of shift points.",
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        default=TEXTGRID_DIR,
        help="Path to output MFA .TextGrid files",
    )
    parser.add_argument(
        "--train_text_path",
        type=str,
        default=TRAIN_TEXT_PATH,
        help="Path to output list of utterances to phonemes",
    )
    parser.add_argument(
        "--durations_path",
        type=str,
        default=DURATIONS_PATH,
        help="Path to output durations file",
    )
    parser.add_argument(
        "--dictionary_path",
        type=str,
        default=DICTIONARY_PATH,
        help="Path to output dictionary file",
    )
    return parser


def get_phoneme_durations(
    data: Dict, original_text: str, fs: int, hop_size: int, n_samples: int
):
    """Get phoneme durations."""
    orig_text = original_text.replace(" ", "").rstrip()
    text_pos = 0
    maxTimestamp = data["end"]
    words = [x for x in data["tiers"]["words"]["entries"]]
    phones = (x for x in data["tiers"]["phones"]["entries"])
    word_end = 0.0
    time2punc = {}
    phrase_words = []
    for word in words:
        start, end, label = word
        if start == word_end:
            phrase_words.append(label)
        else:
            # find punctuation at end of previous phrase
            phrase = "".join(phrase_words)
            for letter in phrase:
                char = orig_text[text_pos]
                while char != letter:
                    text_pos += 1
                    char = orig_text[text_pos]
                text_pos += 1

            timing = (word_end, start)
            puncs = []
            while text_pos < len(orig_text):
                char = orig_text[text_pos]
                if char.isalpha() or char == "'":
                    break
                else:
                    puncs.append(char)
                    text_pos += 1
            time2punc[timing] = puncs if puncs else ["sil"]

            phrase_words = [label]
        word_end = end

    # We preserve the start/end timings and not interval lengths
    #   due to rounding errors when converted to frames.
    timings = [0.0]
    new_phones = []
    prev_end = 0.0
    for phone in phones:
        start, end, label = phone
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

        new_phones.append(label)
        timings.append(end)
        prev_end = end

    # Add end-of-utterance punctuations
    if word_end < maxTimestamp:
        text_pos = len(orig_text) - 1
        while orig_text[text_pos] in punctuation:
            text_pos -= 1
        puncs = orig_text[text_pos + 1 :]
        if not puncs:
            puncs = ["sil"]
        new_phones.extend(puncs)
        num_puncs = len(puncs)
        pause_time = (maxTimestamp - word_end) / num_puncs
        for i in range(1, len(puncs) + 1):
            timings.append(prev_end + pause_time * i)

    assert len(new_phones) + 1 == len(timings)

    # Should use same frame formulation for both
    # STFT frames calculation: https://github.com/librosa/librosa/issues/1288
    # centered stft

    total_durations = int(n_samples / hop_size) + 1
    timing_frames = [int(timing * fs / hop_size) + 1 for timing in timings]
    durations = [
        timing_frames[i + 1] - timing_frames[i] for i in range(len(timing_frames) - 1)
    ]

    sum_durations = sum(durations)
    if sum_durations < total_durations:
        missing = total_durations - sum_durations
        durations[-1] += missing
    assert sum(durations) == total_durations
    return new_phones, durations


def validate(args):
    """Validate arguments."""
    valid = True
    filelist = sorted(Path(args.corpus_dir).glob("**/*.json"))
    for _file in filelist:
        # File contains folder of speaker and file
        filename = (
            _file.as_posix().replace(args.corpus_dir + "/", "").replace(".json", "")
        )
        with codecs.open(_file, "r", encoding="utf-8") as reader:
            _data_dict = json.load(reader)
        phones = (x[-1] for x in _data_dict["tiers"]["phones"]["entries"])
        for phone in phones:
            if phone == "spn":
                with open(Path(args.wavs_dir) / f"{filename}.lab") as f:
                    original_text = f.read()
                    logging.error(f"{filename} contains spn. Text: {original_text}")
                valid = False
    assert valid


def make_durations(args):
    """Make durations file."""

    wavs_dir = Path(args.corpus_dir)
    textgrid_dir = args.textgrid_dir
    train_text_path = args.train_text_path
    durations_path = args.durations_path

    os.makedirs(os.path.dirname(train_text_path), exist_ok=True)
    with open(train_text_path, "w") as text_file:
        with open(durations_path, "w") as durations_file:
            lab_paths = sorted(wavs_dir.glob("**/*.lab"))
            assert (
                len(lab_paths) > 0
            ), f"The folder {wavs_dir} does not contain any transcription."
            for lab_path in lab_paths:
                wav_path = lab_path.as_posix().replace(
                    ".lab", ".wav"
                )  # Assumes .wav files are in same dir as .lab files
                if not os.path.exists(wav_path):
                    logging.warning("There is no wav file for %s, skipping.", lab_path)
                    continue

                # get no. of samples and original sr directly from audio file
                with sf.SoundFile(wav_path) as audio:
                    orig_sr = audio.samplerate
                    # Account for downsampling
                    no_samples = int(audio.frames * (args.samplerate / orig_sr))

                filename = (
                    lab_path.as_posix()
                    .replace(args.corpus_dir.rstrip("/") + "/", "")
                    .replace(".lab", "")
                )
                with open(lab_path) as lab_file:
                    original_text = lab_file.read()
                tg_path = os.path.join(textgrid_dir, f"{filename}.json")
                if not os.path.exists(tg_path):
                    logging.warning("There is no alignment for %s, skipping.", lab_path)
                    continue
                with codecs.open(tg_path, "r", encoding="utf-8") as reader:
                    _data_dict = json.load(reader)
                new_phones, durations = get_phoneme_durations(
                    _data_dict,
                    original_text,
                    args.samplerate,
                    args.hop_size,
                    no_samples,
                )
                key = filename.split("/")[-1]
                text_file.write(f'{key} {" ".join(new_phones)}\n')
                durations = " ".join(str(d) for d in durations)
                durations_file.write(f"{key} {durations} 0\n")


def make_dictionary(args):
    """Generate the dictionary of a given corpus using ESPnet text frontend."""
    corpus_dir = Path(args.corpus_dir)
    filelist = sorted(corpus_dir.glob("**/*.lab"))
    words = list()
    for _file in filelist:
        with codecs.open(_file, "r", encoding="utf-8") as reader:
            text = reader.read().rstrip()
        words.extend(text.split())

    phoneme_tokenizer = PhonemeTokenizer(args.g2p_model)
    words = sorted(list(set(words)))

    with codecs.open(args.dictionary_path, "w", encoding="utf-8") as writer:
        for word in words:
            phonemes = phoneme_tokenizer.text2tokens(word)
            phonemes = [x for x in phonemes if (x != "<space>" and len(x) > 0)]
            if len(phonemes) < 1:
                continue
            phonemes = " ".join(phonemes)
            writer.write(f"{word}\t{phonemes}\n")


def make_labs(args):
    """Make lab file for datasets."""

    from espnet2.text.cleaner import TextCleaner

    if not args.text_cleaner:
        args.text_cleaner = None

    corpus_dir = Path(args.corpus_dir)
    cleaner = None
    if args.text_cleaner is not None:
        cleaner = TextCleaner(args.text_cleaner)

    frontend = None
    if args.g2p_model.startswith("pyopenjtalk"):
        frontend = get_jp_text

    for dset in args.data_sets.split():
        logging.info("Preparing data for %s", dset)
        dset: Path = Path("data") / dset
        # Generate directories according to spk2utt
        utt2spk = dict()
        with open(dset / "utt2spk") as reader:
            for line in reader:
                utt, spk = line.strip().split(maxsplit=1)
                utt2spk[utt] = spk
            for spk in set(utt2spk.values()):
                (corpus_dir / spk).mkdir(parents=True, exist_ok=True)

        # Generate labs according to text file
        with open(dset / "text", encoding="utf-8") as reader:
            for line in reader:
                utt, text = line.strip().split(maxsplit=1)
                if cleaner is not None:
                    text = cleaner(text).lower()
                else:
                    text = text.lower()
                # Convert single quotes into double quotes
                #   so that MFA doesn't confuse them with clitics.
                # Find ' not preceded by a letter to the last ' not followed by a letter
                text = re.sub(r"(\W|^)'(\w[\w .,!?']*)'(\W|$)", r'\1"\2"\3', text)

                # Remove braces because MFA interprets them as enclosing a single word
                text = re.sub(r"[\{\}]", "", text)

                # In case of frontend, preprocess data.
                if frontend is not None:
                    text = frontend(text)

                try:
                    spk = utt2spk[utt]
                    with open(
                        corpus_dir / spk / f"{utt}.lab", "w", encoding="utf-8"
                    ) as writer:
                        writer.write(text)
                except KeyError:
                    logging.warning(f"{utt} is in text file but not in utt2spk")

        # Generate wavs according to wav.scp and segment files
        if (dset / "segments").exists():
            wscp = (dset / "wav.scp").as_posix()
            segments = (dset / "segments").as_posix()
            with kaldiio.ReadHelper(f"scp:{wscp}", segments=segments) as reader:
                for utt, (rate, array) in reader:
                    try:
                        spk = utt2spk[utt]
                        dst_file = (corpus_dir / spk / f"{utt}.wav").as_posix()
                        sf.write(dst_file, array, rate)
                    except KeyError:
                        logging.warning(f"{utt} is in wav.scp file but not in utt2spk")
        else:
            with open(dset / "wav.scp") as reader:
                for line in reader:
                    utt, src_file = line.strip().split(maxsplit=1)
                    src_file = os.path.abspath(src_file)
                    try:
                        spk = utt2spk[utt]
                        dst_file = corpus_dir / spk / f"{utt}.wav"
                        if src_file.endswith(".wav"):
                            # Create symlink
                            dst_file.symlink_to(src_file)
                        else:
                            # Create wav file
                            rate, array = kaldiio.load_mat(src_file)
                            sf.write(dst_file.as_posix(), array, rate)
                    except KeyError:
                        logging.warning(f"{utt} is in wav.scp file but not in utt2spk")
    logging.info("Finished writing .lab files")


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    try:
        if args.task == "labs":
            make_labs(args)
        elif args.task == "validate":
            validate(args)
        elif args.task == "dictionary":
            make_dictionary(args)
        elif args.task == "durations":
            make_durations(args)
        else:
            raise NotImplementedError
    except Exception:
        traceback.print_exc()
        sys.exit(1)
