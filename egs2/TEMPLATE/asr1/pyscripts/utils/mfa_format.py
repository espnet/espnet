#!/usr/bin/env python3

"""Convert files to/from MFA format for use in ESPnet TTS."""

import argparse
import codecs
import json
import logging
import os
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import Dict, Union

import kaldiio
import regex as re
import soundfile as sf
from praatio.data_classes.textgrid import Textgrid
from praatio.textgrid import openTextgrid
from pyopenjtalk import run_frontend

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.utils.types import str_or_none

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
            "Usage: python pyscripts/utils/mfa_format.py TASK [--options]\n"
            "Examples:"
            "   python pyscripts/utils/mfa_format.py labs\n"
            "       --data_sets train dev eval\n"
            "       --text_cleaner mfa_english\n"
            "       --g2p_model english_us_mfa\n"
            "   python pyscripts/utils/mfa_format.py validate\n"
            "   python pyscripts/utils/mfa_format.py durations\n"
            "   python pyscripts/utils/mfa_format.py dictionary "
            "       --g2p_model g2p_en\n"
        )
    )

    parser.add_argument(
        "task",
        choices=["labs", "validate", "durations", "dictionary"],
        help='Must be "labs, "validate", "durations" or "dictionary"',
    )
    parser.add_argument(
        "--data_sets",
        type=str,
        help="""List of the data sets (train, dev, eval)
        employed for generating the wavs/labs.""",
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default=WORK_DIR,
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
        help="Grapheme to phoneme model from ESPnet PhonemeTokenizer.",
    )
    parser.add_argument(
        "--text_cleaner",
        type=str_or_none,
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
        help="Path to output MFA alignment files",
    )
    parser.add_argument(
        "--textgrid_format",
        type=str,
        default="json",
        choices=["json", "long_textgrid"],
        help="Either 'json' or 'long_textgrid'. Default is json because it's "
        "more compact but if you want to view the alignment Praat "
        "requires the regular textgrid format",
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
    data: Union[Dict, Textgrid],
    original_text: str,
    fs: int,
    hop_size: int,
    n_samples: int,
):
    """Get phoneme durations."""
    orig_text = original_text.replace(" ", "").rstrip()
    text_pos = 0

    if isinstance(data, dict):
        max_timestamp = data["end"]
        word_intervals = [x for x in data["tiers"]["words"]["entries"]]
        phone_intervals = (x for x in data["tiers"]["phones"]["entries"])
    elif isinstance(data, Textgrid):
        max_timestamp = data.maxTimestamp
        word_intervals = data.tiers[0].entries
        phone_intervals = data.tiers[1].entries
    else:
        raise ValueError(f"data must be either dict or Textgrid")

    word_end = 0.0
    time2punc = {}
    phrase_words = []
    for word_interval in word_intervals:
        start, end, word = word_interval
        if start == word_end:
            phrase_words.append(word)
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

            phrase_words = [word]
        word_end = end

    # We preserve the start/end timings and not interval lengths
    #   due to rounding errors when converted to frames.
    timings = [0.0]
    new_phones = []
    prev_end = 0.0
    for phone_interval in phone_intervals:
        start, end, phone = phone_interval
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

        new_phones.append(phone)
        timings.append(end)
        prev_end = end

    # Add end-of-utterance punctuations
    if word_end < max_timestamp:
        text_pos = len(orig_text) - 1
        while orig_text[text_pos] in punctuation:
            text_pos -= 1
        puncs = orig_text[text_pos + 1 :]
        if not puncs:
            puncs = ["sil"]
        new_phones.extend(puncs)
        num_puncs = len(puncs)
        pause_time = (max_timestamp - word_end) / num_puncs
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
    is_json = args.textgrid_format == "json"
    fmt = "json" if is_json else "TextGrid"
    filelist = sorted(Path(args.corpus_dir).glob(f"**/*.{fmt}"))
    for _file in filelist:
        # File contains folder of speaker and file
        filename = (
            _file.as_posix().replace(args.corpus_dir + "/", "").replace(f".{fmt}", "")
        )
        if is_json:
            with codecs.open(_file, "r", encoding="utf-8") as reader:
                _data = json.load(reader)
            phones = (x[-1] for x in _data["tiers"]["phones"]["entries"])
        else:
            _data = openTextgrid(_file, False)
            phones = (x[-1] for x in _data.tiers[1].entries)
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
    is_json = args.textgrid_format == "json"
    fmt = "json" if is_json else "TextGrid"
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

                tg_path = os.path.join(textgrid_dir, f"{filename}.{fmt}")
                if not os.path.exists(tg_path):
                    logging.warning("There is no alignment for %s, skipping.", lab_path)
                    continue
                if is_json:
                    with codecs.open(tg_path, "r", encoding="utf-8") as reader:
                        _data = json.load(reader)
                else:
                    _data = openTextgrid(tg_path, False)
                new_phones, durations = get_phoneme_durations(
                    _data,
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
    phoneme_tokenizer = PhonemeTokenizer(args.g2p_model)

    # TODO: dynamically find the word separator, here assumed as |
    word_sep = "|"

    if args.g2p_model.startswith("espeak_ng_english"):
        strings, replacements, compound_words = espeak_english_replacements(word_sep)
        text_hacks = partial(
            multi_replace_text, strings=strings, replacements=replacements
        )
        phoneme_hacks = partial(merge_phonemes, compound_words=compound_words)
    else:

        def text_hacks(text):
            return text

        def phoneme_hacks(words, prons):
            return words, prons

    # We need to generate all possible pronunciations for each word.
    # For example the "a" in "a, b, c" and "a man" are pronounced differently.
    word_regex = re.compile(r"[\p{L}][\p{L}']*")
    punc_regex = re.compile(r"[\p{P}" + word_sep + "]+")

    dict_lines = set()
    with codecs.open(args.dictionary_path, "w", encoding="utf-8") as writer:
        for _file in filelist:
            with codecs.open(_file, "r", encoding="utf-8") as reader:
                text = reader.read().rstrip()
            words = word_regex.findall(text)
            phonemes = phoneme_tokenizer.text2tokens(text_hacks(text))
            prons = []
            start = 0
            stop = 0
            for phoneme in phonemes:
                if punc_regex.fullmatch(phoneme):
                    if start < stop:
                        prons.append(" ".join(phonemes[start:stop]))
                    start = stop + 1
                stop += 1
            words, prons = phoneme_hacks(words, prons)
            for i, pron in enumerate(
                prons
            ):  # Ensure no punctuation marks in pronunciation
                prons[i] = re.sub(r"\p{P}", "", pron)

            assert len(prons) == len(words), (
                f"Word and pronunciation counts do not match at {_file}\n"
                f"Prons: {prons}\n"
                f"Words: {words}"
            )
            for word, pron in zip(words, prons):
                dict_lines.add(f"{word}\t{pron}")

        writer.write("\n".join(sorted(dict_lines)) + "\n")


def multi_replace_text(text, strings, replacements):
    for string, replacement in zip(strings, replacements):
        text = text.replace(string, replacement)
    return text


def merge_phonemes(words, prons, compound_words):
    for i, word in enumerate(words):
        if word in compound_words:
            prons = prons[:i] + [" ".join(prons[i : i + 2])] + prons[i + 2 :]
    return words, prons


def espeak_english_replacements(word_sep):
    # Unfortunately espeak combines many phrases as one word,
    # causing alignments to fail. Some words are also split up,
    # e.g. lunchroom is treated as two words.
    # We have to ensure the same word count before and after phonemization.
    # See https://github.com/espeak-ng/espeak-ng/issues/1841
    import espnet2

    project_root = Path(espnet2.__file__).parent.parent
    rules_file = project_root / "tools" / "espeak-ng" / "dictsource" / "en_list"
    phrase_regex = re.compile(r"\n\([A-Za-z' ]+\)")
    with open(rules_file, "r") as f:
        rules = f.read()
    phrases = phrase_regex.findall(rules)
    phrases = [phrase[2:-1].lower() for phrase in phrases]
    replacements = [phrase.replace(" ", f" {word_sep} ") for phrase in phrases]
    compound_regex = re.compile(r"\n([A-Za-z']+)\t+.+\|\|.+\n")
    compound_words = compound_regex.findall(rules)
    compound_words = set(w.lower() for w in compound_words)
    return phrases, replacements, compound_words


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
        # Generate directories according to utt2spk
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
                if cleaner is None:
                    text = text.lower()
                else:
                    text = cleaner(text)
                # Convert single quotes into double quotes so that MFA doesn't
                # confuse them with clitics.
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
