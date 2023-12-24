import argparse
import glob
import logging
import os
import re
import shutil
from typing import List, TextIO, Tuple

import librosa
import miditoolkit
import numpy as np

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "pydub needed for audio segmentation" "use pip to install pydub"
    )
try:
    import pretty_midi
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "pretty_midi needed for midi data loading" "use pip to install pretty_midi"
    )

from tqdm import tqdm

with open(
    os.path.join("local", "lexicon-en_tone.phones"), "r", encoding="latin1"
) as fid:
    en_lines = fid.readlines()
en_dict = dict(
    [
        (split[0].upper(), split[1:])
        for split in [line.strip().split() for line in en_lines]
    ]
)


def check_en(word):
    return not word[-1].isdigit() and word.upper() in en_dict


from espnet2.fileio.score_scp import SingingScoreWriter

"""Split audio into segments according to structured annotation.""" ""
"""Generate segments according to structured annotation."""
"""Transfer music score into 'score' format."""


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)
    os.makedirs(data_url)


def load_midi_note_scp(midi_note_scp):
    # Note(jiatong): midi format is 0-based
    midi_mapping = {}
    with open(midi_note_scp, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split("\t")
            midi_mapping[key[1]] = int(key[0])
    return midi_mapping


def load_midi(root_path):
    """
    Loads MIDI files from subdirectories in the given root path and extracts tempo information.

    :param root_path: The root directory containing subdirectories with MIDI and WAV files.
    :return: A dictionary mapping each song id (subdirectory name) to its tempo.
    """
    midis = {}

    for subdir in os.listdir(root_path):
        if subdir == "segments" or "-unseg" in subdir:
            continue

        full_subdir_path = os.path.join(root_path, subdir)
        if os.path.isdir(full_subdir_path):
            midi_files = glob.glob(os.path.join(full_subdir_path, "*.mid"))
            if not midi_files:
                print(f"No MIDI files found in {full_subdir_path}")
                continue

            midi_file = midi_files[0]
            try:
                midi_obj = miditoolkit.midi.parser.MidiFile(midi_file)
                tempos = midi_obj.tempo_changes
                tempos.sort(key=lambda x: (x.time, x.tempo))

                if tempos:
                    tempo = int(tempos[0].tempo + 0.5)
                    midis[subdir] = tempo
                else:
                    print(f"No tempo information found in {midi_file}")
            except Exception as e:
                print(f"Error processing {midi_file}: {e}")

    return midis


def get_partitions(input_midi: str, threshold=2.0) -> List[Tuple[float, float]]:
    """ "
    Get partition points [(start, end)] that detach at rests longer than the specified threshold.
    Note: silence truncated when longer than 0.4s.

    Parameters
    -------
    input_midi : str
        Path to the MIDI file.
    threshold : float
        The threshold in beats. Rests longer than this threshold will trigger detachment.

    Returns
    -------
    partitions : List[Tuple[float, float]]
        The partition points. Each entry of the list is a tuple of (start, end)
    """
    midi_data = pretty_midi.PrettyMIDI(input_midi)
    bpm = midi_data.estimate_tempo()
    second_per_beat = 60 / bpm
    thresh_in_second = threshold * second_per_beat

    if len(midi_data.instruments) != 1:
        print(f"Unexpected for kising. Multi-track found: {input_midi}!")
        exit(1)

    partitions = []  # [(start, end)]
    instrument = midi_data.instruments[0]  # i.e., just a track
    notes = instrument.notes
    seg_start = max(notes[0].start - 0.2, 0.0)
    prev_note_end = notes[0].end

    for note in notes[1:]:
        if note.start - prev_note_end >= thresh_in_second:
            partitions.append(
                (seg_start, min((prev_note_end + note.start) / 2, prev_note_end + 0.2))
            )
            seg_start = max(note.start - 0.2, (prev_note_end + note.start) / 2)
        prev_note_end = note.end

    # Note: the end of the last partition can be larger than the corresponding audio length. But does not matter
    partitions.append((seg_start, notes[-1].end + 0.2))

    return partitions


def save_wav_segments_from_partitions(path, input_wav, partitions, songid, singer):
    """
    Partition wav files based on 'partitions' and save the segmented files in 'outdir'.
    """

    audio = AudioSegment.from_wav(input_wav)
    segids = []
    wav_scp = []

    for i, (start_time, end_time) in enumerate(partitions, 1):
        start_time_ms = int(start_time * 1000)
        end_time_ms = int(end_time * 1000)

        # Note: end_time_ms can be larger than the actual audio length. But does not matter here
        segment = audio[start_time_ms:end_time_ms]
        segid = f"{songid}_{i:03}_{singer}"
        outfile = os.path.join(path, f"{segid}.wav")
        segment.export(outfile, format="wav")
        segids.append(segid)
        wav_scp.append((segid, outfile))

    return segids, wav_scp


def get_info_from_partitions(
    input_midi: str, partitions: List[Tuple[float, float]]
) -> Tuple[
    List[List[str]],
    List[List[str]],
    List[List[float]],
    List[List[float]],
    List[List[int]],
]:
    midi_data = pretty_midi.PrettyMIDI(input_midi)
    current_partition = 0
    text = [[]]
    phonemes = [[]]
    pitches = [[]]
    note_durations = [[]]
    phn_durations = [[]]
    is_slur = [[]]

    # Function to split Pinyin into shengmu and yunmu
    def split_pinyin(pinyin):
        PINYIN_DICT = {
            "a": ("^", "a"),
            "ai": ("^", "ai"),
            "an": ("^", "an"),
            "ang": ("^", "ang"),
            "ao": ("^", "ao"),
            "ba": ("b", "a"),
            "bai": ("b", "ai"),
            "ban": ("b", "an"),
            "bang": ("b", "ang"),
            "bao": ("b", "ao"),
            "be": ("b", "e"),
            "bei": ("b", "ei"),
            "ben": ("b", "en"),
            "beng": ("b", "eng"),
            "bi": ("b", "i"),
            "bian": ("b", "ian"),
            "biao": ("b", "iao"),
            "bie": ("b", "ie"),
            "bin": ("b", "in"),
            "bing": ("b", "ing"),
            "bo": ("b", "o"),
            "bu": ("b", "u"),
            "ca": ("c", "a"),
            "cai": ("c", "ai"),
            "can": ("c", "an"),
            "cang": ("c", "ang"),
            "cao": ("c", "ao"),
            "ce": ("c", "e"),
            "cen": ("c", "en"),
            "ceng": ("c", "eng"),
            "cha": ("ch", "a"),
            "chai": ("ch", "ai"),
            "chan": ("ch", "an"),
            "chang": ("ch", "ang"),
            "chao": ("ch", "ao"),
            "che": ("ch", "e"),
            "chen": ("ch", "en"),
            "cheng": ("ch", "eng"),
            "chi": ("ch", "iii"),
            "chong": ("ch", "ong"),
            "chou": ("ch", "ou"),
            "chu": ("ch", "u"),
            "chua": ("ch", "ua"),
            "chuai": ("ch", "uai"),
            "chuan": ("ch", "uan"),
            "chuang": ("ch", "uang"),
            "chui": ("ch", "uei"),
            "chun": ("ch", "uen"),
            "chuo": ("ch", "uo"),
            "ci": ("c", "ii"),
            "cong": ("c", "ong"),
            "cou": ("c", "ou"),
            "cu": ("c", "u"),
            "cuan": ("c", "uan"),
            "cui": ("c", "uei"),
            "cun": ("c", "uen"),
            "cuo": ("c", "uo"),
            "da": ("d", "a"),
            "dai": ("d", "ai"),
            "dan": ("d", "an"),
            "dang": ("d", "ang"),
            "dao": ("d", "ao"),
            "de": ("d", "e"),
            "dei": ("d", "ei"),
            "den": ("d", "en"),
            "deng": ("d", "eng"),
            "di": ("d", "i"),
            "dia": ("d", "ia"),
            "dian": ("d", "ian"),
            "diao": ("d", "iao"),
            "die": ("d", "ie"),
            "ding": ("d", "ing"),
            "din": ("d", "in"),  # non 普通话
            "diu": ("d", "iou"),
            "dong": ("d", "ong"),
            "dou": ("d", "ou"),
            "du": ("d", "u"),
            "duan": ("d", "uan"),
            "dui": ("d", "uei"),
            "dun": ("d", "uen"),
            "duo": ("d", "uo"),
            "e": ("^", "e"),
            "ei": ("^", "ei"),
            "en": ("^", "en"),
            "ng": ("^", "en"),
            "eng": ("^", "eng"),
            "er": ("^", "er"),
            "fa": ("f", "a"),
            "fan": ("f", "an"),
            "fang": ("f", "ang"),
            "fei": ("f", "ei"),
            "fen": ("f", "en"),
            "feng": ("f", "eng"),
            "fo": ("f", "o"),
            "fou": ("f", "ou"),
            "fu": ("f", "u"),
            "ga": ("g", "a"),
            "gai": ("g", "ai"),
            "gan": ("g", "an"),
            "gang": ("g", "ang"),
            "gao": ("g", "ao"),
            "ge": ("g", "e"),
            "gei": ("g", "ei"),
            "gen": ("g", "en"),
            "geng": ("g", "eng"),
            "gong": ("g", "ong"),
            "gou": ("g", "ou"),
            "gu": ("g", "u"),
            "gua": ("g", "ua"),
            "guai": ("g", "uai"),
            "guan": ("g", "uan"),
            "guang": ("g", "uang"),
            "gui": ("g", "uei"),
            "gun": ("g", "uen"),
            "guo": ("g", "uo"),
            "ha": ("h", "a"),
            "hai": ("h", "ai"),
            "han": ("h", "an"),
            "hang": ("h", "ang"),
            "hao": ("h", "ao"),
            "he": ("h", "e"),
            "hei": ("h", "ei"),
            "hen": ("h", "en"),
            "heng": ("h", "eng"),
            "hong": ("h", "ong"),
            "hou": ("h", "ou"),
            "hu": ("h", "u"),
            "hua": ("h", "ua"),
            "huai": ("h", "uai"),
            "huan": ("h", "uan"),
            "huang": ("h", "uang"),
            "hui": ("h", "uei"),
            "hun": ("h", "uen"),
            "huo": ("h", "uo"),
            "ji": ("j", "i"),
            "jia": ("j", "ia"),
            "jian": ("j", "ian"),
            "jiang": ("j", "iang"),
            "jiao": ("j", "iao"),
            "jie": ("j", "ie"),
            "jin": ("j", "in"),
            "jing": ("j", "ing"),
            "jiong": ("j", "iong"),
            "jiu": ("j", "iou"),
            "ju": ("j", "v"),
            "jv": ("j", "v"),  # alternative spelling of ju
            "juan": ("j", "van"),
            "jue": ("j", "ve"),
            "jun": ("j", "vn"),
            "ka": ("k", "a"),
            "kai": ("k", "ai"),
            "kan": ("k", "an"),
            "kang": ("k", "ang"),
            "kao": ("k", "ao"),
            "ke": ("k", "e"),
            "kei": ("k", "ei"),
            "ken": ("k", "en"),
            "keng": ("k", "eng"),
            "kong": ("k", "ong"),
            "kou": ("k", "ou"),
            "ku": ("k", "u"),
            "kua": ("k", "ua"),
            "kuai": ("k", "uai"),
            "kuan": ("k", "uan"),
            "kuang": ("k", "uang"),
            "kui": ("k", "uei"),
            "kun": ("k", "uen"),
            "kuo": ("k", "uo"),
            "la": ("l", "a"),
            "lai": ("l", "ai"),
            "lan": ("l", "an"),
            "lang": ("l", "ang"),
            "lao": ("l", "ao"),
            "le": ("l", "e"),
            "lei": ("l", "ei"),
            "leng": ("l", "eng"),
            "li": ("l", "i"),
            "lia": ("l", "ia"),
            "lian": ("l", "ian"),
            "liang": ("l", "iang"),
            "liao": ("l", "iao"),
            "lie": ("l", "ie"),
            "lin": ("l", "in"),
            "ling": ("l", "ing"),
            "liu": ("l", "iou"),
            "lo": ("l", "o"),
            "long": ("l", "ong"),
            "lou": ("l", "ou"),
            "lu": ("l", "u"),
            "lv": ("l", "v"),
            "luan": ("l", "uan"),
            "lve": ("l", "ve"),
            "lue": ("l", "ve"),
            "lun": ("l", "uen"),
            "luo": ("l", "uo"),
            "ma": ("m", "a"),
            "mai": ("m", "ai"),
            "man": ("m", "an"),
            "mang": ("m", "ang"),
            "mao": ("m", "ao"),
            "me": ("m", "e"),
            "mei": ("m", "ei"),
            "men": ("m", "en"),
            "meng": ("m", "eng"),
            "mi": ("m", "i"),
            "mian": ("m", "ian"),
            "miao": ("m", "iao"),
            "mie": ("m", "ie"),
            "min": ("m", "in"),
            "ming": ("m", "ing"),
            "miu": ("m", "iou"),
            "mo": ("m", "o"),
            "mou": ("m", "ou"),
            "mu": ("m", "u"),
            "na": ("n", "a"),
            "nai": ("n", "ai"),
            "nan": ("n", "an"),
            "nang": ("n", "ang"),
            "nao": ("n", "ao"),
            "ne": ("n", "e"),
            "nei": ("n", "ei"),
            "nen": ("n", "en"),
            "neng": ("n", "eng"),
            "ni": ("n", "i"),
            "nia": ("n", "ia"),
            "nian": ("n", "ian"),
            "niang": ("n", "iang"),
            "niao": ("n", "iao"),
            "nie": ("n", "ie"),
            "nin": ("n", "in"),
            "ning": ("n", "ing"),
            "niu": ("n", "iou"),
            "nong": ("n", "ong"),
            "nou": ("n", "ou"),
            "nu": ("n", "u"),
            "nv": ("n", "v"),
            "nuan": ("n", "uan"),
            "nve": ("n", "ve"),
            "nue": ("n", "ve"),
            "nuo": ("n", "uo"),
            "o": ("^", "o"),
            "ou": ("^", "ou"),
            "pa": ("p", "a"),
            "pai": ("p", "ai"),
            "pan": ("p", "an"),
            "pang": ("p", "ang"),
            "pao": ("p", "ao"),
            "pe": ("p", "e"),
            "pei": ("p", "ei"),
            "pen": ("p", "en"),
            "peng": ("p", "eng"),
            "pi": ("p", "i"),
            "pian": ("p", "ian"),
            "piao": ("p", "iao"),
            "pie": ("p", "ie"),
            "pin": ("p", "in"),
            "ping": ("p", "ing"),
            "po": ("p", "o"),
            "pou": ("p", "ou"),
            "pu": ("p", "u"),
            "qi": ("q", "i"),
            "qia": ("q", "ia"),
            "qian": ("q", "ian"),
            "qiang": ("q", "iang"),
            "qiao": ("q", "iao"),
            "qie": ("q", "ie"),
            "qin": ("q", "in"),
            "qing": ("q", "ing"),
            "qiong": ("q", "iong"),
            "qiu": ("q", "iou"),
            "qu": ("q", "v"),
            "quan": ("q", "van"),
            "que": ("q", "ve"),
            "qun": ("q", "vn"),
            "ran": ("r", "an"),
            "rang": ("r", "ang"),
            "rao": ("r", "ao"),
            "re": ("r", "e"),
            "ren": ("r", "en"),
            "reng": ("r", "eng"),
            "ri": ("r", "iii"),
            "rong": ("r", "ong"),
            "rou": ("r", "ou"),
            "ru": ("r", "u"),
            "rua": ("r", "ua"),
            "ruan": ("r", "uan"),
            "rui": ("r", "uei"),
            "run": ("r", "uen"),
            "ruo": ("r", "uo"),
            "sa": ("s", "a"),
            "sai": ("s", "ai"),
            "san": ("s", "an"),
            "sang": ("s", "ang"),
            "sao": ("s", "ao"),
            "se": ("s", "e"),
            "sen": ("s", "en"),
            "seng": ("s", "eng"),
            "sha": ("sh", "a"),
            "shai": ("sh", "ai"),
            "shan": ("sh", "an"),
            "shang": ("sh", "ang"),
            "shao": ("sh", "ao"),
            "she": ("sh", "e"),
            "shei": ("sh", "ei"),
            "shen": ("sh", "en"),
            "sheng": ("sh", "eng"),
            "shi": ("sh", "iii"),
            "shou": ("sh", "ou"),
            "shu": ("sh", "u"),
            "shua": ("sh", "ua"),
            "shuai": ("sh", "uai"),
            "shuan": ("sh", "uan"),
            "shuang": ("sh", "uang"),
            "shui": ("sh", "uei"),
            "shun": ("sh", "uen"),
            "shuo": ("sh", "uo"),
            "si": ("s", "ii"),
            "song": ("s", "ong"),
            "sou": ("s", "ou"),
            "su": ("s", "u"),
            "suan": ("s", "uan"),
            "sui": ("s", "uei"),
            "sun": ("s", "uen"),
            "suo": ("s", "uo"),
            "ta": ("t", "a"),
            "tai": ("t", "ai"),
            "tan": ("t", "an"),
            "tang": ("t", "ang"),
            "tao": ("t", "ao"),
            "te": ("t", "e"),
            "tei": ("t", "ei"),
            "teng": ("t", "eng"),
            "ti": ("t", "i"),
            "tian": ("t", "ian"),
            "tiao": ("t", "iao"),
            "tie": ("t", "ie"),
            "ting": ("t", "ing"),
            "tong": ("t", "ong"),
            "tou": ("t", "ou"),
            "tu": ("t", "u"),
            "tuan": ("t", "uan"),
            "tui": ("t", "uei"),
            "tun": ("t", "uen"),
            "tuo": ("t", "uo"),
            "wa": ("^", "ua"),
            "wai": ("^", "uai"),
            "wan": ("^", "uan"),
            "wang": ("^", "uang"),
            "wei": ("^", "uei"),
            "wen": ("^", "uen"),
            "weng": ("^", "ueng"),
            "wo": ("^", "uo"),
            "wu": ("^", "u"),
            "xi": ("x", "i"),
            "xia": ("x", "ia"),
            "xian": ("x", "ian"),
            "xiang": ("x", "iang"),
            "xiao": ("x", "iao"),
            "xie": ("x", "ie"),
            "xin": ("x", "in"),
            "xing": ("x", "ing"),
            "xiong": ("x", "iong"),
            "xiu": ("x", "iou"),
            "xu": ("x", "v"),
            "xuan": ("x", "van"),
            "xue": ("x", "ve"),
            "xun": ("x", "vn"),
            "ya": ("^", "ia"),
            "yan": ("^", "ian"),
            "yang": ("^", "iang"),
            "yao": ("^", "iao"),
            "ye": ("^", "ie"),
            "yi": ("^", "i"),
            "yin": ("^", "in"),
            "ying": ("^", "ing"),
            "yo": ("^", "iou"),
            "yong": ("^", "iong"),
            "you": ("^", "iou"),
            "yu": ("^", "v"),
            "yuan": ("^", "van"),
            "yue": ("^", "ve"),
            "yun": ("^", "vn"),
            "za": ("z", "a"),
            "zai": ("z", "ai"),
            "zan": ("z", "an"),
            "zang": ("z", "ang"),
            "zao": ("z", "ao"),
            "ze": ("z", "e"),
            "zei": ("z", "ei"),
            "zen": ("z", "en"),
            "zeng": ("z", "eng"),
            "zha": ("zh", "a"),
            "zhai": ("zh", "ai"),
            "zhan": ("zh", "an"),
            "zhang": ("zh", "ang"),
            "zhao": ("zh", "ao"),
            "zhe": ("zh", "e"),
            "zhei": ("zh", "ei"),
            "zhen": ("zh", "en"),
            "zheng": ("zh", "eng"),
            "zhi": ("zh", "iii"),
            "zhong": ("zh", "ong"),
            "zhou": ("zh", "ou"),
            "zhu": ("zh", "u"),
            "zhua": ("zh", "ua"),
            "zhuai": ("zh", "uai"),
            "zhuan": ("zh", "uan"),
            "zhuang": ("zh", "uang"),
            "zhui": ("zh", "uei"),
            "zhun": ("zh", "uen"),
            "zhuo": ("zh", "uo"),
            "zi": ("z", "i"),
            "zong": ("z", "ong"),
            "zou": ("z", "ou"),
            "zu": ("z", "u"),
            "zuan": ("z", "uan"),
            "zui": ("z", "uei"),
            "zun": ("z", "uen"),
            "zuo": ("z", "uo"),
            "ie": ("^", "ie"),
            "cher": ("ch", "er"),
        }
        _initials = [
            "^",
            "b",
            "c",
            "ch",
            "d",
            "f",
            "g",
            "h",
            "j",
            "k",
            "l",
            "m",
            "n",
            "p",
            "q",
            "r",
            "s",
            "sh",
            "t",
            "x",
            "z",
            "zh",
        ]

        _finals = [
            "a",
            "ai",
            "an",
            "ang",
            "ao",
            "e",
            "ei",
            "en",
            "eng",
            "er",
            "i",
            "ia",
            "ian",
            "iang",
            "iao",
            "ie",
            "ii",
            "iii",
            "in",
            "ing",
            "iong",
            "iou",
            "o",
            "ong",
            "ou",
            "u",
            "ua",
            "uai",
            "uan",
            "uang",
            "uei",
            "uen",
            "ueng",
            "uo",
            "v",
            "van",
            "ve",
            "vn",
        ]
        pinyin = pinyin.lower()

        if not re.match(r"^[a-züv]+[1-5]?$", pinyin):
            return "Invalid Pinyin input."
        if pinyin not in PINYIN_DICT.keys():
            return [pinyin]
        else:
            a = pinyin
            a1, a2 = PINYIN_DICT[a]
            # import pdb; pdb.set_trace()
        if a1 == "^":
            return [a2]
        if a2 == "^":
            return [a1]
        return [a1, a2]

    notes = midi_data.instruments[0].notes
    note_index = 0
    word_index = 0
    # for lyric in midi_data.lyrics:
    while word_index < len(midi_data.lyrics):
        lyric = midi_data.lyrics[word_index]
        assert lyric.time > partitions[current_partition][0]
        if lyric.time > partitions[current_partition][1]:
            text.append([])
            phonemes.append([])
            pitches.append([])
            note_durations.append([])
            phn_durations.append([])
            is_slur.append([])
            current_partition += 1
            assert lyric.time > partitions[current_partition][0]
            assert lyric.time < partitions[current_partition][1]

        if lyric.text == "?" or lyric.text == "-":
            phonemes[-1].append(phonemes[-1][-1])
            is_slur[-1].append(1)
            pitches[-1].append(pitches[-1][-1])
            note_durations[-1].append(note_durations[-1][-1])
            phn_durations[-1].append(phn_durations[-1][-1])
        else:
            if "#" in lyric.text:
                word, id = lyric.text.split("#")
                if id == "1":
                    text[-1].append(word)
                    phn_list = split_pinyin(word)
                    phonemes[-1].extend(phn_list)
                    is_slur[-1].extend([0] * len(phn_list))
            # Note(yueqian): The English logic is not correct yet
            # elif check_en(lyric.text):
            #     text[-1].append(lyric.text)
            #     phn_list = [lyric.text]
            #     phonemes[-1].append(lyric.text)
            #     is_slur[-1].append(0)
            else:
                text[-1].append(lyric.text)
                phn_list = split_pinyin(lyric.text)
                phonemes[-1].extend(phn_list)
                is_slur[-1].extend([0] * len(phn_list))
            current_word_pitch = []
            current_word_duration = []
            # get all the notes from note_index to the end of the next lyric time
            if word_index == len(midi_data.lyrics) - 1:
                while note_index < len(notes):
                    note = midi_data.instruments[0].notes[note_index]
                    current_word_pitch.append(note.pitch)
                    note_duration = note.end - note.start
                    current_word_duration.append(note_duration)
                    note_index += 1
            else:
                while (
                    note_index < len(notes)
                    and notes[note_index].start < midi_data.lyrics[word_index + 1].time
                ):
                    note = midi_data.instruments[0].notes[note_index]
                    current_word_pitch.append(note.pitch)
                    note_duration = note.end - note.start
                    current_word_duration.append(note_duration)
                    note_index += 1

            # if the len(phn_list) = len(current_word_pitch), then we can just assign the pitches to the phonemes
            if len(phn_list) == len(current_word_pitch):
                pitches[-1].extend(current_word_pitch)
                note_durations[-1].extend(current_word_duration)
                phn_durations[-1].extend(current_word_duration)
            # if the len(phn_list) < len(current_word_pitch), then we need to assign the rest of the pitches to the last phoneme
            elif len(phn_list) < len(current_word_pitch):
                pitches[-1].extend(current_word_pitch)
                note_durations[-1].extend(current_word_duration)
                phn_durations[-1].extend(current_word_duration)
                phonemes[-1].extend(
                    [phonemes[-1][-1]] * (len(current_word_pitch) - len(phn_list))
                )
                is_slur[-1].extend([1] * (len(current_word_pitch) - len(phn_list)))
            # if the len(phn_list) > len(current_word_pitch), then we need to assign the rest of the phonemes to the last pitch
            else:
                pitches[-1].extend(current_word_pitch)
                note_durations[-1].extend(current_word_duration)
                phn_durations[-1].extend(current_word_duration)
                pitches[-1].extend(
                    [current_word_pitch[-1]] * (len(phn_list) - len(current_word_pitch))
                )
                note_durations[-1].extend(
                    [current_word_duration[-1]]
                    * (len(phn_list) - len(current_word_pitch))
                )
                phn_durations[-1][-1] = current_word_duration[-1] / (
                    len(phn_list) - len(current_word_pitch) + 1
                )
                phn_durations[-1].extend(
                    [phn_durations[-1][-1]] * (len(phn_list) - len(current_word_pitch))
                )
        word_index += 1

    for i in range(len(phonemes)):
        if not (
            len(phonemes[i])
            == len(pitches[i])
            == len(note_durations[i])
            == len(is_slur[i])
            == len(phn_durations[i])
        ):
            # If the lengths are not equal, print the lengths for debugging
            print(f"Length mismatch in partition {i}:")
            print(f"  Phonemes: {len(phonemes[i])}, {phonemes}")
            print(f"  Pitches: {len(pitches[i])}, {pitches}")
            print(f"  Note Durations: {len(note_durations[i])}", note_durations)
            print(f"  Phoneme Durations: {len(phn_durations[i])}", phn_durations)
            print(f"  Is Slur: {len(is_slur[i])}", is_slur)
            raise ValueError("Length mismatch in partition.")

    return text, phonemes, pitches, note_durations, phn_durations, is_slur


def create_score(uid, phns, midis, syb_dur, keep):
    # Transfer into 'score' format
    assert len(phns) == len(midis)
    assert len(midis) == len(syb_dur)
    assert len(syb_dur) == len(keep)
    lyrics_seq = []
    midis_seq = []
    segs_seq = []
    phns_seq = []
    st = 0
    index_phn = 0
    note_list = []
    while index_phn < len(phns):
        midi = midis[index_phn]
        note_info = [st]
        st += syb_dur[index_phn]
        syb = [phns[index_phn]]
        index_phn += 1
        if (
            index_phn < len(phns)
            and syb_dur[index_phn] == syb_dur[index_phn - 1]
            and midis[index_phn] == midis[index_phn - 1]
            and keep[index_phn] == 0
        ):
            syb.append(phns[index_phn])
            index_phn += 1
        syb = "_".join(syb)
        note_info.extend([st, syb, midi, syb])
        note_list.append(note_info)
        # multi notes in one syllable
        while (
            index_phn < len(phns)
            and keep[index_phn] == 1
            and phns[index_phn] == phns[index_phn - 1]
        ):
            note_info = [st]
            st += syb_dur[index_phn]
            note_info.extend([st, "—", midis[index_phn], phns[index_phn]])
            note_list.append(note_info)
            index_phn += 1
    return note_list


def process_utterance(
    writer,
    wavscp,
    text,
    utt2spk,
    label,
    audio_dir,
    wav_dumpdir,
    segment,
    midi_mapping,
    tempos,
    tgt_sr=24000,
):
    uid, lyrics, phns, midis, syb_dur, phn_dur, keep = segment.strip().split("|")
    phns = phns.split(" ")
    midis = midis.split(" ")
    syb_dur = syb_dur.split(" ")
    phn_dur = phn_dur.split(" ")
    keep = keep.split(" ")

    # load tempo from midi
    id = uid[0:3]
    tempo = tempos[str(id)]
    song, piece, singer = uid.split("_")
    original_uid = uid
    uid = "_".join([singer, song, piece])
    # type convert
    phn_dur = [float(dur) for dur in phn_dur]
    syb_dur = [float(syb) for syb in syb_dur]
    keep = [int(k) for k in keep]
    note_list = create_score(uid, phns, midis, syb_dur, keep)

    text.write("{} {}\n".format(uid, " ".join(phns)))
    utt2spk.write("{} {}\n".format(uid, uid.split("_")[0]))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}/{}.wav".format(
        os.path.join(audio_dir, original_uid),
        tgt_sr,
        wav_dumpdir,
        uid,
    )
    os.system(cmd)

    wavscp.write("{} {}/{}.wav\n".format(uid, wav_dumpdir, uid))

    running_dur = 0
    assert len(phn_dur) == len(phns)
    label_entry = []
    for i in range(len(phns)):
        start = running_dur
        end = running_dur + phn_dur[i]
        label_entry.append("{:.3f} {:.3f} {}".format(start, end, phns[i]))
        running_dur += phn_dur[i]

    label.write("{} {}\n".format(uid, " ".join(label_entry)))
    score = dict(
        tempo=tempo, item_list=["st", "et", "lyric", "midi", "phns"], note=note_list
    )
    writer["{}".format(uid)] = score


def process_subset(args, set_name, tempos):
    makedir(os.path.join(args.tgt_dir, set_name))
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )
    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.tgt_dir, set_name, "score.scp")
    )

    midi_mapping = load_midi_note_scp(args.midi_note_scp)

    with open(
        os.path.join(args.src_data, "segments", set_name + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        segments = f.read().strip().split("\n")
        for segment in segments:
            process_utterance(
                writer,
                wavscp,
                text,
                utt2spk,
                label,
                args.src_data + "/segments/wav",
                args.wav_dumpdir,
                segment,
                midi_mapping,
                tempos,
                tgt_sr=args.sr,
            )


def segment_dataset(args):
    root_path = os.path.join(args.src_data, "segments")
    output_path = os.path.join(root_path, "wav")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    transcript_filepath = os.path.join(root_path, "transcriptions.txt")

    transcript_file = open(transcript_filepath, "w")
    for subdir in os.listdir(args.src_data):
        if subdir == "segments":
            continue
        full_subdir_path = os.path.join(args.src_data, subdir)

        if os.path.isdir(full_subdir_path) and "-unseg" not in subdir:
            print(f"Processing: {full_subdir_path}")
            wav_files = glob.glob(os.path.join(full_subdir_path, "*.wav"))
            midi_file = glob.glob(os.path.join(full_subdir_path, "*.mid"))[0]
            partitions = get_partitions(midi_file)
            songid = subdir
            # skip between 436 and 440
            if int(songid.split("-")[0]) <= 440 and int(songid.split("-")[0]) >= 436:
                continue
            for wav_file in wav_files:
                match = re.search(r"([0-9-]+)-([a-zA-Z-]+)\.wav", wav_file)
                assert songid == match.group(1)
                singer = match.group(2)
                segids, wav_scp = save_wav_segments_from_partitions(
                    output_path, wav_file, partitions, songid, singer
                )
                (
                    lyrics,
                    phonemes,
                    pitches,
                    note_durations,
                    phn_durations,
                    is_slur,
                ) = get_info_from_partitions(midi_file, partitions)
                for i, segid in enumerate(segids):
                    filename = wav_scp[i][1].split("/")[-1].split(".")[0]
                    lyrics_str = " ".join(lyrics[i])
                    phonemes_str = " ".join(phonemes[i])
                    pitches_str = " ".join(f"{pitch}" for pitch in pitches[i])
                    note_durations_str = " ".join(
                        f"{duration:.6f}" for duration in note_durations[i]
                    )
                    phn_durations_str = " ".join(
                        f"{duration:.6f}" for duration in phn_durations[i]
                    )
                    is_slur_str = " ".join(f"{slur}" for slur in is_slur[i])
                    transcript_line = f"{filename}|{lyrics_str}|{phonemes_str}|{pitches_str}|{note_durations_str}|{phn_durations_str}|{is_slur_str}"
                    transcript_file.write(transcript_line + "\n")
    transcript_file.close()
    train_transcript_filepath = os.path.join(root_path, "train.txt")
    test_transcript_filepath = os.path.join(root_path, "test.txt")
    # loop through the transcript file and split into train and test based on song id
    with open(transcript_filepath, "r") as f:
        transcript = f.readlines()
        train_transcript = []
        test_transcript = []
        for line in transcript:
            song_id = line.split("|")[0].split("_")[0].split("-")[0]
            if int(song_id) > 440:
                test_transcript.append(line)
            else:
                train_transcript.append(line)
    with open(train_transcript_filepath, "w") as f:
        f.writelines(train_transcript)
    with open(test_transcript_filepath, "w") as f:
        f.writelines(test_transcript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for KiSing Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--midi_note_scp",
        type=str,
        help="midi note scp for information of note id",
        default="local/midi-note.scp",
    )
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directory (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    parser.add_argument("--g2p", type=str, help="g2p", default="None")
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.src_data, "segments")):
        makedir(os.path.join(args.src_data, "segments"))
    segment_dataset(args)
    tempos = load_midi(args.src_data)
    for name in ["train", "test"]:
        process_subset(args, name, tempos)
