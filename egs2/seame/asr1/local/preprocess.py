#!/usr/bin/env python3
# -*- encoding: utf8 -*-

"""
    This is an python implementation of preprocessing of
    the SEAME Mandarin-English code-switching corpus.
    We follow original papers [1, 2] and the official
    github repository [3] to make this code produces the
    same amount of training and testing data.

    [1] Dau-Cheng Lyu, Tien-Ping Tan, Eng-Siong Chng, and
        Haizhou Li, "SEAME: a Mandarin-English Code-switching
        Speech Corpus in South-East Asia," in Interspeech, 2010.
    [2] Zhiping Zeng, Yerbolat Khassanov, Van Tung Pham, Haihua
        Xu, Eng Siong Chng, and Haizhou Li, "On the End-to-End
        Solution to Mandarin-English Code-switching Speech
        Recognition," in Interspeech, 2019.
    [3] https://github.com/zengzp0912/SEAME-dev-set
"""

import argparse
import collections
import itertools
import os
import random as rd
import re
import sys

rd.seed(531)

remove_punc = '()[]{}.,?·@，。、「」＃"~-—#%_`｀×*（）［］&【】～ｌ\\'
pattern = str.maketrans(remove_punc, " " * len(remove_punc))

translate_char_source = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺé"
translate_char_target = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyze"
pattern2 = str.maketrans(translate_char_source, translate_char_target)

all_chars = (chr(i) for i in range(sys.maxunicode))
categories = {"Cc"}
control_chars = "".join(map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0))))
control_char_re = re.compile("[%s]" % re.escape(control_chars))


def remove_control_chars(text):
    """remove unprintable characters"""
    return control_char_re.sub("", text)


def remove_redundant_whitespaces(text):
    """remove redundant whitespaces"""
    return re.sub(" +", " ", text).strip()


def is_english(c):
    """check character is in English"""
    return ord(c.lower()) >= ord("a") and ord(c.lower()) <= ord("z")


def is_mandarin(c):
    """check character is Mandarin"""
    return (
        not is_english(c)
        and not c.isdigit()
        and c != " "
        and c != "<"
        and c != ">"
        and c != "'"
    )


def extract_mandarin_only(text):
    """remove other symbols except for Mandarin characters in a string"""
    return "".join([c for c in text if is_mandarin(c)])


def extract_non_mandarin(text):
    """remove Mandarin characters in a string"""
    return " ".join([w for w in text.split(" ") if not any(is_mandarin(c) for c in w)])


def insert_space_between_mandarin(text):
    """insert space between Mandarin characters"""

    if len(text) <= 1:
        return text
    out_text = text[0]
    for i in range(1, len(text)):
        if is_mandarin(text[i]):
            out_text += " "
        out_text += text[i]
        if is_mandarin(text[i]):
            out_text += " "

    return out_text


def remove_repeated_noise(text, pattern="<noise>"):
    """remove repeated <noise>"""

    if len(re.findall(pattern, text)) <= 1:
        return text

    out_text = ""
    text_split = text.split()
    out_text = [text_split[0]]
    for i in range(1, len(text_split)):
        if text_split[i] == pattern and text_split[i - 1] == pattern:
            continue
        else:
            out_text.append(text_split[i])

    return " ".join(out_text)


def normalize_text(text):
    """normalize a text sequence"""

    rmtext = re.sub(r"\(((pp)(\w)+)\)", "<noise>", text.lower(),)
    rmtext = re.sub(r"\<((pp)(\w)+)\>", "<noise>", rmtext,)
    rmtext = rmtext.translate(pattern)
    rmtext = remove_control_chars(rmtext)
    output_text = ""
    for wrd in rmtext.split():
        if wrd in {
            "ppl",
            "ppc",
            "ppb",
            "ppo",
            "<v-noise>",
        }:
            wrd = "<noise>"
        output_text += f"{wrd} "

    output_text = output_text.strip()
    output_text = output_text.translate(pattern2)
    output_text = output_text.replace("<unl>", "<unk>")
    output_text = output_text.replace("< unk >", "<unk>")
    output_text = re.sub(r"\<((unk)[a-z ]+)\>", "<unk>", output_text)
    output_text = insert_space_between_mandarin(output_text)
    output_text = remove_redundant_whitespaces(output_text)
    output_text = remove_repeated_noise(output_text, "<noise>")

    return output_text


def read_list(pth):
    """read data list (data/SEAME-dev-set/train/wav_file.txt)"""

    stypes, idxs = [], []
    with open(pth, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            stype, idx = line.split("/")[-3], line.split("/")[-2]
            stypes.append(stype)
            idxs.append(idx)
        return stypes, idxs


def read_text(pth, rmspk=False):
    """read dev set text data (data/SEAME-dev-set/{devset}/text)"""

    idxs = []
    with open(pth, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            line = line.split()[0]
            if rmspk:
                line = line.split("-", 1)[-1]
            idxs.append(line.lower())
        return idxs


def read_trans(data_dict, pth, phs, audio_list, aduio_pth):
    """read transcriptions (SEAME/{type}/transcript/phaseII/??.txt)"""

    audio_dict = set(audio_list)

    with open(pth, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if phs.lower() == "phasei":
                lang = None
                if len(line.split("\t")) == 4:
                    idx, start, end, text = line.split("\t")
                else:
                    idx, cont = line.split("\t", 1)
                    print(f"Skip {idx} with {cont}... (no transcript)")
                    continue
            elif phs.lower() == "phaseii":
                idx, start, end, lang, text = line.split("\t")
            else:
                print("folder error! not PhaseI or PhaseII")
                raise
            # start: start time in msec
            # end: end time in msec

            start_ms = start
            end_ms = end

            # fit the devset format
            s_len, e_len = len(start), len(end)
            if s_len < 5:
                start = int(round(fit_format(start) / 10, 0))
                start = str(start).zfill(5)
            else:
                start = int(round(float(start) / 10, 0))
            if e_len < 5:
                end = int(round(fit_format(end) / 10, 0))
                end = str(end).zfill(5)
            else:
                end = int(round(float(end) / 10, 0))

            name = f"{idx}-{start}-{end}"
            if name not in data_dict:
                if idx.split("_")[0][0].isdigit():
                    spkr = idx.split("_")[0][2:-2].lower()
                else:
                    spkr = idx.split("_")[0][:5].lower()

                if idx.split("-")[0] in audio_dict:
                    apth = os.path.join(audio_pth, name.split("-")[0] + ".flac")
                else:
                    print("FLAC idx error!")
                    raise

                data_dict[name.lower()] = {
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker": spkr,
                    "split": "train",
                    "audio_pth": apth,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "phase": phs,
                }
            else:
                print("Repeated idx!")
                raise


def fit_format(digit):
    """fit file name format"""
    str_digit = str(float(digit) / 10.0)
    if int(str_digit[-1]) >= 5:
        return float(digit) + 1
    else:
        return float(digit)


def check_audio(data_dict, audio_dict):
    """check whether data_dict and audio_dict match"""
    for key in data_dict.keys():
        if key.split("-")[0] not in audio_dict:
            print(f"key = {key} not in audio files")


def check_test_split(test, data_dict, splitname):
    """find testing data in data_dict"""

    train_idx = []
    data = list(data_dict.keys())
    count = 0
    space = {}
    idx_space = {}
    for key in data:
        idx, start, end = key.split("-")
        idx_space[idx] = idx_space.get(idx, []) + [[str(start), str(end)]]
        space[idx] = space.get(idx, []) + [[float(start), float(end)]]

    for key in test:
        idx, start, end = key.split("-")
        start, end = float(start), float(end)
        for list_idx, time in enumerate(space[idx]):
            if abs(start - time[0]) < 3 and abs(end - time[1]) < 3:
                count += 1
                time1, time2 = idx_space.get(idx)[list_idx]
                data_dict[(f"{idx}-{time1}-{time2}")]["split"] = splitname
                break

    print(f"=> Test set = {count}/{len(test)}")


def sieve_train(data_dict, train_dict):
    """tag samples other than training or testing data"""

    for key in data_dict.keys():
        if data_dict[key]["split"] == "train" and key.split("-")[0] in train_dict:
            continue
        elif data_dict[key]["split"] in ["devman", "devsge"]:
            continue
        else:
            data_dict[key]["split"] = "other"


def split_val(data_dict, num_val=None):
    """split train/val sets"""

    count = 0
    test_list = []
    tr_list = []
    for key, content in data_dict.items():
        if content["split"] in {"devman", "devsge"}:
            test_list.append(key)
        elif content["split"] == "train":
            tr_list.append(key)

    rd.shuffle(tr_list)
    val_len = num_val if num_val else int(len(tr_list) * 0.05)
    tr_list, val_list = tr_list[:-val_len], tr_list[-val_len:]

    for key in val_list:
        data_dict[key]["split"] = "valid"

    return data_dict, tr_list, val_list, test_list


def count_data(data_dict):
    """count audio length and number of speakers"""

    lens = {"train": 0.0, "valid": 0.0, "devman": 0.0, "devsge": 0.0, "other": 0.0}
    spkr_dict = {
        "train": set(),
        "valid": set(),
        "devman": set(),
        "devsge": set(),
        "other": set(),
    }
    for key, val in data_dict.items():
        lens[val["split"]] += (float(val["end_ms"]) - float(val["start_ms"])) / 1000.0
        spkr_dict[val["split"]].add(val["speaker"])

    for key in lens.keys():
        print(
            "=> {} set : {:.2f} hours / {} speakers".format(
                key, lens[key] / 3600.0, len(spkr_dict[key])
            )
        )


def write_f(pth, filename, data_dict):
    """write kaldi-compatible files"""

    print(f"=> Writing {filename}...")
    idx_pth = os.path.join(pth, "list")
    txt_pth = os.path.join(pth, "text.ori")
    rmtxt_pth = os.path.join(pth, "text.rm")
    idxtxt_pth = os.path.join(pth, "text.clean")
    idxnoisetxt_pth = os.path.join(pth, "text.rm.noise")
    seg_pth = os.path.join(pth, "segments")
    wav_pth = os.path.join(pth, "wav.scp")
    spk_pth = os.path.join(pth, "utt2spk")
    gender_pth = os.path.join(pth, "spk2gender")
    wav_cmds = {}
    gender = {}
    total_len = 0.0
    total_utt = 0

    # write idx list
    with open(txt_pth, "w") as tlist:
        with open(rmtxt_pth, "w") as rtlist:
            with open(idxtxt_pth, "w") as itlist:
                with open(idxnoisetxt_pth, "w") as intlist:
                    with open(seg_pth, "w") as slist:
                        with open(idx_pth, "w") as flist:
                            with open(wav_pth, "w") as wlist:
                                with open(spk_pth, "w") as spklist:
                                    with open(gender_pth, "w") as genlist:
                                        for idx, content in data_dict.items():
                                            if filename != content["split"]:
                                                continue

                                            # id & text
                                            text = content["text"]
                                            audio_pth = content["audio_pth"]
                                            spkr = content["speaker"]

                                            # process text
                                            normalized_text = normalize_text(text)
                                            no_noise_text = normalized_text.replace(
                                                "<noise>", ""
                                            ).replace("<unk>", "")
                                            no_noise_text = remove_redundant_whitespaces(
                                                no_noise_text
                                            )
                                            normalized_text = normalized_text.replace(
                                                "<unk>", "<UNK>"
                                            )

                                            # remove short utterances
                                            if len(no_noise_text) == 0:
                                                continue

                                            # fit kaldi format
                                            prefix, id_start, id_end = idx.split("-")

                                            # remove some short utterance
                                            if float(id_end) - float(id_start) <= 1:
                                                continue
                                            idx = (
                                                prefix
                                                + "-"
                                                + "0" * (6 - len(id_start))
                                                + id_start
                                                + "-"
                                                + "0" * (6 - len(id_end))
                                                + id_end
                                            )

                                            uttidx = f"{spkr}-{idx}"
                                            if spkr[-1] in ["m", "f"]:
                                                gender[spkr] = spkr[-1]
                                            else:
                                                # some SEAME's bug
                                                for g in reversed(prefix.split("_")[0]):
                                                    if g.lower() in ["m", "f"]:
                                                        gender[spkr] = g.lower()

                                            spklist.write(f"{uttidx} {spkr}\n")
                                            flist.write(f"{uttidx}\n")
                                            tlist.write(f"{uttidx} {text}\n")
                                            _, recordid, start, end = uttidx.split("-")
                                            wav_cmds[
                                                recordid
                                            ] = f"flac -c -d -s {audio_pth} |"
                                            # map to sec, original ms,
                                            # idx here has /10.
                                            start, end = (
                                                float(start) / 100,
                                                float(end) / 100,
                                            )
                                            # write segments
                                            slist.write(
                                                f"{uttidx} {recordid} {start} {end}\n"
                                            )

                                            rtlist.write(normalized_text + "\n")
                                            intlist.write(
                                                f"{uttidx} {normalized_text}\n"
                                            )

                                            itlist.write(f"{uttidx} {no_noise_text}\n")

                                            total_len += end - start
                                            total_utt += 1

                                        for recordid in sorted(wav_cmds.keys()):
                                            wlist.write(
                                                f"{recordid} {wav_cmds[recordid]}\n"
                                            )

                                        for spkr in sorted(gender.keys()):
                                            genlist.write(f"{spkr} {gender[spkr]}\n")

    print(
        "=>    {}: {} utts, {:.2f} hours, avg {:.2f} sec/utt".format(
            filename, total_utt, total_len / 3600.0, total_len / total_utt
        )
    )


def write_mandarin_only_text(data_dict, file, char_file1, char_file2):
    """write Mandarin text data"""

    counter = collections.Counter()
    with open(file, "w") as fp:
        for idx, content in data_dict.items():
            if "train" == content["split"]:
                text = normalize_text(content["text"])
                text = text.replace("<noise>", "")
                text = text.replace("<unk>", "")
                text = remove_redundant_whitespaces(text)
                text = extract_mandarin_only(text)
                counter.update(text)
                if text != "":
                    fp.write(text + "\n")

    vocab_list = sorted(counter.keys())
    print(f"=> Mandarin vocab size = {len(vocab_list)}")

    with open(char_file1, "w") as fp:
        fp.write("\n".join(vocab_list))
    with open(char_file2, "w") as fp:
        fp.write('bpe_nlsyms="<noise>,▁' + ",▁".join(vocab_list) + '"\n')
        fp.write(f"man_chars={len(vocab_list)}")


def write_bpe_train_text(data_dict, file):
    """write English BPE training text data"""

    with open(file, "w") as fp:
        for idx, content in data_dict.items():
            if "train" == content["split"]:
                text = normalize_text(content["text"])
                text = text.replace("<noise>", "")
                text = text.replace("<unk>", "")
                text = remove_redundant_whitespaces(text)
                text = extract_non_mandarin(text)
                if text != "":
                    fp.write(text + "\n")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", "-o", type=str, help="Path to output directory.",
    )
    parser.add_argument("--data", "-d", type=str, help="Path to original corpus.")
    parser.add_argument(
        "--repo", "-r", type=str, help="Path to official repo (downloaded)."
    )
    args = parser.parse_args()

    # basic variables setup
    out_pth = args.out
    ori_data_pth = args.data

    # read data
    print("=> Preprocessing transcription files...")
    audio_type = ["conversation", "interview"]
    audios, trans = [], []
    data_dict, audio_idx_list = {}, []
    for atp in audio_type:
        # read audio
        audio_pth = os.path.abspath(os.path.join(ori_data_pth, atp, "audio"))
        for au in os.listdir(os.path.join(ori_data_pth, atp, "audio")):
            audios.append(au.strip(".flac"))
            audio_idx_list.append(au.split("/")[-1].strip(".flac").lower())

        # read transcription
        for phs in ["phaseII"]:
            for txt in os.listdir(os.path.join(ori_data_pth, atp, "transcript", phs)):
                trans_pth = os.path.join(ori_data_pth, atp, "transcript", phs, txt)
                read_trans(data_dict, trans_pth, phs, audios, audio_pth)

    # check whether the audio file exists for each utterance
    print("=> Checking audio files...")
    check_audio(data_dict, set(audio_idx_list))

    # get train set
    print("=> Reading wav_file.txt for training set...")
    all_audio_pth = os.path.join(args.repo, "train", "wav_file.txt")
    folder_type, all_audio_idx = read_list(all_audio_pth)

    print("=> Getting training set...")
    sieve_train(data_dict, set(all_audio_idx))

    # dev set
    print("=> Reading dev set indices...")
    rmspk = True
    dev_man = os.path.join(args.repo, "dev_man", "text")
    devman_idx = read_text(dev_man, rmspk)

    dev_sge = os.path.join(args.repo, "dev_sge", "text")
    devsge_idx = read_text(dev_sge, rmspk)

    # check
    print("=> Checking testing sets...")
    check_test_split(devman_idx, data_dict, "devman")
    check_test_split(devsge_idx, data_dict, "devsge")

    # split
    print("=> Splitting train/val sets...")
    data_dict, tr_list, val_list, test_list = split_val(data_dict)

    # report some results
    print(f"=> Audio files = {len(audios)}")
    print(f"=> Total utterance = {len(data_dict.keys())}")
    print(f"=> Number of train set = {len(tr_list)}; validation set = {len(val_list)}")
    print(f"=> Number of devman = {len(devman_idx)}; devsge = {len(devsge_idx)}")

    # report corpus size (in hours)
    count_data(data_dict)

    # sort by speaker
    print("=> Sorting data by speaker id...")
    data_idx = []
    spkr_dict = collections.OrderedDict([])
    for k, v in data_dict.items():
        speaker = data_dict[k]["speaker"]
        spkr_dict[speaker] = spkr_dict.get(speaker, []) + [k]
    for k in sorted(spkr_dict.keys()):
        data_idx += sorted(spkr_dict[k])

    sorted_idx = []
    prev_name = None
    buff = {}
    for idx in data_idx:
        name, start = idx.split("-")[0], idx.split("-")[1]
        if prev_name:
            if prev_name == name:
                buff[int(start)] = idx
            else:
                sorted_idx += [buff[k] for k in sorted(buff.keys())]
                # clean buff
                buff = {int(start): idx}
                prev_name = name
        else:
            prev_name = name
            buff = {int(start): idx}
    sorted_data_dict = collections.OrderedDict()
    for key in sorted_idx:
        sorted_data_dict[key] = data_dict[key]

    # make kaldi format files
    print("=> Writing files...")
    for name in ["train", "valid", "devman", "devsge"]:
        data_pth = os.path.join(out_pth, name)
        os.makedirs(data_pth, exist_ok=True)
        write_f(data_pth, name, sorted_data_dict)

    write_mandarin_only_text(
        sorted_data_dict,
        os.path.join(out_pth, "train", "text.man"),
        os.path.join(out_pth, "train", "token.man.1"),
        os.path.join(out_pth, "train", "token.man.2"),
    )

    write_bpe_train_text(
        sorted_data_dict, os.path.join(out_pth, "train", "text.eng.bpe"),
    )
