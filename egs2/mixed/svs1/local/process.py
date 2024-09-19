#!/usr/bin/env python3
import os
import re
import sys
import argparse

from espnet2.fileio.read_text import read_label
from espnet2.fileio.score_scp import SingingScoreReader, SingingScoreWriter
from ACE_phonemes.main import pinyin_to_phoneme
from espnet2.text.build_tokenizer import build_tokenizer

"""Process origin phoneme in music score using ACE-phoneme"""

jp_datasets = [
    "ameboshi",
    "itako",
    "kiritan",
    "oniku",
    "ofuton",
    "namine",
    "natsume",
]

zh_datasets = [
    "opencpop",
    "m4singer",
    "acesinger",
    "kising",
]

heads_list = ['b', 'z', 'l', 'sh', 'p', 'd',
              'm', 'x', 's', 'y', 'r', 'f', 
              'n', 'h', 'c', 'j', 'zh', 'ch',
              't', 'g', 'q', 'w', 'k']


def load_customed_dic(file):
    """If syllable-to-phone tranlation differs from g2p,"""
    """ customed tranlation can be added to customed_dic."""
    customed_dic = {}
    with open(file, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split(" ")
            customed_dic[key[0]] = key[1].split("_")
    return customed_dic


def check_language(dataset):
    if dataset in jp_datasets:
        return "jp"
    elif dataset in zh_datasets:
        return "zh"
    else:
        raise ValueError(f"{dataset} is not supported.")


def fix_phns(
    key, 
    lyric, 
    org_phns, 
    pro_phns, 
    index, 
    labels, 
    lang,
    phn_seg = {
        1: [1],
        2: [0.25, 1],
        3: [0.1, 0.5, 1],
        4: [0.05, 0.1, 0.5, 1],
    }
):
    new_labels = []
    if len(org_phns) == len(pro_phns):
        for i in range(len(org_phns)):
            if org_phns[i] != pro_phns[i]:
                print("Warning: Mismatch in syllable [{}]-> ace: {} and org: {}, {}-th phoneme {} vs {} in {}".format(
                        lyric, "_".join(pro_phns), "_".join(org_phns), i, pro_phns[i], org_phns[i], key
                ))
            new_labels.append([labels[index][0], labels[index][1], convert_phn_with_lang(pro_phns[i], lang)])
            index += 1
    else:
        print("Warning: Different length in syllable [{}]-> ace: {} and org: {} in {}".format(lyric, pro_phns, org_phns, key))
        st = float(labels[index][0])
        index += len(org_phns) 
        ed = float(labels[index - 1][1])
        tot_dur = (ed - st)
        phn_num = len(pro_phns)
        pre_seg = 0
        for i in range(len(pro_phns)):
            phn_ruled_dur = (phn_seg[phn_num][i] - pre_seg) * tot_dur
            pre_seg = phn_seg[phn_num][i]
            new_labels.append([st, st + phn_ruled_dur, convert_phn_with_lang(pro_phns[i], lang)])
            st += phn_ruled_dur
    return index, new_labels


def convert_phn_with_lang(phn, lang):
    return phn + "@" + lang


def convert(
    key, 
    score, 
    labels, 
    phn_seg = {
        1: [1],
        2: [0.25, 1],
        3: [0.1, 0.5, 1],
        4: [0.05, 0.1, 0.5, 1],
    },
    sli = ["AP", "SP"],
):
    aux_ace_dict = load_customed_dic(args.aux_ace_dict)
    aux_ace_tails_dict = load_customed_dic(args.aux_ace_tails_dict)
    dataset = key.split("_")[0]
    tokenizer = None
    lang = check_language(dataset)
    if lang == "jp":
        tokenizer = build_tokenizer(
            token_type="phn",
            bpemodel=None,
            delimiter=None,
            space_symbol="<space>",
            non_linguistic_symbols=None,
            g2p_type=args.g2p,
        )
    else:
        tokenizer = pinyin_to_phoneme

    index = 0
    new_labels = []

    # NOTE(Yuxun): pre_head_phn for Kising dataset
    # In Kising, you will encouter caese below for its various tones.
    # note[1]: lyrics: "zh"; note: 61
    # note[2]: lyrics: "i"; note: 65
    # Here using pre_head_phn to concat 'zh'(pre_head_phn) with 'i'(current phn) to get ace_phns
    pre_head_phn = None 

    for i in range(len(score)):
        lyric = score[i][2]

        if lyric in sli: # silence case
            new_labels.append([labels[index][0], labels[index][1], lyric])
            continue
        elif lyric == "—": # slur case
            phn = new_labels[-1][2]
            score[i][4] = phn
            new_labels.append([labels[index][0], labels[index][1], phn])
            index += 1
            continue

        org_phns = score[i][4].split("_")
        org_lyric = lyric
        if lang == "jp":
            pro_phns = tokenizer.g2p(lyric)
        else:
            new_phns = []
            for phn in org_phns:
                if phn in aux_ace_tails_dict:
                    new_phns.append(aux_ace_tails_dict[phn][0])
                else:
                    new_phns.append(phn)
            org_phns = new_phns
            lyric = "".join(org_phns)

            if pre_head_phn is None:
                if lyric in aux_ace_dict: 
                    lyric = aux_ace_dict[lyric][0]
            pro_phns = tokenizer(lyric)
        # print("{}, {}".format(lyric, pro_phns))

        set_phn_type = 0
        if pre_head_phn is not None:
            lyric = pre_head_phn + lyric
            if lyric in aux_ace_dict: 
                lyric = aux_ace_dict[lyric][0]
            pro_phns = tokenizer(lyric)
            set_phn_type = 2
            print("Warning: Concat {}+{}->{} in {}".format(pre_head_phn, org_lyric, lyric, key))
        elif pro_phns == "pinyin not found" and lyric in heads_list:
            if dataset == "kising":
                if lyric in heads_list:
                    pre_head_phn = lyric
                    set_phn_type = 1
                    print("Warning: set prevous head phoneme {} in {}".format(pre_head_phn, key))
                    continue
                
        if pro_phns == "pinyin not found":
            raise ValueError(f"lyric: \"{lyric}\" is not supported in ACE_phonemes and org_phns: {org_phns} in {key}")

        if set_phn_type == 0:
            score_phns = []
            for phn in pro_phns:
                score_phns.append(convert_phn_with_lang(phn, lang))
            score[i][4] = "_".join(score_phns)
            index, tmp_labels = fix_phns(key, lyric, org_phns, pro_phns, index, labels, lang, phn_seg)
            new_labels.extend(tmp_labels)
        elif set_phn_type == 2:
            assert len(pro_phns) == 2
            org_phns = score[i - 1][4].split("_")
            org_phns.extend(score[i][4].split("_"))
            score[i - 1][4] = convert_phn_with_lang(pro_phns[0], lang)
            score[i][4] = convert_phn_with_lang(pro_phns[1], lang)
            index, tmp_labels = fix_phns(key, lyric, org_phns, pro_phns, index, labels, lang, phn_seg)
            new_labels.extend(tmp_labels)
        if set_phn_type != 1:
            pre_head_phn = None
    return score, new_labels


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process origin phoneme in music score using ACE-phoneme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scp", type=str, help="data directory scp")
    parser.add_argument(
        "--score_dump", default="score_dump", type=str, help="score dump directory"
    )
    parser.add_argument(
        "--silence", action="append", help="silence_phone", default=["AP", "SP"]
    )
    parser.add_argument(
        "--aux_ace_dict",
        type=str,
        help="phoneme list for supplement on ACE_phoneme",
        default="local/aux_ace_dict.txt",
    )
    parser.add_argument(
        "--aux_ace_tails_dict",
        type=str,
        help="bias of tails list on ACE_phoneme",
        default="local/aux_ace_tails_dict.txt",
    )
    parser.add_argument("--g2p", type=str, help="g2p", default="pyopenjtalk")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    reader = SingingScoreReader(os.path.join(args.scp, "score.scp"))
    writer = SingingScoreWriter(
        args.score_dump,
        os.path.join(args.scp, "score.scp.tmp")
    )
    labels = read_label(os.path.join(args.scp, "label"))
    lablel_writer = open(os.path.join(args.scp, "label.tmp"), "w", encoding="utf-8")
    text_writer = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")
    utt2lang_writer = open(os.path.join(args.scp, "utt2lang.tmp"), "w", encoding="utf-8")
    for key in labels:
        score = reader[key]
        score["note"], new_labels = convert(key, score["note"], labels[key], sli=args.silence)

        dataset = key.split("_")[0]
        lang = check_language(dataset)
        utt2lang_writer.write(f"{key} {lang}\n")

        writer[key] = score
        lablel_writer.write(f"{key} ")
        text_writer.write(f"{key} ")
        for st, ed, phn in new_labels:
            lablel_writer.write(f"{st} {ed} {phn} ")
            text_writer.write(f"{phn} ")
        lablel_writer.write('\n')
        text_writer.write('\n')
