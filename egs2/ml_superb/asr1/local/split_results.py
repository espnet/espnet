import argparse
import glob
import json
import os
import re
from typing import Callable, List

from linguistic_tree import LanguageTree

from espnet2.utils.types import str2bool

""" Global Settings """
RESERVE_LANG = [
    "dan",
    "lit",
    "tur",
    "srp",
    "vie",
    "kaz",
    "zul",
    "tsn",
    "epo",
    "frr",
    "tok",
    "umb",
    "bos",
    "ful",
    "ceb",
    "luo",
    "kea",
    "sun",
    "tso",
    "tos",
]

LID = False
ONLY_LID = False


""" Global Objects """


class Categorizer(object):
    """Template for categorizing."""

    def __init__(self) -> None:
        self.category_fn = None

    def set_category_func(self, fn: Callable):
        self.category_fn = fn

    def exec(self, data: List):
        if self.category_fn is None:
            raise NotImplementedError
        res = {}
        for d in data:
            k = self.category_fn(d)
            if k is None:
                continue
            if k not in res:
                res[k] = []
            res[k].append(d)
        return res


categorizer = Categorizer()
tree = LanguageTree()
if os.path.exists("downloads/linguistic.json"):
    tree.build_from_json("downloads/linguistic.json")
    with open(f"downloads/macro.json", "r", encoding="utf-8") as f:
        macros = json.load(f)
    with open(f"downloads/exception.json", "r", encoding="utf-8") as f:
        exceptions = json.load(f)
else:
    print("[warning] linguistic information not loading")


def read_lines(path):
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line == "\n":
                continue
            res.append(line.rstrip())
    return res


def write_lines(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def get_info_from_trn_line(line):
    utt_id = line.split("\t")[-1]
    utt_id = utt_id[1 : len(utt_id) // 2]
    iso = utt_id.split("_")[-2]
    return iso, utt_id


def get_info_from_raw_line(line):
    segs = line.split(" ", 1)
    if len(segs) == 1:  # null prediction
        utt_id, text = segs[0], ""
    else:
        [utt_id, text] = line.split(" ", 1)
    isos = re.findall(r"\[[a-z]{3}\]", text)
    isos.extend(
        re.findall(r"\[[a-z]{3}_[a-z]{3}\]", text)
    )  # There is a LID call [org_jpn]
    return isos, utt_id


def lid_parse(ref_path, hyp_path):
    def read_lid(path):
        res = {}
        for line in read_lines(path):
            try:
                isos, utt_id = get_info_from_raw_line(line)
            except Exception:
                print(line)
                raise
            res[utt_id] = isos
        return res

    lid_info = []
    ref_lid_info = read_lid(ref_path)
    hyp_lid_info = read_lid(hyp_path)
    for utt_id, hyp_isos in hyp_lid_info.items():
        assert (
            utt_id in ref_lid_info
        ), f"Can not find groundtruth of utterance ({utt_id}) in {ref_path}."
        assert (
            len(ref_lid_info[utt_id]) == 1
        ), f"Utternace ({utt_id}) should have exactly one LID in {ref_path}."
        if len(hyp_isos) == 0:
            hyp_isos.append("[UNK]")
        ref_iso = ref_lid_info[utt_id][0]
        hyp_iso = hyp_isos[0]
        lid_info.append(f"{ref_iso}\t{hyp_iso}\t({utt_id}-{utt_id})")

    return lid_info


def no_rule(iso):
    return "all"


def independent_rule(iso):
    return iso


def few_shot_rule(iso):
    if iso not in RESERVE_LANG:
        return "trained"
    return "reserved"


def language_family_rule(iso):
    if iso in macros or iso in exceptions:
        return None
    try:
        node = tree.get_node("iso", iso)
        family_name = node.get_ancestors()[1].name
        family_name = family_name.replace(" ", "-")  # remove whitespace
        return family_name
    except Exception:
        raise ValueError(f"Unknown ISO code ({iso})...")


def split_trn_by_rule(root, name, rule_fn, trn_path):
    lines = read_lines(trn_path)

    categorizer.set_category_func(lambda line: rule_fn(get_info_from_trn_line(line)[0]))
    set2lines = categorizer.exec(lines)
    for k, v in set2lines.items():
        write_lines(v, f"{root}/{name}/{k}/{os.path.basename(trn_path)}")


def main(args):
    roots = []
    if not ONLY_LID:  # TER will be parsed from trn and using sclite as the usual case
        for txt_path in glob.glob(f"{args.dir}/*/*/score_cer/result.txt"):
            roots.append(os.path.dirname(txt_path))
        for txt_path in glob.glob(f"{args.dir}/*/*/score_wer/result.txt"):
            roots.append(os.path.dirname(txt_path))
        for root in roots:
            print(f"Parsing TER results in {root}...")
            ref_trn_path = f"{root}/ref.trn"
            hyp_trn_path = f"{root}/hyp.trn"

            if args.score_type == "independent":
                split_trn_by_rule(root, "independent", independent_rule, ref_trn_path)
                split_trn_by_rule(root, "independent", independent_rule, hyp_trn_path)
            elif args.score_type == "normal":
                split_trn_by_rule(root, "few_shot", few_shot_rule, ref_trn_path)
                split_trn_by_rule(root, "few_shot", few_shot_rule, hyp_trn_path)
            elif args.score_type == "language_family":
                split_trn_by_rule(
                    root, "language_family", language_family_rule, ref_trn_path
                )
                split_trn_by_rule(
                    root, "language_family", language_family_rule, hyp_trn_path
                )
            elif args.score_type == "all":
                split_trn_by_rule(root, "all", no_rule, ref_trn_path)
                split_trn_by_rule(root, "all", no_rule, hyp_trn_path)

    if LID or ONLY_LID:  # LID will be parsed from inferenced text file directly
        tasks = []
        for hyp_txt_path in glob.glob(f"{args.dir}/*/*/text"):
            root = os.path.dirname(hyp_txt_path)
            data_dirname = os.path.basename(root)
            ref_txt_path = f"data/{data_dirname}/text"
            tasks.append((root, ref_txt_path, hyp_txt_path))
        for root, ref_txt_path, hyp_txt_path in tasks:
            print(f"Parsing LID results in {root}...")
            root = f"{root}/score_lid"
            lid_trn_path = f"{root}/lid.trn"

            lid_info = lid_parse(ref_txt_path, hyp_txt_path)
            write_lines(lid_info, lid_trn_path)

            if args.score_type == "independent":
                split_trn_by_rule(root, "independent", independent_rule, lid_trn_path)
            elif args.score_type == "normal":
                split_trn_by_rule(root, "few_shot", few_shot_rule, lid_trn_path)
            elif args.score_type == "language_family":
                split_trn_by_rule(
                    root, "language_family", language_family_rule, lid_trn_path
                )
            elif args.score_type == "all":
                split_trn_by_rule(root, "all", no_rule, lid_trn_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--lid", type=str2bool, default=False)
    parser.add_argument("--only_lid", type=str2bool, default=False)
    parser.add_argument("--score_type", type=str, default="all")

    args = parser.parse_args()
    LID = args.lid
    ONLY_LID = args.only_lid

    main(args)
