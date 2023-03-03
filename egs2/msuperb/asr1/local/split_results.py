import argparse
import glob
import json
import os
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
tree.build_from_json("downloads/linguistic.json")
with open(f"downloads/macro.json", "r", encoding="utf-8") as f:
    macros = json.load(f)
with open(f"downloads/exception.json", "r", encoding="utf-8") as f:
    exceptions = json.load(f)


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


def get_info_from_line(line):
    utt_id = line.split("\t")[-1]
    utt_id = utt_id[1 : len(utt_id) // 2]
    iso = utt_id.split("_")[-2]
    return iso, utt_id


def lid_parse(root, lines):
    new_lines = []
    lid_info = None
    # Extract LID info from WER results if LID exists
    if (LID or ONLY_LID) and "score_wer" in root:
        lid_info = []
        for line in lines:
            if line[0] == "\t":  # prediction is NULL...
                lid = "[UNK]"
            else:
                words = line.split("\t")[0].split(" ")
                lid = words[0]
            iso, utt_id = get_info_from_line(line)
            lid_info.append(f"{lid}\t{iso}\t{utt_id}")

    # Remove LID in Multilingual + LID case
    for line in lines:
        if LID and (not ONLY_LID):
            if line[0] == "\t":  # prediction is NULL...
                pass
            else:
                text = line.split("\t")[0]
                if "score_wer" in root:
                    if " " in text:
                        line = line.split(" ", 1)[1]
                    else:
                        line = line[len(text) :]
                elif "score_cer" in root:
                    chars = line.split(" ")
                    if "<space>" in chars:
                        idx = chars.index("<space>")
                        line = " ".join(chars[idx + 1 :])
                    else:
                        line = line[len(text) :]
        new_lines.append(line)
    return new_lines, lid_info


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
    lines, lid_info = lid_parse(root, lines)

    categorizer.set_category_func(lambda line: rule_fn(get_info_from_line(line)[0]))
    set2lines = categorizer.exec(lines)
    for k, v in set2lines.items():
        write_lines(v, f"{root}/{name}/{k}/{os.path.basename(trn_path)}")

    if lid_info is not None and "hyp.trn" in trn_path:
        categorizer.set_category_func(lambda line: rule_fn(line.split("\t")[1]))
        set2lid_results = categorizer.exec(lid_info)
        for k, v in set2lid_results.items():
            write_lines(v, f"{root}/{name}/{k}/lid.trn")


def main(args):
    roots = []
    if not ONLY_LID:
        for txt_paths in glob.glob(f"{args.dir}/*/*/score_cer/result.txt"):
            roots.append(os.path.dirname(txt_paths))
    for txt_paths in glob.glob(f"{args.dir}/*/*/score_wer/result.txt"):
        roots.append(os.path.dirname(txt_paths))
    for root in roots:
        print(f"Parsing results in {root}...")
        ref_trn_path = f"{root}/ref.trn"
        hyp_trn_path = f"{root}/hyp.trn"
        split_trn_by_rule(root, "independent", independent_rule, ref_trn_path)
        split_trn_by_rule(root, "independent", independent_rule, hyp_trn_path)
        split_trn_by_rule(root, "few_shot", few_shot_rule, ref_trn_path)
        split_trn_by_rule(root, "few_shot", few_shot_rule, hyp_trn_path)
        split_trn_by_rule(root, "language_family", language_family_rule, ref_trn_path)
        split_trn_by_rule(root, "language_family", language_family_rule, hyp_trn_path)
        split_trn_by_rule(root, "all", no_rule, ref_trn_path)
        split_trn_by_rule(root, "all", no_rule, hyp_trn_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--lid", type=str2bool, default=False)
    parser.add_argument("--only_lid", type=str2bool, default=False)

    args = parser.parse_args()
    LID = args.lid
    ONLY_LID = args.only_lid
    main(args)
