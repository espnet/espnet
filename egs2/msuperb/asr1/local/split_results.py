import argparse
import glob
import json
import os

from linguistic_tree import LanguageTree

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


def read_lines(path):
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line == "\n":
                continue
            res.append(line.strip())
    return res


def write_lines(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def get_iso_from_line(line):
    utt_id = line.split("\t")[-1]
    utt_id = utt_id[1 : len(utt_id) // 2]
    iso = utt_id.split("_")[-2]
    return iso


def create_independent_trn(root, trn_path):
    iso2lines = {}
    lines = read_lines(trn_path)
    for line in lines:
        iso = get_iso_from_line(line)
        if iso not in iso2lines:
            iso2lines[iso] = []
        iso2lines[iso].append(line)

    for iso, lines in iso2lines.items():
        write_lines(lines, f"{root}/independent/{iso}/{os.path.basename(trn_path)}")


def create_few_shot_trn(root, trn_path):
    set2lines = {"trained": [], "reserved": []}
    lines = read_lines(trn_path)
    for line in lines:
        iso = get_iso_from_line(line)
        if iso not in RESERVE_LANG:
            set2lines["trained"].append(line)
        else:
            set2lines["reserved"].append(line)

    write_lines(
        set2lines["trained"], f"{root}/few_shot/trained/{os.path.basename(trn_path)}"
    )
    write_lines(
        set2lines["reserved"], f"{root}/few_shot/reserved/{os.path.basename(trn_path)}"
    )


def create_language_family_trn(root, trn_path):
    tree = LanguageTree()
    tree.build_from_json("downloads/linguistic.json")
    with open(f"downloads/macro.json", "r", encoding="utf-8") as f:
        macros = json.load(f)
    with open(f"downloads/exception.json", "r", encoding="utf-8") as f:
        exceptions = json.load(f)

    family2lines = {}
    lines = read_lines(trn_path)
    for line in lines:
        iso = get_iso_from_line(line)
        if iso in macros or iso in exceptions:
            continue
        try:
            node = tree.get_node("iso", iso)
        except Exception:
            print(f"Unknown ISO code ({iso})...")
            continue
        family_name = node.get_ancestors()[1].name
        family_name = family_name.replace(" ", "-")  # remove whitespace
        if family_name not in family2lines:
            family2lines[family_name] = []
        family2lines[family_name].append(line)

    for family_name, lines in family2lines.items():
        write_lines(
            lines, f"{root}/language_family/{family_name}/{os.path.basename(trn_path)}"
        )


def main(args):
    roots = []
    for txt_paths in glob.glob(f"{args.dir}/*/*/*/result.txt"):
        roots.append(os.path.dirname(txt_paths))
    for root in roots:
        print(f"Parsing results in {root}...")
        ref_trn_path = f"{root}/ref.trn"
        hyp_trn_path = f"{root}/hyp.trn"
        create_independent_trn(root, ref_trn_path)
        create_independent_trn(root, hyp_trn_path)
        create_few_shot_trn(root, ref_trn_path)
        create_few_shot_trn(root, hyp_trn_path)
        create_language_family_trn(root, ref_trn_path)
        create_language_family_trn(root, hyp_trn_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()
    main(args)
