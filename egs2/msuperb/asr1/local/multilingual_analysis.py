import argparse
import os
import glob
import json

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


def line2result(line):
    res = {}
    fields = ["name", "snt", "wrd", "corr", "sub", "del", "ins", "err", "serr"]
    cur = 0
    for val in line.split():
        if val == "|":
            continue
        res[fields[cur]] = val
        cur += 1

    return res


def independent_score(iso_results):
    for iso, vs in iso_results.items():
        errors = [float(v["err"]) for v in vs]
        print(f"{iso}: {sum(errors) / len(errors):.2f}%")


def family_analysis(iso_results, linguistic_info):
    tree, macros, exceptions = linguistic_info
    errs = {}
    for iso, vs in iso_results.items():
        if iso in macros or iso in exceptions:
            continue
        try:
            node = tree.get_node("iso", iso)
        except:
            print(f"Inknown ISO code ({iso})...")
            continue
        family_name = node.get_ancestors()[1].name
        if family_name not in errs:
            errs[family_name] = []
        errs[family_name].extend([float(v["err"]) for v in vs])
    return errs


def few_shot_analysis(iso_results):
    errs = {}
    errs['reserved'] = []
    errs['trained'] = []
    errs['all'] = []
    for iso, vs in iso_results.items():
        if iso in RESERVE_LANG:
            errs['reserved'].extend([float(v["err"]) for v in vs])
        else:
            errs['trained'].extend([float(v["err"]) for v in vs])
        errs['all'].extend([float(v["err"]) for v in vs])
    return errs


def main(args):
    roots = []
    for txt_paths in glob.glob(f"{args.dir}/*/*/*/result.txt"):
        roots.append(os.path.dirname(txt_paths))
    for root in roots:
        output_path = f"{root}/multilingual_score.txt"
        results = []
        with open(f"{root}/result.txt", "r", encoding="utf-8") as f:
            anchored = False
            while 1:
                try:
                    if not anchored:
                        line = next(f)
                    else:
                        line = next(f)
                        if "Sum/Avg" in line:
                            break
                        line = next(f)
                        if "Sum/Avg" in line:
                            break
                        results.append(line2result(line.strip()))
                    if "Corr" in line and "Sub" in line and "Del" in line:
                        anchored = True
                except:
                    print("Oops! Should break earlier...")
                    break
        iso_results = {}
        for res in results:
            try:
                iso = res["name"].split("_")[-2]
            except:
                continue
            if iso not in iso_results:
                iso_results[iso] = []
            iso_results[iso].append(res)

        print(f"Write to {output_path}.")
        with open(output_path, "w", encoding="utf-8") as f:
            for iso, vs in iso_results.items():
                errors = [float(v["err"]) for v in vs]
                f.write(f"{iso}: {sum(errors) / len(errors):.2f}%\n")

        tree = LanguageTree()
        tree.build_from_json("downloads/linguistic.json")
        with open(f"downloads/macro.json", "r", encoding="utf-8") as f:
            macros = json.load(f)
        with open(f"downloads/exception.json", "r", encoding="utf-8") as f:
            exceptions = json.load(f)
        errs = family_analysis(iso_results, (tree, macros, exceptions))
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n\n")
            for k, v in errs.items():
                f.write(f"{k}: {sum(v) / len(v):.2f}%\n")

        errs = few_shot_analysis(iso_results)
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write("\n\n")
            for k, v in errs.items():
                f.write(f"{k}: {sum(v) / len(v):.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()
    main(args)
