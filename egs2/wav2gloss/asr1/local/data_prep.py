import argparse
import shutil
import json
from collections import defaultdict
from pathlib import Path


LANGUAGES = (
    "adyg1241", "ainu1240", "apah1238", "arap1274", "arta1239",
    "balk1252", "beja1238", "bora1263", "dolg1241", "even1259",
    "goro1270", "jeju1234", "kaby1243", "kach1280", "kaka1265",
    "kama1378", "kara1499", "koii1238", "komn1238", "mand1415",
    "nngg1234", "nort2641", "pnar1238", "port1286", "ruul1235",
    "sanz1248", "savo1255", "selk1253", "slav1254", "sout2856",
    "sumb1241", "sumi1235", "taba1259", "taul1251", "tehr1242",
    "teop1238", "texi1237", "tond1251", "trin1278", "vera1241",
)
TASKS = (
    "transcription", "underlying", "gloss", "translation",
)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Process Raw data to Kaldi-like format"
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("downloads"),
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--langs",
        type=str,  # comma separated values
    )
    parser.add_argument(
        "--tasks",
        type=str,  # comma separated values
    )
    parser.add_argument(
        "--min_wav_length",
        type=float,
        default=0.5,
    )
    return parser


def build_dir(task, lang, split):
    target_dir = args.target_dir / f"w2g_{task}_{lang}_{split}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _cleaner(s):
    s = " ".join(s.split())
    return "".join([chr(ord(c)) for c in s if ord(c) != 160])


def write_dir(target_dir, metadata, min_wav_length):
    wavscp = open(target_dir / "wav.scp", "w", encoding="utf-8")
    text = open(target_dir / "text", "w", encoding="utf-8")
    utt2spk = open(target_dir / "utt2spk", "w", encoding="utf-8")
    lm = open(target_dir / "lm.txt", "w", encoding="utf-8")
    text_prev = open(target_dir / "text_prev", "w", encoding="utf-8")

    count = 0
    for fname, meta in metadata.items():
        if len(meta[task]) < 1:
            continue
        if meta["discard"]:
            continue
        if " " in fname:
            continue
        if meta["length"] < min_wav_length * 1000:
            continue

        header = f"<task|{task}> <lang|{lang}>"
        content = _cleaner(meta[task])

        _id = f"aaaaa_{lang}_{task}_{meta['id']}"
        _id = "".join(_id.split())

        wavscp.write(f"{_id} downloads/data/{lang}/audio/{split}/{fname}\n")
        text.write(f"{_id} {content}\n")
        utt2spk.write(f"{_id} aaaaa\n")
        lm.write(f"{count:010} {header} {content}\n")
        text_prev.write(f"{_id} {header}\n")

        count += 1

    wavscp.close()
    text.close()
    utt2spk.close()

    print(f"{target_dir}: {count} lines written.")


def merge_dir(target_dir, source_dirs):
    for fname in ("wav.scp", "text", "utt2spk", "lm.txt", "text_prev"):
        with open(target_dir / fname, "w", encoding="utf-8") as target:
            count = 0
            for d in source_dirs:
                if (d / fname).exists():
                    with open(d / fname, encoding="utf-8") as f:
                        for line in f:
                            if fname == "lm.txt":
                                target.write(f"{count:010}{line[10:]}")
                            else:
                                target.write(line)
                            count += 1
            if count == 0:
                break

    if count == 0:
        print(f"{target_dir}: No file to merge. Clean up & break.")
        shutil.rmtree(target_dir)
    else:
        print(f"{target_dir}: merged {source_dirs}")


def sanity_check(args):
    assert set(LANGUAGES) == {p.name for p in (args.source_dir / "data").glob("*/")}
    assert len(set(args.langs) - set(LANGUAGES)) == 0
    assert len(set(args.tasks) - set(TASKS)) == 0


def write_symbols(path):
    with open(path, "w", encoding="utf-8") as f:
        for lang in LANGUAGES:
            f.write(f"<lang|{lang}>\n")
        for task in TASKS:
            f.write(f"<task|{task}>\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args.langs = args.langs.split(",")
    args.tasks = args.tasks.split(",")
    sanity_check(args)

    args.target_dir.mkdir(parents=True, exist_ok=True)
    write_symbols(args.target_dir / "non_linguistic_symbols.txt")

    for lang in args.langs:
        for split in ("train", "dev", "test"):
            metafile = args.source_dir / "data" / lang / f"{split}.json"
            if metafile.exists():
                metadata = json.load(open(metafile, encoding="utf-8"))
                for task in args.tasks:
                    target_dir = build_dir(task, lang, split)
                    write_dir(target_dir, metadata, args.min_wav_length)

    for split in ("train", "dev"):
        for lang in args.langs:
            target_dir = build_dir("all", lang, split)
            merge_dir(
                target_dir,
                [args.target_dir / f"w2g_{task}_{lang}_{split}" for task in args.tasks],
            )
        for task in args.tasks:
            target_dir = build_dir(task, "full", split)
            merge_dir(
                target_dir,
                [args.target_dir / f"w2g_{task}_{lang}_{split}" for lang in args.langs],
            )
        target_dir = build_dir("all", "full", split)
        merge_dir(
            target_dir,
            [
                args.target_dir / f"w2g_{task}_{lang}_{split}"
                for lang in args.langs
                for task in args.tasks
            ],
        )

    print("pre-processing finished.")
