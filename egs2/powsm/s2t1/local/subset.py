import os

import kaldiio
from tqdm import tqdm

"""
Preparation
1. Make a directory to store the train/dev/test dump directory
    e.g., ipapack_plus/s2t1/dump/raw
    - in this case, "newdir" is "dump/raw"
    - and "olddir" is the path to the original dump directory
2. Run this script in newdir's parent directory to get correct path for wav.scp files
3. `genwav` takes 20 min for fleco -- might take 4+ hours for reduced ipapack_plus

Functions
- subsample: subsample the dataset by 1/ratio
- filter: filter the dataset by a keyword
- rename: handle dataset renaming for wav.scp and data/
- genwav: generate new data_wav.ark and wav.scp for each split
- combine: combine files from a list of datasets
"""


def subsample(olddir, dataset, newdir, newdataset, ratio):
    print(f"-> Subsample {dataset} by 1/{ratio}...")
    currentdir = f"{newdir}/{newdataset}"
    os.system(f"mkdir -p {currentdir}/texts")
    tasks = ["pr", "asr", "g2p", "p2g"]

    # 1. direct copy: feats_type & audio_format
    os.system(f"cp {olddir}/{dataset}/feats_type {currentdir}/feats_type")
    os.system(f"echo flac.ark > {currentdir}/audio_format")

    # 2. subsampling
    for file in ["utt2spk", "wav.scp", "utt2num_samples"]:
        os.system(
            f"awk 'NR % {ratio} == 0' {olddir}/{dataset}/{file} > {currentdir}/{file}"
        )
        # add task name to the original utterance ID to get unique IDs for wav.scp
        for task in tasks:
            # take the first column which is the utterance ID
            os.system(
                (
                    f'awk \'{{ $1 = $1 "_{task}"; print }}\' OFS=" " '
                    f"{currentdir}/{file} >> {currentdir}/{file}.tmp"
                )
            )
        os.system(f"mv {currentdir}/{file}.tmp {currentdir}/{file}")

    # spk2utt should be handled separately
    #   map same speaker ("aaaaa") to all utterances on one line
    file = "spk2utt"
    with open(f"{olddir}/{dataset}/{file}", "r") as spk2utt, open(
        f"{currentdir}/{file}.tmp", "w"
    ) as spk2utt_tmp:
        lines = spk2utt.readlines()
        assert len(lines) == 1
        lines = lines[0].split(" ")
        spk2utt_tmp.write(lines[0])  # speaker
        spk2utt_tmp.write(" ")
        spk2utt_tmp.write(" ".join(lines[1:]))  # utterances
    os.system(f"mv {currentdir}/{file}.tmp {currentdir}/{file}")

    # note that we assume each file
    #   contains the same utterances across each
    texts, textprevs, textctcs = [], [], []
    for task in tasks:
        texts.append(f"text.{task}" if task != "pr" else "text")
        textprevs.append(f"text.{task}_prev" if task != "pr" else "text.prev")
        textctcs.append(f"text.{task}_ctc" if task != "pr" else "text.ctc")
    for file in texts + textprevs + textctcs:
        os.system(
            (
                f"awk 'NR % {ratio} == 0' "
                f"{olddir}/{dataset}/{file} > {currentdir}/texts/{file}"
            )
        )

    # 3. combine text files
    for file in texts:
        os.system(f"cat {currentdir}/texts/{file} >> {currentdir}/text")
    for file in textprevs:
        os.system(f"cat {currentdir}/texts/{file} >> {currentdir}/text.prev")
    for file in textctcs:
        os.system(f"cat {currentdir}/texts/{file} >> {currentdir}/text.ctc")
    os.system(f"rm -rf {currentdir}/texts")

    # we need to sort the files by utterance ID
    # we concatenated each task's utterances separately, which messes up the order
    os.system(
        (
            "utils/fix_data_dir.sh "
            '--utt_extra_files "utt2num_samples text.ctc text.prev" '
            f"{currentdir}"
        )
    )

    print("Finished!")


def filter(olddir, newdir, dataset, key):
    print(f"-> Finding lines with '{key}' in {dataset}...")
    currentdir = f"{newdir}/{dataset}"
    # direct copy: feats_type & audio_format
    os.system(f"cp {olddir}/{dataset}/feats_type {currentdir}/feats_type")
    os.system(f"echo flac.ark > {currentdir}/audio_format")
    # filtering: find lines with key
    for file in ["spk2utt", "text", "utt2spk", "wav.scp", "utt2num_samples"]:
        os.system(f"grep '{key}' {olddir}/{dataset}/{file} > {currentdir}/{file}")


def rename(currentdir, key, newkey):
    print(f"-> Replacing '{key}' with '{newkey}' in wav.scp...")
    os.system(
        f"sed 's|{key}|{newkey}|g' {currentdir}/wav.scp > {currentdir}/wav.scp.tmp"
    )
    os.system(f"mv {currentdir}/wav.scp.tmp {currentdir}/wav.scp")
    for i in range(32):
        formatdir = f"{currentdir}/data/format.{i + 1}"
        os.system(
            f"sed 's|{key}|{newkey}|g' {formatdir}/wav.scp > {formatdir}/wav.scp.tmp"
        )
        os.system(f"mv {formatdir}/wav.scp.tmp {formatdir}/wav.scp")
    print("Finished!")


def genwav(olddir, currentdir, nsplit=32):
    print("-> Setting up splits...")
    os.system(f"mkdir {currentdir}/data")

    # 1. form n splits
    os.system(f"split -d -n l/{nsplit} {currentdir}/wav.scp {currentdir}/data/wav.")
    os.system(
        (
            f"split -d -n l/{nsplit} "
            f"{currentdir}/utt2num_samples {currentdir}/data/samples."
        )
    )
    for i in range(nsplit):
        formatdir = f"{currentdir}/data/format.{i + 1}"
        os.system(f"mkdir {formatdir}")
        os.system(f"mv {currentdir}/data/wav.{i:02d} {formatdir}/wav.scp")
        os.system(f"mv {currentdir}/data/samples.{i:02d} {formatdir}/utt2num_samples")

    # 2. In each split, get new data_wav.ark and wav.scp
    for i in tqdm(range(nsplit)):
        formatdir = f"{currentdir}/data/format.{i + 1}"
        os.system(
            f"sed 's|dump/raw|{olddir}|g' {formatdir}/wav.scp > {formatdir}/wav.scp.tmp"
        )
        os.system(f"mv {formatdir}/wav.scp.tmp {formatdir}/wav.scp")
        d = kaldiio.load_scp(f"{formatdir}/wav.scp")
        kaldiio.save_ark(
            f"{formatdir}/data_wav.ark",
            d,
            write_function="soundfile",
            scp=f"{formatdir}/wav.scp",
        )

    # 3. Update wav.scp by combining wav.scp in data/format.1~nsplit
    os.system(f"rm {currentdir}/wav.scp")
    for i in range(nsplit):
        formatdir = f"{currentdir}/data/format.{i + 1}"
        os.system(f"cat {formatdir}/wav.scp >> {currentdir}/wav.scp")

    print(f"Finished!")


def combine(newdir, datasets, remove=False):
    print("-> Combine all datasets...")
    for dataset in datasets:
        for file in [
            "spk2utt",
            "utt2spk",
            "wav.scp",
            "utt2num_samples",
            "text",
            "text.prev",
            "text.ctc",
        ]:
            os.system(f"cat {newdir}/{dataset}/{file} >> {newdir}/{file}")
        if remove:
            os.system(f"rm -rf {newdir}/{dataset}")
    print("Finished!")


if __name__ == "__main__":
    basedir = "dump/raw"
    subsample(basedir, "train", basedir, "train_n", ratio=1)
    subsample(basedir, "dev", basedir, "dev_n", ratio=1)
    os.system(f"mv {basedir}/train {basedir}/train_full")
    os.system(f"mv {basedir}/dev {basedir}/dev_full")
    os.system(f"mv {basedir}/train_n {basedir}/train")
    os.system(f"mv {basedir}/dev_n {basedir}/dev")
    """
    # Getting the 1k subset
    subsample("dump/raw", "dev_full", "dump/raw", "dev_1k", ratio=80)
    subsample("dump/raw", "train_full", "dump/raw", "train_1k", ratio=80)
    """
