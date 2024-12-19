import argparse
import os
import re
import statistics
import string
import unicodedata

from jiwer import cer

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir")
args = parser.parse_args()


def remove_punctuation(sentence):
    new_sentence = ""
    for char in sentence:
        # all unicode punctuation is of type P
        if unicodedata.category(char).startswith("P"):
            continue
        else:
            new_sentence = f"{new_sentence}{char}"
    return new_sentence


def normalize_and_calculate_cer(ref, hyp, remove_spaces):
    if remove_spaces:
        hyp = re.sub(r"\s", "", hyp)
        ref = re.sub(r"\s", "", ref)

    # remove punctuation
    hyp = remove_punctuation(hyp)
    ref = remove_punctuation(ref)

    hyp = hyp.upper()
    ref = ref.upper()
    if len(ref) == 0:
        return -1
    return cer(ref, hyp)


def calculate_acc(hyps, refs):
    acc = 0
    for hyp, ref in zip(hyps, refs):
        if hyp == ref:
            acc += 1
    return acc / (len(refs))


def score(references, lids, hyps):
    # utts: zip of (reference lid tag + " " + reference transcript, hyp lid tag + " " + hyp text)
    all_cers = []
    all_accs = []
    remove_space_langs = ["[cmn]", "[jpn]", "[tha]", "[yue]"]
    langs = list(set(lids))
    for lang in langs:
        lang_cers = []
        lang_acc_hyps = []
        lang_acc_refs = []
        for ref, lid, hyp in zip(references, lids, hyps):
            if lid == lang:
                if lang in remove_space_langs:
                    remove_spaces = True
                else:
                    remove_spaces = False

                # hyp/ref format is [iso] this is an utt
                lang_cer = normalize_and_calculate_cer(
                    ref[5:].strip(), hyp[5:].strip(), remove_spaces
                )

                # guard against empty reference
                if lang_cer < 0:
                    continue

                lang_cers.append(lang_cer)
                lang_acc_hyps.append(hyp[0:5])
                lang_acc_refs.append(lid)

        all_accs.append(calculate_acc(lang_acc_hyps, lang_acc_refs))
        all_cers.append(sum(lang_cers) / len(lang_cers))  # average CER of a language

    all_cers.sort(reverse=True)
    print(f"LID ACCURACY: {sum(all_accs) / len(all_accs)}")
    print(f"AVERAGE CER: {sum(all_cers) / len(all_cers)}")
    print(f"CER Standard Deviation: {statistics.stdev(all_cers)}")
    print(f"WORST 15 Lang CER: {sum(all_cers[0:15]) / 15}")


reference_text = open("data/dev/text").readlines()
reference_lids = [line.split()[1] for line in reference_text]
reference_text = [line.split(" ", 1)[1] for line in reference_text]

dirs = os.listdir(args.exp_dir)
for directory in dirs:
    if "decode_asr" in directory:
        print(directory)
        hypothesis_text = open(f"{args.exp_dir}/{directory}/org/dev/text").readlines()
        hypothesis_text = [line.split(" ", 1)[1] for line in hypothesis_text]

        assert len(hypothesis_text) == len(reference_text) == len(reference_lids)
        score(reference_text, reference_lids, hypothesis_text)
