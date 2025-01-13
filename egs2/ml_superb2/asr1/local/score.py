import argparse
import os
import re
import statistics
import string
import unicodedata

from jiwer import cer


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
    """
    Calculates CER given normalized hyp and ref.
    Normalization removes all punctuation and uppercases all text.
    For Chinese, Thai, and Japanese, we remove spaces too.

    Example:
    I'll be going to the CMU campus. -> ILL BE GOING TO THE CMU CAMPUS
    ill be going to the see them you campus -> ILL BE GOING TO THE SEE THEM YOU CAMPUS
     我想去餐厅 我非常饿 -> 我想去餐厅我非常饿
    """

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
    assert len(references) == len(lids) == len(hyps)
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

                if len(ref) == 0 or len(lid) == 0 or len(hyp) == 0:
                    raise Exception(
                        f"You have an empty utterance somewhere! {ref, lid, hyp}"
                    )

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
    lid = round(sum(all_accs) / len(all_accs) * 100, 1)
    cer_res = round(sum(all_cers) / len(all_cers) * 100, 1)
    std = round(statistics.stdev(all_cers) * 100, 1)
    worst = round(sum(all_cers[0:15]) / 15 * 100, 1)

    print(f"LID ACCURACY: {lid}")
    print(f"AVERAGE CER: {cer_res}")
    print(f"CER Standard Deviation: {std}")
    print(f"WORST 15 Lang CER: {worst}")

    return lid, cer_res, worst, std


def score_dialect(references, lids, hyps):
    assert len(references) == len(lids) == len(hyps)
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

                if len(ref) == 0 or len(lid) == 0 or len(hyp) == 0:
                    raise Exception(
                        f"You have an empty utterance somewhere! {ref, lid, hyp}"
                    )

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

    lid = round(sum(all_accs) / len(all_accs) * 100, 1)
    cer_res = round(sum(all_cers) / len(all_cers) * 100, 1)
    print(f"DIALECT LID ACCURACY: {lid}")
    print(f"DIALECT CER: {cer_res}")

    return cer_res, lid


"""
    We assume that the reference and hypothesis are stored in the kaldi style:
        uttid00 [langid0] this is a sample text
        uttid01 [langid1] this is a sample text1
        uttid02 [langid2] this is a sample text2
    The script will look for the ref/hyp under this structure:
    root/
      - data/dev # kaldi style folder
            - text # ref file for standard dev set
      - data/dev_dialect # kaldi style folder
            - text # ref file for dialect dev set

      - exp/asr_train_asr_raw_char # your model folder
            - decode_asr_asr_model_valid.loss.ave # inference results
                - org/dev
                    - text # hyp file for standard dev set
                - dev_dialect
                    - text # hyp file for dialect dev set
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir")
    args = parser.parse_args()
    reference_text = open("data/dev/text").readlines()
    reference_lids = [line.split()[1] for line in reference_text]
    reference_text = [line.split(" ", 1)[1] for line in reference_text]

    dialect_reference_text = open("data/dev_dialect/text").readlines()
    dialect_reference_lids = [line.split()[1] for line in dialect_reference_text]
    dialect_reference_text = [line.split(" ", 1)[1] for line in dialect_reference_text]

    dirs = os.listdir(args.exp_dir)

    with open(f"{args.exp_dir}/challenge_results.md", "w") as out_f:
        out_f.write("# RESULTS\n\n")
        out_f.write("## args.exp_dir\n\n")
        out_f.write(
            "|decode_dir|Standard CER|Standard LID|"
            + "Worst 15 CER|CER StD|Dialect CER|Dialect LID|\n"
        )
        out_f.write("|---|---|---|---|---|---|---|\n")
        for directory in dirs:
            if "decode_asr" in directory:
                print(directory)
                hypothesis_text = open(
                    f"{args.exp_dir}/{directory}/org/dev/text"
                ).readlines()
                hypothesis_text = [line.split(" ", 1)[1] for line in hypothesis_text]

                assert (
                    len(hypothesis_text) == len(reference_text) == len(reference_lids)
                )
                lid, cer_res, worst, std = score(
                    reference_text, reference_lids, hypothesis_text
                )

                dialect_hypothesis_text = open(
                    f"{args.exp_dir}/{directory}/dev_dialect/text"
                ).readlines()
                dialect_hypothesis_text = [
                    line.split(" ", 1)[1] for line in dialect_hypothesis_text
                ]

                assert (
                    len(dialect_hypothesis_text)
                    == len(dialect_reference_text)
                    == len(dialect_reference_lids)
                )
                dialect_cer, dialect_lid = score_dialect(
                    dialect_reference_text,
                    dialect_reference_lids,
                    dialect_hypothesis_text,
                )

                out_f.write(
                    f"{directory}|{cer_res}|{lid}|{worst}"
                    + f"|{std}|{dialect_cer}|{dialect_lid}|\n"
                )
