"""
Calculate sentence-level and speaker-level Aphasia detection accuracy
"""

import argparse
from collections import Counter

from data import spk2aphasia_label, utt2spk


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Will print which sentence is incorrect if this is given",
    )
    parser.add_argument(
        "--field",
        "-f",
        type=int,
        default=1,
        help="The 0-based index of the field that is APH tag prediction",
    )
    return parser.parse_args()


def main():
    args = get_args()

    utt_num = 0
    correct_aph = 0

    spk2tags = {}
    spk2ref = {}

    utt2text = {}
    if args.text is not None:
        with open(args.text, encoding="utf-8") as f:
            for line in f:
                utt, text = line.rstrip("\n").split(maxsplit=1)
                utt2text[utt] = text

    # sentence-level scoring
    spk2incorrect_sent = {}
    spk2correct_sent = {}
    with open(args.hyp, encoding="utf-8") as f:
        for hyp in f:
            hyp = hyp.strip().split()  # utt [APH]

            utt_id = hyp[0]
            hyp_aph = ""
            if len(hyp) > args.field:
                hyp_aph = hyp[args.field]
            else:
                print(f"WARNING: {utt_id} has no APH tag output")

            spk = utt2spk(utt_id)
            ref_aph = spk2aphasia_label[spk]
            ref_aph = f"[{ref_aph.upper()}]"
            spk2ref[spk] = ref_aph

            spk2tags.setdefault(spk, []).append(hyp_aph)

            if ref_aph == hyp_aph:
                correct_aph += 1
                spk2correct_sent.setdefault(spk, 0)
                spk2correct_sent[spk] += 1
            else:
                spk2incorrect_sent.setdefault(spk, 0)
                spk2incorrect_sent[spk] += 1

                if utt_id in utt2text:
                    print(
                        f"Incorrect prediction (should be {ref_aph}) of "
                        f"{utt_id}: {utt2text[utt_id]}"
                    )

            utt_num += 1

    for spk in spk2tags:
        incorrect = spk2incorrect_sent.get(spk, 0)
        correct = spk2correct_sent.get(spk, 0)
        print(
            f"Correct/incorrect sentence-level prediction for {spk}: "
            f"{correct}/{incorrect}"
        )

    print("=" * 80)
    print(
        f"Sentence-level Aphasia detection accuracy "
        f"{(correct_aph / float(utt_num)):.4f} ({correct_aph}/{utt_num})"
    )
    print("=" * 80)

    # speaker-level scoring
    correct_speakers = 0
    n_speakers = 0
    for spk, tags in spk2tags.items():
        count = Counter(tags)
        n_aph = count["[APH]"]
        n_nonaph = count["[NONAPH]"]

        pred = "[NONAPH]"
        if n_aph >= n_nonaph:
            pred = "[APH]"

        if spk2ref[spk] == pred:
            correct_speakers += 1
            print(f"CORRECT majority voted prediction for {spk}: {count}")
        else:
            print(f"INCORRECT majority voted prediction for {spk}: {count}")
        n_speakers += 1

    print("=" * 80)
    print(
        f"Speaker-level Aphasia detection accuracy "
        f"{(correct_speakers / float(n_speakers)):.4f} "
        f"({correct_speakers}/{n_speakers})"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
