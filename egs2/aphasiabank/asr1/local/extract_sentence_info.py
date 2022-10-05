"""
Create text and utt2spk

Based on https://github.com/monirome/AphasiaBank/blob/main/clean_transcriptions.ipynb
"""
import json
import os
import re
from argparse import ArgumentParser

import pylangacq as pla


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--transcript-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--spk2aphasia-type", type=str, default=None)
    return parser.parse_args()


def clean_trans(trans: str) -> str:
    trans = trans.split("\x15")[0]  # remove timestamp
    trans = trans.lower()

    # Precodes and postcodes
    trans = re.sub(r"\[[+-] \w+]?", "", trans)

    # Word errors
    trans = re.sub(
        r"\[: .+?] \[\* \S+]", "", trans
    )  # '[: put] [* p:w-ret]'   '[: princess]'
    trans = re.sub(r"\[: .+?]", "", trans)  # [: x [* n:uk]
    trans = re.sub(r"\[\* .+?]", "", trans)  # [* s:r-ret]
    trans = re.sub(r"\[\*]", "", trans)  # [*]

    # Interruption by other people
    trans = re.sub(r"\*\w+:[\w_]+", "", trans)  # *INV: thank_you

    # Comments, explanation, paralinguistic
    trans = re.sub(
        r"\[% .+?]", "", trans
    )  # [% mimics opening her eyes for the first time]
    trans = re.sub(r"\[= .+?]", "", trans)  # [= growling voice]
    trans = re.sub(r"\[=! [\w :_]+?]", "", trans)  # [=! laughing]

    # Retracing and repetition
    trans = re.sub(r"\[/+]", "", trans)
    trans = re.sub(
        r"\[x \d]", "", trans
    )  # FIXME: we shouldn't remove this but there is only one occurrence in UNH03b

    # Pauses
    trans = re.sub(r"\([\d:.]+\)", "", trans)  # (.) (..) (2:13.12)

    # Special utterance terminators
    trans = re.sub(
        r"\+[<+\",./?]+", "", trans
    )  # +... +..? +/. +/? +//. +//? +, ++ +"/. +" +". +<

    # Overlap precedes
    trans = re.sub(r"\[[<>]\d?]", "", trans)  # [<]   [<1]  [>]  [>2]

    # Special form markers (CHAT manual page 45)
    trans = re.sub(r"@\S+", "", trans)

    # Local events and gestures
    trans = re.sub(r"[<>]", "", trans)  # remove all <> before adding <xxx>
    # laughter
    trans = re.sub(r"&=laughs", "<LAU>", trans)
    trans = re.sub(r"&=chuckles", "<LAU>", trans)
    trans = re.sub(
        r"&=\w*[:\w]*", "", trans
    )  # '&=points:picture_1'  '&=clears:throat'  '&=twiddles'

    # Fillers and fragments
    # TODO: trans = re.sub(r'&-[a-z]*', '<FLR>', trans)  # &-uh  &-um
    trans = re.sub(r"&-", "", trans)  # &-uh  &-um
    trans = re.sub(r"&\+?", "", trans)  # phonological fragment

    # Special words
    trans = re.sub(r"www|xxx", "", trans)

    # Other remaining punctuations
    trans = re.sub(r"[,?!;:\"“%‘”.�‡„$^↓↑≠()\[\]↫\x15+]", "", trans)  # chars to remove
    trans = re.sub(
        r"_|-| {2,}", " ", trans
    )  # '_', '-', and multiple whitespaces are replaced with a single whitespace

    return trans.strip()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    spk2aphasia_type = None
    if args.spk2aphasia_type is not None:
        with open(args.spk2aphasia_type, encoding="utf-8") as f:
            spk2aphasia_type = json.load(f)

    # get a list of all CHAT files
    files = []
    for file in os.listdir(args.transcript_dir):
        if os.path.isfile(os.path.join(args.transcript_dir, file)) and file.endswith(
            ".cha"
        ):
            files.append(file)

    all_chars = set()
    with open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as text, open(
        os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8"
    ) as utt2spk:
        for file in files:
            spk = file.split(".cha")[0]

            path = os.path.join(args.transcript_dir, file)
            chat: pla.Reader = pla.read_chat(path)

            utts = chat.utterances(participants="PAR")
            n_utts = len(utts)
            for i in range(n_utts):
                time_marks = utts[i].time_marks
                if time_marks is None:  # skipping utterances without time marks
                    continue

                start, end = utts[i].time_marks
                utt_id = f"{spk}-{start}_{end}"

                trans = clean_trans(utts[i].tiers["PAR"])
                if len(trans) == 0:
                    continue

                # gather all unique chars
                for c in trans:
                    all_chars.add(c)

                # write lines
                if spk2aphasia_type is not None:
                    trans = f"[{spk2aphasia_type[spk]}] {trans}"
                text.write(f"{utt_id}\t{trans}\n")
                utt2spk.write(f"{utt_id}\t{spk}\n")

    print(all_chars)


if __name__ == "__main__":
    main()
