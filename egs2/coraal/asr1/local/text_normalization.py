import collections
import glob
import math
import re
import sys

import inflect
import pandas as pd
from text2digits import text2digits

# adapted from https://github.com/cmu-llab/s3m-aave/blob/master/
#   data/preprocess.py
#   which was adapted from Koenecke et al 2020
#   https://github.com/stanford-policylab/asr-disparities/blob/master/
#   src/transcript_cleaning_functions.py


def fix_numbers(text):
    # Standardize number parsing and dollars
    split_words_num = text.split()
    new_list = []
    for i in range(len(split_words_num)):
        x = split_words_num[i]

        # deal with years
        if x.isdigit():
            if (1100 <= int(x) < 2000) or (2010 <= int(x) < 2100) or (int(x) == 5050):
                # deal with years as colloquially spoken
                new_word = p.number_to_words(x[:2]) + " " + p.number_to_words(x[2:])
            else:
                new_word = p.number_to_words(x)

        # deal with cases like 1st, 2nd, etc.
        elif re.match(r"(\d+)(\w+)", x, re.I):
            single_digits = ["1st", "2nd", "3rd", "5th", "8th", "9th"]
            double_digits = ["12th"]
            single_num = ["1", "2", "3", "5", "8", "9"]
            double_num = ["12"]
            single_digit_labels = [
                "first",
                "second",
                "third",
                "fifth",
                "eighth",
                "ninth",
            ]
            double_digit_labels = ["twelfth"]
            all_digits = single_digits + double_digits
            all_labels = single_digit_labels + double_digit_labels
            if x in all_digits:
                new_word = all_labels[all_digits.index(x)]
            else:
                items = re.match(r"(\d+)(\w+)", x, re.I).groups()
                if items[1] not in ["s", "th", "st", "nd", "rd"]:
                    new_word = fix_numbers(items[0]) + " " + items[1]
                elif items[0][-2:] in double_num:
                    new_word = (
                        fix_numbers(str(100 * int(items[0][:-2])))
                        + " "
                        + fix_numbers(items[0][-2:] + items[1])
                    )
                elif (items[0][-1:] in single_num) and items[0][-2:-1] != "1":
                    try:
                        new_word = (
                            fix_numbers(str(10 * int(items[0][:-1])))
                            + " "
                            + fix_numbers(items[0][-1:] + items[1])
                        )
                    except Exception:
                        new_word = fix_numbers(items[0]) + items[1]
                # deal with case e.g. 80s
                elif (items[1] in ["s", "th"]) and (
                    p.number_to_words(items[0])[-1] == "y"
                ):
                    new_word = fix_numbers(items[0])[:-1] + "ie" + items[1]
                else:
                    new_word = fix_numbers(items[0]) + items[1]

        # deal with dollars
        elif re.match(r"\$[^\]]+", x, re.I):
            # deal with $ to 'dollars'
            money = fix_numbers(x[1:])
            if x[1:] in ["1", "a"]:
                new_word = money + " dollar"
            else:
                new_word = money + " dollars"

        elif re.match(r"\£[^\]]+", x, re.I):
            # deal with £ to 'pounds'
            money = fix_numbers(x[1:])
            if x[1:] in ["1", "a"]:
                new_word = money + " pound"
            else:
                new_word = money + " pounds"

        else:
            new_word = x

        new_list.append(new_word)

    text = " ".join(new_list)
    # NSP needs // (overlapping speech), [] (dysfluencies, [laugh])
    #   before forced alignment
    # this script assumes [] (overlapping speech), // (annotations)
    #   have already been removed in CORAAL
    text = re.sub(r"[^\s\w$\'\[\]\/]|_", " ", text)

    # Deal with written out years (two thousand and ten -> twenty ten)
    for double_dig in range(10, 100):
        double_dig_str = p.number_to_words(double_dig)
        text = re.sub(
            "two thousand and " + double_dig_str,
            "twenty " + double_dig_str,
            text.lower(),
        )
        text = re.sub(
            "two thousand " + double_dig_str, "twenty " + double_dig_str, text.lower()
        )

    text = re.sub(r"\s+", " ", "".join(text))  # standardize whitespace

    return text


def normalize_text(text):
    # Relabel CORAAL words
    split_words = text.split()
    split_words = [x if x != "busses" else "buses" for x in split_words]
    split_words = [x if x != "aks" else "ask" for x in split_words]
    split_words = [x if x != "aksing" else "asking" for x in split_words]
    split_words = [x if x != "aksed" else "asked" for x in split_words]
    text = " ".join(split_words)

    # expand abbrevations specific to NSP before text normalization
    text = text.replace("St Louis", "Saint Louis")
    text = text.replace("St. Louis", "Saint Louis")
    text = text.replace("St Patty", "Saint Patty")
    text = text.replace("St. Patty", "Saint Patty")
    text = text.replace("St. John", "Saint John")
    text = text.replace("3rd St.", "3rd Street")
    # 3rd will be expanded by normalize_text

    # fix spacing in certain spellings
    spacing_before_after = [
        ("carryout", "carry out"),
        ("sawmill", "saw mill"),
        ("highschool", "high school"),
        ("worldclass", "world class"),
    ]
    for before, after in spacing_before_after:
        text = re.sub(before, after, "".join(text))

    # general string cleaning
    text = re.sub(
        r"([a-z])\-([a-z])", r"\1 \2", text, 0, re.IGNORECASE
    )  # replace inter-word hyphen with space

    # replace special characters (punctuation) with space, except $ and apostrophes
    # NSP needs // (overlapping speech), [] (dysfluencies, [laugh])
    #   before forced alignment
    # this script assumes [] (overlapping speech), // (annotations)
    #   have already been removed in CORAAL
    text = re.sub(r"[^\s\w$\'\[\]\/]|_", "", text)
    # standardize whitespace
    text = re.sub(r"\s+", " ", "".join(text))

    # deal with cardinal directions
    split_words_dir = text.split()
    # requires uppercase to disambiguate
    # NSP is lowercase but does not abbreviate cardinal directions
    # needs to be done before expanding of acronyms
    #   b/c an acronym could contain N, E, S, W
    pre_cardinal = ["N", "E", "S", "W", "NE", "NW", "SE", "SW"]
    post_cardinal = [
        "North",
        "East",
        "South",
        "West",
        "Northeast",
        "Northwest",
        "Southeast",
        "Southwest",
    ]
    for i in range(len(pre_cardinal)):
        split_words_dir = [
            x if x != pre_cardinal[i] else post_cardinal[i] for x in split_words_dir
        ]
    text = " ".join(split_words_dir)

    # standardize spellings
    split_words = []

    # expand acronyms
    # seen in LibriSpeech (train 960)
    # e.g. NYC -> N Y C
    # e.g. RN -> R N
    for word in text.split():
        # length 2+, all letters (not numbers), all caps - expand out
        if (
            len(word) >= 2
            and word.isalpha()
            and word.upper() == word
            and word not in {"GMAT", "AIDS", "ARC"}
        ):
            # print(word)
            split_words += list(word)
        else:
            split_words.append(word)

    spelling_before_after = [
        ("ok", "okay"),
        ("till", "til"),
        ("yup", "yep"),
        ("imma", "ima"),
        ("mr", "mister"),
        ("ms", "miss"),
        ("mrs", "missus"),
        ("dr", "doctor"),
    ]
    for before, after in spelling_before_after:
        split_words = [x if x.lower() != before else after for x in split_words]

    # convert reduced constructions to full form
    # see page 21 of the User Guide
    reduced_to_full = [
        # have reduction
        ("musta", "must have"),
        ("woulda", "would have"),
        ("shoulda", "should have"),
        ("coulda", "could have"),
        ("mighta", "might have"),
        # to reduction
        ("gonna", "going to"),
        ("i'm'a", "ima"),
        ("hafta", "have to"),
        ("tryna", "trying to"),
        ("sposta", "supposed to"),
        ("finna", "fixing to"),
        ("gotta", "got to"),
        ("wanna", "want to"),
        ("oughta", "ought to"),
        # syllable reduction
        ("til", "till"),  # "TILL" occurs in LibriSpeech
        # note: neither CORAAL nor Librispeech uses 'cause; they use CAUSE
        # NSP uses 'cause
        # do not normalize CAUSE -> BECAUSE since sometimes CAUSE is a noun
        ("'cause", "cause"),
        # other reduction
        ("lemme", "let me"),
        ("whatchu", "what do you"),
        ("whatcha", "what are you"),
        ("gotcha", "got you"),
    ]
    # keep contractions like I'm and 'em
    for reduced, full in reduced_to_full:
        split_words = [x if x.lower() != reduced else full for x in split_words]

    # standardize filler words
    filler_before_after = [
        ("ha", "huh"),
        ("nah", "naw"),
        ("mmhmm", "mm hm"),
        ("mmhm", "mm hm"),
        ("mmm", "mm"),
        ("mhm", "mm"),
        ("uhm", "um"),
    ]
    # left as is: 'um', 'uh', 'mm', 'mhm', 'hm', 'ooh', 'woo', 'huh'
    for before, after in filler_before_after:
        split_words = [x if x.lower() != before else after for x in split_words]

    text = " ".join(split_words)

    # update numeric numbers to strings and remove $
    text = fix_numbers(text)
    text = re.sub(r"\$", "dollars", "".join(text))
    text = re.sub(r"\£", "pounds", "".join(text))

    # lowercase text
    text = text.lower()

    # update spacing in certain spellings
    # "all right" is more common in Librispeech
    spacing_list_pre = [
        "north east",
        "north west",
        "south east",
        "south west",
        "alright",
    ]
    spacing_list_post = [
        "northeast",
        "northwest",
        "southeast",
        "southwest",
        "all right",
    ]
    for i in range(len(spacing_list_pre)):
        text = re.sub(spacing_list_pre[i], spacing_list_post[i], "".join(text))

    return text


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Help: python local/text_normalization.py "
            "<path_to_transcript> <desired_output_path>"
        )
        print(
            "ex: python local/snippet_generation.py "
            "downloads/transcript.tsv.bak downloads/transcript.tsv"
        )
        exit(1)
    path_to_transcript, output_path = sys.argv[1:3]

    p = inflect.engine()
    t2d = text2digits.Text2Digits()

    # at this point, all non-linguistic markers should have been removed
    transcripts = pd.read_csv(path_to_transcript, sep="\t")
    transcripts["normalized_text"] = transcripts["content"].apply(normalize_text)
    transcripts.to_csv(output_path, sep="\t", index=None)
