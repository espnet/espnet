#!/usr/bin/env python3
# encoding: utf-8 -*-

# Copyright 2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import codecs
import logging
import os
import re
import shutil
import sys

import configargparse


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Get transcription for KsponSpeech dataset (aihub.or.kr).",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # general configuration
    parser.add_argument(
        "--raw-trans",
        type=str,
        required=True,
        help="Filename of raw transcription ({text-filename} :: {raw-text})",
    )
    parser.add_argument("--out-fn", type=str, help="Output filename of refining script")
    parser.add_argument(
        "--type",
        type=str,
        default="dt",
        choices=["df", "dt", "fl"],
        help="""Type of output transcription ('df', 'dt', or 'fl').
                If type is df, output to disfluent transcription
                               with repeated and filler words
                Else if type is dt, output to disfluent transcription
                               with '/' and '+' tags for repeated and filler words
                Else if type is fl, output to fluent transcription
                               without repeated and filler words""",
    )
    parser.add_argument(
        "--notation-type",
        type=str,
        default="char",
        help="""Notation of transcription ('char' or 'pron').
                If notation-type is char, extract from string ($1)/($2) -> $1
                Else if notation-type is pron, extract from string ($1)/($2) -> $2""",
    )
    parser.add_argument(
        "--unk-sym",
        type=str,
        default="[unk]",
        help="If unk-sym is empty(''), remove the unknown symbol (\\u)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory name that saves information for each stage",
    )
    parser.add_argument("--clear", action="store_true", help="Remove log directory")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Start from 0 if you need to start from raw-trans data gathering",
    )
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    return parser


def split_fn_trans(fn_trans):
    fn_trans_tok = fn_trans.split()
    fn = fn_trans_tok[0]  # utt-id
    trans = " ".join(fn_trans_tok[2:])  # transcription
    return fn, trans


def main(args):
    """Run the main refining function."""
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check input arguments
    if not os.path.exists(args.raw_trans) and args.stage > 0:
        logging.error("Filename '" + args.raw_trans + "' is not exist")
        logging.error("Please check the '--raw-trans' argument")
        sys.exit(1)
    else:
        logging.info("Loading raw text: " + args.raw_trans)

    # make logging dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # run refining processes
    logging.debug(
        "stage 1: Remove meta-symbols for unwanted pause and elongated segment"
    )
    in_fn = args.raw_trans
    out_fn = "1_remove_meta_symbols.txt"
    if args.raw_trans is not None and args.stage <= 1:
        logging.debug(">> reading %s", in_fn)
        with codecs.open(in_fn, mode="r", encoding="utf-8") as f:
            trans_raw = f.readlines()

        # save log-file of stage 1
        trans_rm_meta_sym = codecs.open(
            args.log_dir + "/" + out_fn, mode="w", encoding="utf-8"
        )
        for ix, _line in enumerate(trans_raw):
            fn_trans = _line.strip()
            fn, trans = split_fn_trans(fn_trans)  # txt_path, transcription

            line = trans.replace("*", "").strip()  # symbol for noisy speech
            line = line.replace("o/", "").strip()  # symbol for overlapped signals
            line = line.replace("l/", "").strip()  # symbol for laugh signals
            line = line.replace("b/", "").strip()  # symbol for breath signals
            line = line.replace("n/", "").strip()  # symbol for noise signals
            line = line.replace("u/", args.unk_sym).strip()  # symbol for unknown words
            line = " ".join(line.split())  # remove multiple space

            trans_rm_meta_sym.write(str(fn) + " :: " + str(line).strip() + "\n")
        trans_rm_meta_sym.close()

    logging.debug("stage 2: Remove repeated symbols")
    in_fn = "1_remove_meta_symbols.txt"
    out_fn = "2_remove_repeat_symbols.txt"
    if os.path.exists(args.log_dir + "/" + in_fn) and args.stage <= 2:
        logging.debug(">> reading %s", args.log_dir + "/" + in_fn)
        with codecs.open(args.log_dir + "/" + in_fn, mode="r", encoding="utf-8") as f:
            trans_rm_meta_sym = f.readlines()

        # save log-file of stage 2
        trans_rm_repeat_sym = codecs.open(
            args.log_dir + "/" + out_fn, mode="w", encoding="utf-8"
        )
        for ix, _line in enumerate(trans_rm_meta_sym):
            fn_trans = _line.strip()
            fn, trans = split_fn_trans(fn_trans)  # txt_path, transcription

            line = trans.replace(
                " +", "+ "
            ).strip()  # fix the notation errors ('w1 +w2' -> 'w1+ w2')
            if args.type == "fl":  # remove repeated words
                line = re.sub(
                    r"\(([^\)]+)\)/\(([^\)]+)\)[^ \+]*\+", "", line
                ).strip()  # ('(w1)/(w1)+' -> '')
                line = re.sub(
                    r"[^ \+]+[\+]($|\s)", " ", line
                ).strip()  # remove repeated words ('w1+' -> '')
                line = re.sub(
                    r".*\+.*", "", line
                ).strip()  # remove error sentences ('w1+w2' -> '')
            elif args.type == "df":  # remain repeated words
                line = re.sub(
                    r"\(([^\)]+)\)/\(([^\)]+)\)([^ \+]*)\+", r"(\1)/(\2)\3", line
                ).strip()
                line = re.sub(
                    r"([^ \+]+)[\+]($|\s)", r"\1 ", line
                ).strip()  # remove repeated symbol ('w1+' -> 'w1')
                line = re.sub(
                    r"([^ \+]+)[\+](\,|\.|\?|\!)", r"\1\2", line
                ).strip()  # remove repeated symbol ('w1+,' -> 'w1')
                line = re.sub(
                    r".*\+.*", "", line
                ).strip()  # remove mis-used special symbol ('w1+w2' -> '')
            else:  # remain repeated words and symbol '+'
                line = line.strip()
            line = " ".join(line.split())  # remove multiple space

            trans_rm_repeat_sym.write(str(fn) + " :: " + str(line).strip() + "\n")
        trans_rm_repeat_sym.close()

    logging.debug("stage 3: Select transcription notation")
    in_fn = "2_remove_repeat_symbols.txt"
    out_fn = "3_select_notation.txt"
    if os.path.exists(args.log_dir + "/" + in_fn) and args.stage <= 3:
        logging.debug(">> reading %s", args.log_dir + "/" + in_fn)
        with codecs.open(args.log_dir + "/" + in_fn, mode="r", encoding="utf-8") as f:
            trans_raw = f.readlines()

        # save log-file of stage 3
        trans_notation = codecs.open(
            args.log_dir + "/" + out_fn, mode="w", encoding="utf-8"
        )
        for ix, _line in enumerate(trans_raw):
            fn_trans = _line.strip()
            fn, trans = split_fn_trans(fn_trans)  # txt_path, transcription

            line = re.sub(r"\)\s*/\s*\(", r")/(", trans)
            if args.notation_type == "char":
                line = re.sub(
                    r"\(([^\)]+)\)/\(([^\)]+)\)", r"\1", line
                )  # ($1)/($2) -> $1
            else:
                line = re.sub(
                    r"\(([^\)]+)\)/\(([^\)]+)\)", r"\2", line
                )  # ($1)/($2) -> $2
            if re.search(r"[\(\)]]", line):
                logging.warning(
                    "INVALID EXPR at line %d - %s" % (ix + 1, _line.strip())
                )
                continue

            trans_notation.write(str(fn) + " :: " + str(line).strip() + "\n")
        trans_notation.close()

    logging.debug("stage 4: Remove filler words")
    in_fn = "3_select_notation.txt"
    out_fn = "4_remove_filler_words.txt"
    if os.path.exists(args.log_dir + "/" + in_fn) and args.stage <= 4:
        logging.debug(">> reading %s", args.log_dir + "/" + in_fn)
        with codecs.open(args.log_dir + "/" + in_fn, mode="r", encoding="utf-8") as f:
            trans_rm_repeat_sym = f.readlines()

        # save log-file of stage 4
        trans_rm_filler_word = codecs.open(
            args.log_dir + "/" + out_fn, mode="w", encoding="utf-8"
        )
        for ix, _line in enumerate(trans_rm_repeat_sym):
            fn_trans = _line.strip()
            fn, trans = split_fn_trans(fn_trans)  # txt_path, transcription

            if args.type == "fl":  # remove filler words
                line = re.sub(
                    r"[^ \+]+[\/]", "", trans
                ).strip()  # remove filler words ('/')
            elif args.type == "df":  # remain filler words
                line = re.sub(
                    r"([^ \+]+)[\/]", r"\1", trans
                ).strip()  # remain filler words ('/')
            else:  # remain filler words with symbol '/'
                line = trans.strip()

            trans_rm_filler_word.write(str(fn) + " :: " + str(line).strip() + "\n")
        trans_rm_filler_word.close()

    logging.debug("stage 5: Remove miss punctuation marks")
    in_fn = "4_remove_filler_words.txt"
    out_fn = "5_remove_error_punctuation_marks.txt"
    if os.path.exists(args.log_dir + "/" + in_fn) and args.stage <= 5:
        logging.debug(">> reading %s", args.log_dir + "/" + in_fn)
        with codecs.open(args.log_dir + "/" + in_fn, mode="r", encoding="utf-8") as f:
            trans_rm_repeat_sym = f.readlines()

        # save log-file of stage 5
        trans_rm_error_punctuation_mark = codecs.open(
            args.log_dir + "/" + out_fn, mode="w", encoding="utf-8"
        )
        for ix, _line in enumerate(trans_rm_repeat_sym):
            fn_trans = _line.strip()
            fn, trans = split_fn_trans(fn_trans)  # txt_path, transcription

            # remove error punctuation marks
            line = trans.upper()  # replace lowercase with uppercase letters
            line = re.sub(r"^[\,\.\?\!]+ ", " ", line).strip()  # '. w1' -> 'w1'
            line = re.sub(r" [\,\.\?\! ]+ ", " ", line).strip()  # 'w1 . w2 ' -> 'w1 w2'
            line = re.sub(r" [\,\.\?\!]+$", " ", line).strip()  # 'w1 .' -> 'w1'
            line = re.sub(
                r"(^| )[\,\.\?\!]([^ ]+)", r" \2", line
            ).strip()  # '.w1' -> 'w1'
            line = re.sub(
                r"^[\,\.\?\!]$", "", line
            ).strip()  # if the line has only '.', remove it
            line = line.replace("#", "").strip()
            line = line.replace("/", "").strip() if args.type != "dt" else line.strip()
            line = " ".join(line.split())  # remove multiple space

            if len(line) > 0:  # remove empty lines
                trans_rm_error_punctuation_mark.write(
                    str(fn) + " :: " + str(line).strip() + "\n"
                )
        trans_rm_error_punctuation_mark.close()

    # save refined transcription
    logging.debug("stage 6: Save refined transcription")
    out_fn = (
        os.path.basename(args.raw_trans) + "." + str(args.type) + ".out"
        if args.out_fn is None
        else args.out_fn
    )
    shutil.copy(args.log_dir + "/" + "5_remove_error_punctuation_marks.txt", out_fn)

    if args.clear:  # remove log directory and files
        shutil.rmtree(args.log_dir)

    logging.info("Succeeded creating transcription: " + out_fn)


if __name__ == "__main__":
    main(sys.argv[1:])
