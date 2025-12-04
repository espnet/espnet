import argparse
import json
import os
import shutil
from collections import defaultdict

import langcodes
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
from langcodes import tag_is_valid
from tqdm import tqdm

"""
Converts IPAPack++ into OWSM's expected format
"""

ASR = "<asr>"
PR = "<pr>"
G2P = "<g2p>"
P2G = "<p2g>"
TEXT_NA = "<na>"
NO_TIME = "<notimestamps>"
SAMPLE_RATE = 16000
LANG = "<LANG>"  # Should be mapping from utt_id to language code
UNK_LANG = "<unk>"
remove_space_lang = ["<cmn>", "<jpn>", "<kor>", "<tha>"]
copy_files = ["feats_type", "spk2utt", "utt2num_samples", "utt2spk", "wav.scp"]


def text_normalization(orthography):
    # most of the text normalization seems to have done
    #   in the creation of IPAPack++
    # we just need to remove punctuation and symbols
    # see local/all_symbols to see all symbols
    # see local/bad_symbols for which are removed by this regex
    return re.sub(r"\p{P}|\p{S}", "", orthography)


def draw_figure(lang2dur, subset_name, image_dir):
    # Sort by count and keep only the top 10 languages
    sorted_langs = sorted(lang2dur.items(), key=lambda item: item[1], reverse=True)[:10]

    for lang, duration in sorted_langs:
        plt.bar(lang, duration / 3600)

    plt.xlabel("Language")
    plt.ylabel("Hours")
    plt.title(f"Language Distribution in {subset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"{subset_name}_language_distribution.png"))
    plt.close()
    plt.clf()


def get_lang_duration(utt2lang, process_dir, lang_dist_json):
    lang2dur = {}  # lang -> duration (sec)

    utt2num_samples_path = os.path.join(process_dir, "utt2num_samples")
    with open(utt2num_samples_path, "r") as f:
        for line in f:
            utt_id, num_samples = line.strip().split(maxsplit=1)
            lang = utt2lang.get(utt_id, UNK_LANG)
            lang2dur[lang] = lang2dur.get(lang, 0) + float(num_samples) / SAMPLE_RATE

    json.dump(lang2dur, open(os.path.join(process_dir, lang_dist_json), "w"))
    return lang2dur


def main(root_dir, output_dir, lang_dist_json, draw_only=False):
    # from the source directory
    ROOT_DUMP_DIR = os.path.join(root_dir, "dump/raw")
    ROOT_DATA_DIR = os.path.join(root_dir, "data")
    ROOT_DF_DIR = os.path.join(root_dir, "downloads")

    columns = ["utt_id", "lang"]
    normalize_df = pd.read_csv(
        os.path.join(ROOT_DF_DIR, "transcript_normalized.csv"), usecols=columns
    )
    # doreco_df = pd.read_csv(
    #    os.path.join(ROOT_DF_DIR, "transcript_doreco.csv"), usecols=columns
    # )
    # combined_df = pd.concat([normalize_df, doreco_df])
    combined_df = normalize_df
    utt2lang = {row["utt_id"]: row["lang"] for _, row in combined_df.iterrows()}

    # target directory
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    all_dump_dirs = sorted(os.listdir(ROOT_DUMP_DIR))
    all_data_dirs = sorted(os.listdir(ROOT_DATA_DIR))
    if draw_only:
        # aggregate duration for all test directories
        test_stats = {}
        for data_dir in tqdm(all_data_dirs):
            process_dir = os.path.join(output_dir, data_dir)
            with open(os.path.join(process_dir, lang_dist_json), "r") as f:
                lang2dur = json.load(f)

            if "test" in data_dir:
                # a language could appear in multiple test sets
                for lang, dur in lang2dur.items():
                    test_stats[lang] = test_stats.get(lang, 0) + dur

            draw_figure(lang2dur, data_dir, image_dir)
        draw_figure(test_stats, "test", image_dir)
    else:
        # Process each data directory
        for data_dir in tqdm(all_data_dirs):
            print(f"Processing {data_dir}")

            dump_dir = os.path.join(ROOT_DUMP_DIR, data_dir)
            data_dir_path = os.path.join(ROOT_DATA_DIR, data_dir)
            process_dir = os.path.join(output_dir, data_dir)

            os.makedirs(process_dir, exist_ok=True)

            # Copy necessary files from dump_dir to process_dir
            if os.path.abspath(dump_dir) != os.path.abspath(process_dir):
                for file_name in copy_files:
                    src_file = os.path.join(dump_dir, file_name)
                    dst_file = os.path.join(process_dir, file_name)
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, dst_file)
            else:
                print(f"Skip copying: source and destination are the same")

            # Read orthography and phoneme sequences
            with open(os.path.join(data_dir_path, "orthography"), "r") as orth_file:
                orthography = orth_file.readlines()
            with open(os.path.join(dump_dir, "text"), "r") as phoneme_file:
                phoneme_seq = phoneme_file.readlines()
            with open(os.path.join(dump_dir, "text.ctc"), "r") as phoneme_ctc_file:
                phoneme_ctcseq = phoneme_ctc_file.readlines()

            unk_language_set = set()

            # Create mappings
            utt2orthography = {
                o.strip().split(maxsplit=1)[0]: (
                    o.strip().split(maxsplit=1)[1]
                    if len(o.strip().split(maxsplit=1)) > 1
                    else ""
                )
                for o in orthography
            }
            utt2phoneme_seq = {
                p.strip().split(maxsplit=1)[0]: p.strip().split(maxsplit=1)[1]
                for p in phoneme_seq
            }
            utt2phoneme_ctcseq = {
                p.strip().split(maxsplit=1)[0]: p.strip().split(maxsplit=1)[1]
                for p in phoneme_ctcseq
            }

            # Write text file
            # phoneme recognition
            #   text: phonemes, with task tokens
            #   text.prev: previous context for phoneme recognition (none for now)
            #   text.ctc: phonemes w/o length and break marks, no task tokens
            # speech recognition (orthography as output)
            #   text.asr: phonemes, with task tokens
            #   text.asr_prev: previous context for ASR (none for now)
            #   text.asr_ctc: phonemes w/o length and break marks, no task tokens
            # G2P (grapheme/orthography to phoneme)
            #   text.g2p: phonemes, with task tokens
            #   text.g2p_prev: previous context for G2P (graphemes)
            #   text.g2p_ctc: phonemes w/o length and break marks, no task tokens
            # P2G (phoneme to grapheme/orthography)
            #   text.p2g: with task tokens
            #   text.p2g_prev: previous context for P2G (phonemes)
            #   text.p2g_ctc: phonemes w/o length and break marks, no task tokens
            # note - the contents for each file across tasks may be the same
            #   but the utterance IDs need to be different
            with open(os.path.join(process_dir, "text"), "w") as pr_text, open(
                os.path.join(process_dir, "text.prev"), "w"
            ) as prev_text, open(
                os.path.join(process_dir, "text.ctc"), "w"
            ) as text_ctc, open(
                os.path.join(process_dir, "text.asr"), "w"
            ) as asr_text, open(
                os.path.join(process_dir, "text.asr_prev"), "w"
            ) as asr_text_prev, open(
                os.path.join(process_dir, "text.asr_ctc"), "w"
            ) as asr_text_ctc, open(
                os.path.join(process_dir, "text.g2p"), "w"
            ) as g2p_text, open(
                os.path.join(process_dir, "text.g2p_prev"), "w"
            ) as prev_g2p_text, open(
                os.path.join(process_dir, "text.g2p_ctc"), "w"
            ) as g2p_text_ctc, open(
                os.path.join(process_dir, "text.p2g"), "w"
            ) as p2g_text, open(
                os.path.join(process_dir, "text.p2g_prev"), "w"
            ) as prev_p2g_text, open(
                os.path.join(process_dir, "text.p2g_ctc"), "w"
            ) as p2g_ctc:

                for utt_id, p in utt2phoneme_seq.items():
                    p = "/" + "//".join(p.split()) + "/"
                    pctc = "/" + "//".join(utt2phoneme_ctcseq[utt_id].split()) + "/"
                    o = utt2orthography[utt_id]
                    lang = utt2lang[utt_id]
                    try:
                        if tag_is_valid(lang):
                            langcode = langcodes.get(lang).to_alpha3()
                        else:
                            langcode = langcodes.find(lang).to_alpha3()
                        if langcode == "zho":
                            LANG = "<cmn>"
                        else:
                            LANG = f"<{langcode}>"
                    except Exception:
                        unk_language_set.add(lang)
                        LANG = UNK_LANG
                    if LANG == "ina":
                        # remove Interlingua
                        continue
                    utt2lang[utt_id] = LANG

                    if LANG in remove_space_lang:
                        o = o.replace(" ", "")
                    o = text_normalization(o)

                    # each utterance is used 4 times
                    # but each instance of the utterance needs a diff utt id
                    pr_text.write(f"{utt_id}_pr {LANG}{PR}{NO_TIME} {p}\n")
                    prev_text.write(f"{utt_id}_pr {TEXT_NA}\n")
                    text_ctc.write(f"{utt_id}_pr {pctc}\n")

                    asr_text.write(f"{utt_id}_asr {LANG}{ASR}{NO_TIME} {o}\n")
                    asr_text_prev.write(f"{utt_id}_asr {TEXT_NA}\n")
                    asr_text_ctc.write(f"{utt_id}_asr {pctc}\n")

                    g2p_text.write(f"{utt_id}_g2p {LANG}{G2P}{NO_TIME} {p}\n")
                    prev_g2p_text.write(f"{utt_id}_g2p {o}\n")
                    g2p_text_ctc.write(f"{utt_id}_g2p {pctc}\n")

                    p2g_text.write(f"{utt_id}_p2g {LANG}{P2G}{NO_TIME} {o}\n")
                    prev_p2g_text.write(f"{utt_id}_p2g {p}\n")
                    p2g_ctc.write(f"{utt_id}_p2g {pctc}\n")

            # Get language distribution
            lang2dur = get_lang_duration(utt2lang, process_dir, lang_dist_json)

            # Plot language distribution
            draw_figure(lang2dur, os.path.basename(data_dir_path), image_dir)

            with open(os.path.join(process_dir, "unk_languages.txt"), "w") as f:
                for lang in unk_language_set:
                    f.write(f"{lang}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IPA Pack data")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument(
        "--output_dir", type=str, default="dump/raw", help="Output directory"
    )
    parser.add_argument(
        "--lang_dist_json",
        type=str,
        default="language_distribution.json",
        help="Language distribution JSON filename",
    )
    parser.add_argument(
        "--draw_only", action="store_true", help="Only draw the figures"
    )
    args = parser.parse_args()

    main(**vars(args))
