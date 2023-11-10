import argparse
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from iso639 import languages as iso_languages

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)

try:
    from datasets import load_dataset
except Exception:
    traceback.print_exc()
    logging.warning("Error importing datasets library")
    exit()


def preprocess_text(text: str) -> str:
    # Note(jinchuan): not sure how should we treat a text
    # that is naturally with "<" or ">"
    if "<" in text or ">" in text:
        logging.warning(f"find an invalid text: {text}")
        text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split())
    return text


def collect_data(examples, split, prefix, lang_id_dict):
    ans = []
    for idx, eg in enumerate(examples):
        text = preprocess_text(eg["raw_transcription"])
        if not len(text) > 0:
            logging.warning(f"skip the example due to empty text: {text}")
            continue

        if not Path(eg["audio"]["path"]).is_file():
            logging.warning(
                f"skip the example due to missing file: {eg['audio']['path']}"
            )
            continue

        if eg["language"] not in lang_id_dict:
            logging.warning(
                f"skip the example due to unknown language: {eg['language']}"
            )
            continue

        speech_id = eg["path"].split("/")[-1].replace(".wav", "")
        length = eg["num_samples"] / eg["audio"]["sampling_rate"]

        ans.append(
            [
                Utterance(
                    utt_id=f"{prefix}_{speech_id}",
                    wav_id=f"{prefix}_{speech_id}",
                    wav_path=eg["path"],
                    start_time=0,
                    end_time=length,
                    lang=f"<{lang_id_dict[eg['language']]}>",
                    task="<asr>",
                    text=text,
                    asr_text=text,
                )
            ]
        )

        if idx > 0 and idx % 10 == 0:
            logging.info(f"processed {idx} examples")
    return ans


def main():
    parser = argparse.ArgumentParser(description="Download and format FLEURS dataset")
    parser.add_argument(
        "--lang",
        default="all",
        type=str,
        help="language to download data for (default: all languages)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="path to cache the data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory for prepared datasets",
    )

    parser.add_argument(
        "--prefix",
        default="FLEURS",
        type=str,
        help="data prefix",
    )

    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    fleurs_asr = load_dataset(
        "google/xtreme_s", f"fleurs.all", cache_dir=args.cache, num_proc=16
    )

    # Language IDs are in ISO-639-3 format.
    # Some of the results are not exactly identical to:
    # https://arxiv.org/pdf/2205.12446.pdf since we only
    # search the ISO code by the name.
    lang_id_dict = {}
    all_lang_ids = (
        fleurs_asr["train"].features["lang_id"].names
        + fleurs_asr["validation"].features["lang_id"].names
        + fleurs_asr["test"].features["lang_id"].names
    )
    for lang_id in list(set(all_lang_ids)):
        lang_id_short = lang_id.split("_")[0]
        if len(lang_id_short) == 2:
            lang = iso_languages.get(alpha2=lang_id_short)
        else:
            lang = iso_languages.get(part3=lang_id_short)
        lang_id_iso, lang_name = lang.part3, lang.name
        lang_id_dict[lang_name] = lang_id_iso

    # Add missing terms manually.
    lang_id_dict["Sorani-Kurdish"] = "ckb"
    lang_id_dict["Greek"] = "ell"
    lang_id_dict["Fula"] = "ful"
    lang_id_dict["Kamba"] = "kam"
    lang_id_dict["Khmer"] = "khm"
    lang_id_dict["Kyrgyz"] = "kir"
    lang_id_dict["Luo"] = "luo"
    lang_id_dict["Malay"] = "mas"
    lang_id_dict["Norwegian"] = "nob"
    lang_id_dict["Nepali"] = "npi"
    lang_id_dict["Northern-Sotho"] = "nso"
    lang_id_dict["Occitan"] = "oci"
    lang_id_dict["Oriya"] = "ori"
    lang_id_dict["Punjabi"] = "pan"
    lang_id_dict["Pashto"] = "pus"
    lang_id_dict["Swahili"] = "swh"
    lang_id_dict["Fula"] = "ful"
    lang_id_dict["Cantonese Chinese"] = "yue"
    lang_id_dict["Mandarin Chinese"] = "zho"

    talk_splits = {
        "valid": collect_data(
            fleurs_asr["validation"], "valid", args.prefix, lang_id_dict
        ),
        "test": collect_data(fleurs_asr["test"], "test", args.prefix, lang_id_dict),
        "train": collect_data(fleurs_asr["train"], "train", args.prefix, lang_id_dict),
    }

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split, talks in talk_splits.items():
        write_dir = args.output_dir / split
        write_dir.mkdir(parents=True, exist_ok=True)

        wavscp_fp = open(write_dir / "wav.scp", "w")  # wav-id wav-path
        segments_fp = open(
            write_dir / "segments", "w"
        )  # utt-id wav-id start-time end-time
        text_fp = open(write_dir / "text", "w")  # utt-id transcript
        textprev_fp = open(write_dir / "text.prev", "w")
        textctc_fp = open(
            write_dir / "text.ctc", "w"
        )  # text for ASR CTC w/o special tokens
        utt2spk_fp = open(write_dir / "utt2spk", "w")

        for talk in talks:
            for u in generate_long_utterances(talk):
                wavscp_fp.write(f"{u.wav_id} {u.wav_path}\n")
                segments_fp.write(
                    f"{u.utt_id} {u.wav_id} {u.start_time:.2f} {u.end_time:.2f}\n"
                )
                text_fp.write(f"{u.utt_id} {u.lang}{u.task}{u.text_with_time}\n")
                textprev_fp.write(f"{u.utt_id} {u.prev_text}\n")
                textctc_fp.write(f"{u.utt_id} {u.asr_text}\n")
                utt2spk_fp.write(f"{u.utt_id} {u.utt_id}\n")

        wavscp_fp.close()
        segments_fp.close()
        text_fp.close()
        textprev_fp.close()
        textctc_fp.close()
        utt2spk_fp.close()

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        *[f"<{lang}>" for lang in lang_id_dict.values()],
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")


if __name__ == "__main__":
    main()
