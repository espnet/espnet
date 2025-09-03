"""
This script performs CTC segmentation on audio files to align them
with their corresponding text transcriptions.
Reference: Section 2.1.1 in the paper (https://arxiv.org/pdf/2506.00338)
"""

import argparse
import json
import re
from pathlib import Path

import chinese_converter
import emoji
import librosa
import torch
from espnet_model_zoo.downloader import ModelDownloader
from tqdm import tqdm

from espnet2.bin.s2t_ctc_align import CTCSegmentation
from utils import TO_ISO_LANGUAGE_CODE

owsm_langs = [
    "abk",
    "afr",
    "amh",
    "ara",
    "asm",
    "ast",
    "aze",
    "bak",
    "bas",
    "bel",
    "ben",
    "bos",
    "bre",
    "bul",
    "cat",
    "ceb",
    "ces",
    "chv",
    "ckb",
    "cmn",
    "cnh",
    "cym",
    "dan",
    "deu",
    "dgd",
    "div",
    "ell",
    "eng",
    "epo",
    "est",
    "eus",
    "fas",
    "fil",
    "fin",
    "fra",
    "frr",
    "ful",
    "gle",
    "glg",
    "grn",
    "guj",
    "hat",
    "hau",
    "heb",
    "hin",
    "hrv",
    "hsb",
    "hun",
    "hye",
    "ibo",
    "ina",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kab",
    "kam",
    "kan",
    "kat",
    "kaz",
    "kea",
    "khm",
    "kin",
    "kir",
    "kmr",
    "kor",
    "lao",
    "lav",
    "lga",
    "lin",
    "lit",
    "ltz",
    "lug",
    "luo",
    "mal",
    "mar",
    "mas",
    "mdf",
    "mhr",
    "mkd",
    "mlt",
    "mon",
    "mri",
    "mrj",
    "mya",
    "myv",
    "nan",
    "nep",
    "nld",
    "nno",
    "nob",
    "npi",
    "nso",
    "nya",
    "oci",
    "ori",
    "orm",
    "ory",
    "pan",
    "pol",
    "por",
    "pus",
    "quy",
    "roh",
    "ron",
    "rus",
    "sah",
    "sat",
    "sin",
    "skr",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "sot",
    "spa",
    "srd",
    "srp",
    "sun",
    "swa",
    "swe",
    "swh",
    "tam",
    "tat",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tig",
    "tir",
    "tok",
    "tpi",
    "tsn",
    "tuk",
    "tur",
    "twi",
    "uig",
    "ukr",
    "umb",
    "urd",
    "uzb",
    "vie",
    "vot",
    "wol",
    "xho",
    "yor",
    "yue",
    "zho",
    "zul",
]


def clean_text(text: str) -> str:
    # remove special spaces
    text = " ".join(text.strip().split())

    # remove emojis
    text = emoji.replace_emoji(text, replace="")

    # remove [xx] or (xx)
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)

    text = " ".join(text.strip().split())

    return text


def align_audio(text, wav_path, aligner):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        speech = librosa.load(wav_path, sr=16000)[0]
        res = aligner(speech, text)

    output = []
    start_err = 0.0
    end_err = 0.0
    for utt_id, (start_time, end_time, confidence) in zip(res.utt_ids, res.segments):
        fields = utt_id.split("-")
        ori_start_time = float(fields[-2]) / 100
        ori_end_time = float(fields[-1]) / 100

        start_err += abs(start_time - ori_start_time)
        end_err += abs(end_time - ori_end_time)

        output.append((utt_id, start_time, end_time, confidence))

    return output, start_err / len(res.utt_ids), end_err / len(res.utt_ids)


def process_json(file, out_dir, aligner, additional_text_cleaners=[]):
    """Segment all audios in the input json file and
    write the new jsonl file to the output directory.
    """
    fout = open(out_dir / (Path(file).stem + ".jsonl"), "w")

    audio_dir = Path(file).parent.parent / "audio"

    json_obj_lst = json.loads(open(file, "r").read())
    for json_obj in tqdm(json_obj_lst, disable=False, mininterval=30, maxinterval=300):
        audio_id = json_obj["audio_id"]
        wav_path = str(audio_dir / f"{audio_id}.flac")

        kaldi_text = ""
        raw_texts = []
        cleaned_texts = []
        for k, v in sorted(json_obj["text"].items()):
            ori = str(v)  # save the original text

            for cleaner in additional_text_cleaners:
                v = cleaner(v)

            v = clean_text(v)

            if len(v) > 0:
                kaldi_text += f"{k} {v}\n"
                raw_texts.append(ori)
                cleaned_texts.append(v)

        if len(raw_texts) > 0:
            try:
                segments, ave_start_err, ave_end_err = align_audio(
                    kaldi_text, wav_path, aligner
                )

                sample = {
                    "audio_id": audio_id,
                    "wav_path": wav_path,
                    "ave_start_err": ave_start_err,
                    "ave_end_err": ave_end_err,
                    "utts": [
                        (
                            utt_id,
                            start_time,
                            end_time,
                            confidence,
                            cleaned_text,
                            raw_text,
                        )
                        for (
                            utt_id,
                            start_time,
                            end_time,
                            confidence,
                        ), cleaned_text, raw_text in zip(
                            segments, cleaned_texts, raw_texts
                        )
                    ],
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

            except Exception as e:
                tqdm.write(f"Failed to process {audio_id}: {e}")

    fout.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_list", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    d = ModelDownloader()
    downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.2_ft_1B")

    aligner = CTCSegmentation(
        **downloaded,
        fs=16000,
        ngpu=1,
        batch_size=32,  # batched parallel decoding; reduce to fit your GPU memory
        kaldi_style_text=True,
        time_stamps="auto",
        lang_sym="<eng>",
        task_sym="<asr>",
        context_len_in_secs=2,  # left and right context in buffered decoding
    )

    with open(args.file_list, "r") as fin:
        for file in tqdm(fin):
            file = file.strip()

            lang = Path(file).parent.parent.name[:2]
            assert TO_ISO_LANGUAGE_CODE[lang] in owsm_langs
            aligner.lang_sym = f"<{TO_ISO_LANGUAGE_CODE[lang]}>"
            tqdm.write("------ Current file: " + file)
            tqdm.write("------ Current lang: " + aligner.lang_sym)

            additional_text_cleaners = []
            if lang == "zh":
                additional_text_cleaners.append(chinese_converter.to_simplified)
                tqdm.write(
                    "**** Additional text cleaner: chinese_converter.to_simplified"
                )

            out_dir = Path(file).parent.parent / "text_reseg"
            out_dir.mkdir(exist_ok=True)

            process_json(file, out_dir, aligner, additional_text_cleaners)
