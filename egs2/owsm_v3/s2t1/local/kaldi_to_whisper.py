"""Transform the standard kaldi data directory into whisper (OWSM) data directory"""

import logging
from argparse import ArgumentParser
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import kaldiio
import librosa
import soundfile

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def parse_args():
    parser = ArgumentParser(description="Prepare data.")
    parser.add_argument("--data_dir", type=Path, help="Path to raw data.")
    parser.add_argument(
        "--prefix", type=str, help="Prefix that will be added to utt id."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to save the output data.",
    )
    parser.add_argument(
        "--src",
        type=str,
        help="langauge code for source langauge",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default=None,
        help="langauge code for tgt langauge. for ST only",
    )
    parser.add_argument(
        "--src_field",
        type=int,
        default=1,
        help="field of wav.scp to wav path to find length. Utt-id is excluded."
        "if -1, find with the whole pipe command",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=10,
        help="number of multi-processing to find duration",
    )
    parser.add_argument(
        "--nlsyms",
        type=str,
        nargs="+",
        default=[],
        help="nlsyms of this dataset",
    )
    parser.add_argument(
        "--lower_case",
        action="store_true",
        default=False,
        help="If set, all text is in lower case",
    )

    args = parser.parse_args()
    return args


def preprocess_text(text, nlsyms) -> str:
    for nlsym in nlsyms:
        text = text.replace(nlsym, " ")

    # Note(jinchuan): not sure how should we treat a text
    # that is naturally with "<" or ">"
    if "<" in text or ">" in text:
        logging.warning(f"find an invalid text: {text}")
        text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split())
    return text


def find_duration(tup):
    # pipe: copy from format_wav_scp.py
    if tup[1].endswith("|"):
        with kaldiio.open_like_kaldi(tup[1], "rb") as f:
            with BytesIO(f.read()) as g:
                array, rate = soundfile.read(g)
        return (tup[0], 0.0, len(array) / rate)
    else:
        # find duration with librosa (version 0.9.2)
        return (
            tup[0],
            0.0,
            librosa.get_duration(filename=tup[1]),
        )


def find_durations(datas, num_proc, src_field):
    pool = Pool(num_proc)
    if src_field >= 0:
        datas = [(k, v.strip().split()[src_field]) for k, v in datas.items()]
    else:
        datas = [(k, v.strip()) for k, v in datas.items()]
    datas = pool.map(find_duration, datas)
    pool.close()
    pool.join()
    datas = {tup[0]: tup for tup in datas}
    datas = {k: v for k, v in datas.items() if v[2] > 0.1}  # minimum length
    return datas


def process_kaldi_directory(
    data_dir,
    prefix,
    src,
    tgt,
    src_field,
    num_proc,
    nlsyms,
    lower_case,
):
    # read all files
    text_dict = open(str(data_dir / "text")).readlines()
    text_dict = [line.strip().split() for line in text_dict]
    text_dict = {line[0]: " ".join(line[1:]) for line in text_dict}

    if tgt is not None:
        tgt_dict = open(str(data_dir / "text.tgt")).readlines()
        tgt_dict = [line.strip().split() for line in tgt_dict]
        tgt_dict = {line[0]: " ".join(line[1:]) for line in tgt_dict}
    else:
        tgt_dict = None

    wav_dict = open(str(data_dir / "wav.scp")).readlines()
    wav_dict = [line.strip().split() for line in wav_dict]
    wav_dict = {line[0]: " ".join(line[1:]) for line in wav_dict}

    if (data_dir / "segments").is_file():
        logging.info("segments file found. Use it.")
        seg_dict = open(str(data_dir / "segments")).readlines()
        seg_dict = [line.strip().split() for line in seg_dict]
        seg_dict = {
            line[0]: (line[1], float(line[-2]), float(line[-1])) for line in seg_dict
        }

    # if the input file is not compatible with librosa, call get_utt2dur.sh
    # before calling this script
    elif (data_dir / "utt2dur").is_file():
        logging.info("utt2dur file found. Use it.")
        seg_dict = open(str(data_dir / "utt2dur")).readlines()
        seg_dict = [line.strip().split() for line in seg_dict]
        seg_dict = {line[0]: (line[0], 0, float(line[1])) for line in seg_dict}

    else:
        logging.info("segments not found. find the duration then")
        seg_dict = find_durations(wav_dict, num_proc, src_field)

    talks = {}
    for uttid, seg_info in seg_dict.items():
        wavid, start, end = seg_info

        if wavid not in talks:
            talks[wavid] = []

        text = preprocess_text(text_dict[uttid], nlsyms)
        text = text.lower() if lower_case else text
        if len(text) <= 0:
            logging.info(f"{uttid} has the empty text. skip")
            continue

        if tgt_dict is not None:
            tgt_text = tgt_text[uttid]
        else:
            tgt_text = text

        tgt_text = preprocess_text(tgt_text, nlsyms)
        tgt_text = tgt_text.lower() if lower_case else tgt_text
        if len(tgt_text) <= 0:
            logging.info(f"{uttid} has the empty tgt text. skip")
            continue

        talks[wavid].append(
            Utterance(
                utt_id=f"{prefix}_{wavid}_{uttid}",
                wav_id=f"{prefix}_{wavid}",
                wav_path=wav_dict[wavid],
                start_time=start,
                end_time=end,
                lang=f"<{src}>",
                task=f"<st_{tgt}>" if tgt_dict is not None else "<asr>",
                text=tgt_text,
                asr_text=text,
            )
        )

    return talks.values()


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    talks = process_kaldi_directory(
        data_dir=args.data_dir,
        prefix=args.prefix,
        src=args.src,
        tgt=args.tgt,
        src_field=args.src_field,
        num_proc=args.num_proc,
        nlsyms=args.nlsyms,
        lower_case=args.lower_case,
    )

    write_dir = args.output_dir
    write_dir.mkdir(parents=True, exist_ok=True)

    wavscp_fp = open(write_dir / "wav.scp", "w")
    segments_fp = open(write_dir / "segments", "w")
    text_fp = open(write_dir / "text", "w")
    textprev_fp = open(write_dir / "text.prev", "w")
    textctc_fp = open(write_dir / "text.ctc", "w")
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
        f"<{args.src}>",
        *SYMBOLS_TIME,
    ]
    if args.tgt is not None:
        special_tokens.insert(3, f"<{args.tgt}>")
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")


if __name__ == "__main__":
    main()
