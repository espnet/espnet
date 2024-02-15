"""Prepare CoVoST2 data for ASR and ST."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def collect_data(
    data_dir: Union[Path, str],
    split: str,
    prefix: str,
    src_lang: str,
    tgt_lang: str,
) -> List[List[Utterance]]:
    """Collect utterances."""
    data_dir = Path(data_dir) / f"{split}.{src_lang}-{tgt_lang}"

    ret = []
    with open(data_dir / f"text.tc.{src_lang}", "r") as f_src, open(
        data_dir / f"text.tc.{tgt_lang}", "r"
    ) as f_tgt, open(data_dir / "wav.scp", "r") as f_wav:
        for src_line, tgt_line, wav_line in zip(f_src, f_tgt, f_wav):
            uttid = tgt_line.split(maxsplit=1)[0]
            assert (
                src_line.split(maxsplit=1)[0] == uttid
                and wav_line.split(maxsplit=1)[0] == uttid
            )

            audio_file = (
                wav_line.split("ffmpeg -i ")[-1].split(" -f wav -ar")[0].strip()
            )
            assert Path(audio_file).is_file()
            duration = librosa.get_duration(filename=audio_file)

            ret.append(
                [
                    Utterance(
                        utt_id=f"{prefix}_{split}_{uttid}_{src_lang}-{tgt_lang}_asr",
                        wav_id=f"{prefix}_{split}_{uttid}_{src_lang}-{tgt_lang}",
                        wav_path=wav_line.strip().split(maxsplit=1)[-1],
                        start_time=0.0,
                        end_time=duration,
                        lang=f"<{src_lang.split('-')[0]}>",
                        task="<asr>",
                        text=src_line.strip().split(maxsplit=1)[-1],
                        asr_text=src_line.strip().split(maxsplit=1)[-1],
                    )
                ]
            )
            ret.append(
                [
                    Utterance(
                        utt_id=f"{prefix}_{split}_{uttid}_{src_lang}-{tgt_lang}_st",
                        wav_id=f"{prefix}_{split}_{uttid}_{src_lang}-{tgt_lang}",
                        wav_path=wav_line.strip().split(maxsplit=1)[-1],
                        start_time=0.0,
                        end_time=duration,
                        lang=f"<{src_lang.split('-')[0]}>",
                        task=f"<st_{tgt_lang.split('-')[0]}>",
                        text=tgt_line.strip().split(maxsplit=1)[-1],
                        asr_text=src_line.strip().split(maxsplit=1)[-1],
                    )
                ]
            )
    return ret


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
        "--splits",
        type=str,
        nargs="+",
        default=["val", "train"],
        help="Data splits to prepare.",
    )
    parser.add_argument(
        "--src_langs", type=str, nargs="+", help="Source languages: X -> En."
    )
    parser.add_argument(
        "--tgt_langs", type=str, nargs="+", help="Target languages: En -> X."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        "<en>",
        *[f"<{s.split('-')[0]}>" for s in args.src_langs],
        "<asr>",
        "<st_en>",
        *[f"<st_{t.split('-')[0]}>" for t in args.tgt_langs],
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")

    lang_pairs = [(src, "en") for src in args.src_langs] + [
        ("en", tgt) for tgt in args.tgt_langs
    ]

    for split in args.splits:
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

        for src_lang, tgt_lang in lang_pairs:
            talks = collect_data(
                data_dir=args.data_dir,
                split=split,
                prefix=args.prefix,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
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
