from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml


def parse_data(root: Path, lang: str, dataset: str) -> Dict[str, List[Dict]]:
    txt_path = root / f"en-{lang}" / "data" / dataset / "txt"
    wav_path = root / f"en-{lang}" / "data" / dataset / "wav"

    with open(txt_path / f"{dataset}.yaml", "r") as fp:
        utts = yaml.safe_load(fp)

    with open(txt_path / f"{dataset}.en", "r") as fp:
        src_text = [ln.strip() for ln in fp.readlines()]

    with open(txt_path / f"{dataset}.{lang}", "r") as fp:
        tgt_text = [ln.strip() for ln in fp.readlines()]

    assert len(src_text) == len(tgt_text) and len(tgt_text) == len(utts)

    wav2utts = {}
    for idx, utt in enumerate(utts):
        wav_name = utt["wav"]  # e.g.: 'ted_767.wav'
        if wav_name not in wav2utts:
            wav2utts[wav_name] = []

        utt["wav"] = str((wav_path / wav_name).resolve())  # get full path
        utt["src"] = " ".join(src_text[idx].split())  # always use standard space
        utt["tgt"] = " ".join(tgt_text[idx].split())
        wav2utts[wav_name].append(utt)

    return wav2utts


def find_max_prev(utts: List[Dict], max_sec: float, pos: int) -> List[int]:
    # ..., pos-2, pos-1 (not including pos)

    results = []
    end_time = utts[pos]["offset"]
    while pos - 1 >= 0 and (end_time - utts[pos - 1]["offset"]) <= max_sec:
        results.append(pos - 1)
        pos -= 1

    return results[::-1]


def find_max_utts(utts: List[Dict], max_sec: float, pos: int) -> List[int]:
    # pos, pos+1, pos+2, ... (including pos)

    results = []
    start_time = utts[pos]["offset"]
    while (
        pos < len(utts)
        and (utts[pos]["duration"] + utts[pos]["offset"] - start_time) <= max_sec
    ):
        results.append(pos)
        pos += 1

    return results


def time2token(x: float, resolution: float) -> str:
    x = round(x / resolution) * resolution
    return f"<{x:.2f}>"


def prepare_single(
    utts: List[Dict], wav_id: str, max_sec: float = 30, resolution: float = 0.02
) -> List[Dict]:
    """Prepare a singe TED talk by generating long utterances with context/prompt."""

    new_utts: List[Dict] = []
    uttids: List[str] = []
    for idx in range(len(utts)):
        max_utts = find_max_utts(utts, max_sec, idx)

        if max_utts:
            offset = utts[max_utts[0]]["offset"]  # start time
            src = []
            tgt = []
            for u in max_utts:
                src.extend(
                    [
                        time2token(utts[u]["offset"] - offset, resolution),
                        " " + utts[u]["src"],  # add space before text
                        time2token(
                            utts[u]["offset"] + utts[u]["duration"] - offset, resolution
                        ),
                    ]
                )
                tgt.extend(
                    [
                        time2token(utts[u]["offset"] - offset, resolution),
                        " " + utts[u]["tgt"],  # add space before text
                        time2token(
                            utts[u]["offset"] + utts[u]["duration"] - offset, resolution
                        ),
                    ]
                )
            src_ctc = [utts[u]["src"] for u in max_utts]  # for ASR CTC

            max_prev = find_max_prev(utts, max_sec, idx)
            prev_src = [utts[u]["src"] for u in max_prev]
            prev_tgt = [utts[u]["tgt"] for u in max_prev]

            long_utt = {
                "start": offset,
                "end": utts[max_utts[-1]]["offset"] + utts[max_utts[-1]]["duration"],
                "wav": utts[max_utts[0]]["wav"],
                "src": "".join(src),  # no space between special tokens
                "tgt": "".join(tgt),  # no space between special tokens
                "src_ctc": " ".join(src_ctc),
                "prev_src": " ".join(prev_src) if prev_src else "<na>",
                "prev_tgt": " ".join(prev_tgt) if prev_tgt else "<na>",
            }
            long_utt["utt_id"] = (
                f"{wav_id}_{round(1000*long_utt['start']):07d}"
                f"_{round(1000*long_utt['end']):07d}"
            )

            if long_utt["utt_id"] not in uttids:
                uttids.append(long_utt["utt_id"])
                new_utts.append(long_utt)
            else:
                print(f"Long utt {long_utt['utt_id']} already exists.")

    return new_utts


def prepare_all(
    data_path: Path,
    output_path: Path,
    datasets: List[str],
    langs: List[str],
    max_sec: float,
    resolution: float,
):
    """Prepare all TED talks."""

    for dataset in datasets:  # 'train', 'dev'
        out_dir = output_path / dataset
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)

        wavscp_fp = open(out_dir / "wav.scp", "w")  # wav-id wav-path
        segments_fp = open(out_dir / "segments", "w")  # utt-id wav-id start end
        text_fp = open(out_dir / "text", "w")
        textprev_fp = open(out_dir / "text.prev", "w")
        textctc_fp = open(
            out_dir / "text.ctc", "w"
        )  # text for ASR CTC w/o special tokens
        utt2spk_fp = open(out_dir / "utt2spk", "w")
        for lang in langs:
            print(f"Preparing {dataset} {lang}...")
            wav2utts = parse_data(data_path, lang, dataset)
            for wav_name, utts in wav2utts.items():
                wav_id = (
                    f"ted_{int(wav_name[:-len('.wav')][len('ted_'):]):05d}_en_{lang}"
                )
                long_utts = prepare_single(utts, wav_id, max_sec, resolution)
                if long_utts:
                    wavscp_fp.write(f"{wav_id} {utts[0]['wav']}\n")
                else:
                    print(f"Wav {wav_id} has no valid long utterances. Skip it.")

                for u in long_utts:
                    # 1. transcribe: en -> en
                    utt_id = f"{u['utt_id']}_transcribe"
                    category = "<en>"  # reserved for classification tasks
                    task = "<transcribe>"

                    text_fp.write(f"{utt_id} {category}{task}{u['src']}\n")
                    textprev_fp.write(f"{utt_id} {u['prev_src']}\n")
                    textctc_fp.write(f"{utt_id} {u['src_ctc']}\n")
                    utt2spk_fp.write(f"{utt_id} {wav_id}\n")
                    segments_fp.write(
                        f"{utt_id} {wav_id} {u['start']:.2f} {u['end']:.2f}\n"
                    )

                    # 2. translate: en -> lang
                    utt_id = f"{u['utt_id']}_translate{lang}"
                    category = "<en>"  # reserved for classification tasks
                    task = f"<translate{lang}>"

                    text_fp.write(f"{utt_id} {category}{task}{u['tgt']}\n")
                    textprev_fp.write(f"{utt_id} {u['prev_tgt']}\n")
                    textctc_fp.write(f"{utt_id} {u['src_ctc']}\n")
                    utt2spk_fp.write(f"{utt_id} {wav_id}\n")
                    segments_fp.write(
                        f"{utt_id} {wav_id} {u['start']:.2f} {u['end']:.2f}\n"
                    )

        wavscp_fp.close()
        segments_fp.close()
        text_fp.close()
        textprev_fp.close()
        textctc_fp.close()
        utt2spk_fp.close()


def parse_args():
    parser = ArgumentParser(description="Prepare data.")
    parser.add_argument("--data_path", type=Path, help="Path to raw data.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("./data"),
        help="Path to save the output.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["dev", "train"],
        help="Data splits that will be prepared.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=["de", "ja", "zh"],
        help="Target languages that will be prepared.",
    )
    parser.add_argument(
        "--max_sec", type=float, default=30, help="Maximum audio length."
    )
    parser.add_argument(
        "--resolution", type=float, default=0.02, help="Time resolution."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    prepare_all(
        data_path=args.data_path,
        output_path=args.output_path,
        datasets=args.datasets,
        langs=args.langs,
        max_sec=args.max_sec,
        resolution=args.resolution,
    )

    # save special tokens
    category_tokens = [
        "<nospeech>",
        "<en>",
    ]
    task_tokens = [
        "<transcribe>",
        *[f"<translate{x}>" for x in args.langs],
    ]
    timestamp_tokens = [
        "<notimestamps>",
        *[
            f"<{i * args.resolution:.2f}>"
            for i in range(round(args.max_sec / args.resolution) + 1)
        ],
    ]

    specials = [
        "<na>",  # text is not available
        *category_tokens,
        *task_tokens,
        *timestamp_tokens,
    ]

    with open(args.output_path / "nlsyms.txt", "w") as fp:
        fp.write("\n".join(specials))
