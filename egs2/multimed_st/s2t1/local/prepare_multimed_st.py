#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
from datasets import Audio, load_dataset

LANG_INFO = {
    "eng": ("English", "<eng>"),
    "vie": ("Vietnamese", "<vie>"),
    "fra": ("French", "<fra>"),
    "deu": ("German", "<deu>"),
    "zho": ("Chinese", "<zho>"),
}

SPLIT_MAP = {
    "train": "train",
    "eval": "valid",
    "corrected.test": "test",
}


def clean_text(s: object) -> str:
    if s is None:
        return ""
    return " ".join(str(s).replace("\n", " ").split())


def safe_id(s: object, max_len: int = 80) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")
    return s[:max_len] or "utt"


def decode_and_write_wav(audio: Dict[str, object], wav_path: Path) -> Tuple[int, float]:
    audio_bytes = audio.get("bytes")
    if not audio_bytes:
        raise ValueError("Missing audio bytes")

    speech, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    # ESPnet S2T inference expects mono audio.
    # MultiMed-ST audio can be stereo, so downmix before writing wav.
    if getattr(speech, "ndim", 1) == 2:
        speech = speech.mean(axis=1)
    sf.write(wav_path, speech, sr)
    duration = float(len(speech) / sr)
    return int(sr), duration


def write_spk2utt(utt2spk_lines: List[str], out_path: Path) -> None:
    spk2utts = defaultdict(list)
    for line in utt2spk_lines:
        utt, spk = line.split(maxsplit=1)
        spk2utts[spk].append(utt)

    lines = [spk + " " + " ".join(utts) for spk, utts in sorted(spk2utts.items())]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_dataset(hf_config: str, hf_dataset: str, split: str):
    ds = load_dataset(
        hf_dataset,
        hf_config,
        split=split,
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    return iter(ds)


def prepare_split(
    hf_dataset: str,
    src_lang: str,
    tgt_lang: str,
    split: str,
    out_root: Path,
    task: str,
    max_samples: int,
) -> None:
    esp_split = SPLIT_MAP[split]
    data_dir = out_root / "data" / esp_split
    wav_dir = out_root / "wav" / esp_split

    data_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    hf_src, src_token = LANG_INFO[src_lang]
    hf_tgt, _ = LANG_INFO[tgt_lang]
    src_code = src_lang
    tgt_code = tgt_lang

    wav_scp: List[str] = []
    utt2spk: List[str] = []
    text_lines: List[str] = []
    text_asr_lines: List[str] = []
    text_st_lines: List[str] = []
    text_prev_lines: List[str] = []
    text_ctc_lines: List[str] = []

    n_ok = 0

    for idx, ex in enumerate(iter_dataset(hf_src, hf_dataset, split)):
        if max_samples > 0 and n_ok >= max_samples:
            break

        audio = ex.get("audio")
        if not isinstance(audio, dict) or not audio.get("bytes"):
            continue

        src_text = clean_text(ex.get("text", ""))
        tgt_text = clean_text(ex.get(hf_tgt, ""))

        if not src_text:
            continue
        if task in {"st", "multitask_asr_st"} and not tgt_text:
            continue

        audio_path = audio.get("path", f"{idx}.ogg")
        utt_base = safe_id(Path(str(audio_path)).stem)

        examples: List[Tuple[str, str, str, str]] = []

        if task in {"asr", "multitask_asr_st"}:
            utt_id = f"{src_code}_asr_{esp_split}_{n_ok:08d}_{utt_base}"
            text = f"{src_token}<asr><notimestamps> {src_text}"
            examples.append((utt_id, text, src_text, "asr"))

        if task in {"st", "multitask_asr_st"}:
            utt_id = f"{src_code}_{tgt_code}_st_{esp_split}_{n_ok:08d}_{utt_base}"
            text = f"{src_token}<st_{tgt_code}><notimestamps> {tgt_text}"
            examples.append((utt_id, text, src_text, "st"))

        wav_path = wav_dir / f"{src_code}_{esp_split}_{n_ok:08d}_{utt_base}.wav"

        try:
            sr, dur = decode_and_write_wav(audio, wav_path)
        except Exception as e:
            print(f"skip idx={idx}: audio decode failed: {e}")
            continue

        for utt_id, text, ctc_text, mode in examples:
            wav_scp.append(f"{utt_id} {wav_path}")
            # Use utterance-id as speaker-id because MultiMed-ST metadata does not
            # provide reliable speaker turns; this also satisfies Kaldi/ESPnet sorting.
            utt2spk.append(f"{utt_id} {utt_id}")
            text_lines.append(f"{utt_id} {text}")
            text_prev_lines.append(f"{utt_id} <na>")
            text_ctc_lines.append(f"{utt_id} {ctc_text}")

            if mode == "asr":
                text_asr_lines.append(f"{utt_id} {text}")
            elif mode == "st":
                text_st_lines.append(f"{utt_id} {text}")

        if n_ok < 5:
            print("=" * 80)
            print("split:", split, "esp_split:", esp_split, "idx:", idx)
            print("wav:", wav_path)
            print("sr:", sr, "duration:", f"{dur:.2f}")
            print("src:", src_text[:200])
            print("tgt:", tgt_text[:200])

        n_ok += 1

    (data_dir / "wav.scp").write_text(
        "\n".join(sorted(wav_scp)) + "\n", encoding="utf-8"
    )
    (data_dir / "utt2spk").write_text(
        "\n".join(sorted(utt2spk)) + "\n", encoding="utf-8"
    )
    (data_dir / "text").write_text(
        "\n".join(sorted(text_lines)) + "\n", encoding="utf-8"
    )
    (data_dir / "text.prev").write_text(
        "\n".join(sorted(text_prev_lines)) + "\n", encoding="utf-8"
    )
    (data_dir / "text.ctc").write_text(
        "\n".join(sorted(text_ctc_lines)) + "\n", encoding="utf-8"
    )
    write_spk2utt(sorted(utt2spk), data_dir / "spk2utt")

    if text_asr_lines:
        (data_dir / "text.asr").write_text(
            "\n".join(sorted(text_asr_lines)) + "\n", encoding="utf-8"
        )
    if text_st_lines:
        (data_dir / "text.st").write_text(
            "\n".join(sorted(text_st_lines)) + "\n", encoding="utf-8"
        )

    print(
        f"Wrote {data_dir} with {len(text_lines)} examples from {n_ok} original samples"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", default="leduckhai/MultiMed-ST")
    parser.add_argument("--src_lang", default="eng", choices=list(LANG_INFO))
    parser.add_argument("--tgt_lang", default="deu", choices=list(LANG_INFO))
    parser.add_argument(
        "--task", default="st", choices=["asr", "st", "multitask_asr_st"]
    )
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    args = parser.parse_args()

    if args.task in {"st", "multitask_asr_st"} and args.src_lang == args.tgt_lang:
        raise ValueError("For ST, src_lang and tgt_lang must be different")

    out_root = Path(args.out_root)

    split_to_max = {
        "train": args.max_train_samples,
        "eval": args.max_valid_samples,
        "corrected.test": args.max_test_samples,
    }

    for split, max_samples in split_to_max.items():
        prepare_split(
            hf_dataset=args.hf_dataset,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            split=split,
            out_root=out_root,
            task=args.task,
            max_samples=max_samples,
        )


if __name__ == "__main__":
    main()
