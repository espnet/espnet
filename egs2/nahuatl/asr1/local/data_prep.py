#!/usr/bin/env python3
"""Convert one HF dataset split to a Kaldi-format data directory.

Usage:
    python local/data_prep.py \
        --hf_data_dir /path/to/hf_data \
        --split hidalgo-train \
        --output_dir data/nahuatl_hidalgo_train \
        --wav_dir data/wav/nahuatl_hidalgo_train \
        --region_token "<nah_hid>" \
        [--max_examples 10]
"""
import argparse
import os
import re

from datasets import load_from_disk

_SPEAKER_PAT = re.compile(r'[A-Z]{2,4}\d{3,4}')


def _sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', s)


def _speaker(raw_id: str) -> str:
    m = _SPEAKER_PAT.search(raw_id)
    return m.group(0) if m else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_data_dir', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--wav_dir', required=True)
    parser.add_argument('--region_token', required=True)
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Truncate dataset for testing')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)

    ds = load_from_disk(args.hf_data_dir)[args.split]
    if args.max_examples is not None:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    rows: list[tuple[str, str, str, str]] = []
    for ex in ds:
        raw_id: str = ex['id']
        utt_id = _sanitize(raw_id)
        spk_id = _speaker(raw_id)
        text: str = ex['text']
        audio_bytes: bytes = ex['audio']['bytes']

        wav_path = os.path.abspath(os.path.join(args.wav_dir, f'{utt_id}.wav'))
        with open(wav_path, 'wb') as f:
            f.write(audio_bytes)

        rows.append((utt_id, spk_id, wav_path, text))

    rows.sort(key=lambda r: r[0])

    utt2spk: dict[str, str] = {}
    with (
        open(os.path.join(args.output_dir, 'wav.scp'), 'w') as wav_f,
        open(os.path.join(args.output_dir, 'text'), 'w') as text_f,
        open(os.path.join(args.output_dir, 'utt2spk'), 'w') as u2s_f,
    ):
        for utt_id, spk_id, wav_path, text in rows:
            wav_f.write(
                f"{utt_id} sox {wav_path} -r 16000 -c 1 -t wav - |\n"
            )
            text_f.write(
                f"{utt_id} {args.region_token}<asr><notimestamps> {text}\n"
            )
            u2s_f.write(f"{utt_id} {spk_id}\n")
            utt2spk[utt_id] = spk_id

    spk2utt: dict[str, list[str]] = {}
    for utt, spk in utt2spk.items():
        spk2utt.setdefault(spk, []).append(utt)
    with open(os.path.join(args.output_dir, 'spk2utt'), 'w') as f:
        for spk, utts in sorted(spk2utt.items()):
            f.write(f"{spk} {' '.join(sorted(utts))}\n")

    print(f"Wrote {len(rows)} utterances to {args.output_dir}")


if __name__ == '__main__':
    main()
