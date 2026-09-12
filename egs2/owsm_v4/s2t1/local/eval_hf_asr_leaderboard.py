#!/usr/bin/env python3
"""Leaderboard-style WER and RTFx of an OWSM checkpoint on a sampled test set.

Reads a data directory written by local/prepare_hf_asr_leaderboard.py, decodes
it with espnet2.bin.s2t_inference.Speech2Text and scores it the way the Open
ASR Leaderboard does: references and hypotheses go through the Whisper English
text normalizer, WER is the corpus-level word error rate from jiwer, and RTFx
is total audio duration divided by total decoding time. Only the decode calls
are timed; model loading and one untimed warm-up batch are excluded.

With --batch_size > 1 the utterances are decoded with Speech2Text.batch_decode
(one beam search over the whole minibatch); --sort orders them by duration
first so that a minibatch holds utterances of similar length.

Example:
    python local/eval_hf_asr_leaderboard.py --data data/lb_librispeech_test_clean \
        --model_tag espnet/owsm_v4_base_102M --batch_size 8 --sort \
        --out exp/hf_asr_leaderboard/test_clean_base_bs8.json
"""

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import soundfile as sf
import torch

from espnet2.text.cleaner import TextCleaner
from espnet2.utils.types import str2bool


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", required=True, help="data dir with wav.scp, text and utt2dur"
    )
    parser.add_argument("--model_tag", default="espnet/owsm_v4_base_102M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--ctc_weight", type=float, default=0.0)
    parser.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="0 (default): the maximum output length is the number of encoder "
        "frames and the beam search runs its end detection; 1.0: same maximum "
        "length without end detection",
    )
    parser.add_argument(
        "--sort",
        type=str2bool,
        default=False,
        help="decode the utterances in order of duration",
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="torch intra-op threads, 0 = default"
    )
    parser.add_argument(
        "--quantize",
        type=str2bool,
        default=False,
        help="dynamic int8 quantization of nn.Linear (Speech2Text option)",
    )
    parser.add_argument(
        "--n", type=int, default=0, help="use only the first n utterances, 0 = all"
    )
    parser.add_argument("--tag", default="", help="free-text label for the summary")
    parser.add_argument(
        "--out", required=True, help="summary json; per-utterance jsonl goes beside it"
    )
    return parser


def read_kaldi_file(path: Path) -> dict:
    """Read a Kaldi-style "key value" text file into a dict."""
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            key, value = line.split(maxsplit=1)
            result[key] = value
    return result


def main() -> None:
    """Decode the data directory and write the summary."""
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    from jiwer import wer as jiwer_wer

    from espnet2.bin.s2t_inference import Speech2Text

    normalizer = TextCleaner(["whisper_en"])
    if normalizer.whisper_cleaner is None:
        raise RuntimeError(
            "the Whisper English text normalizer is needed; pip install openai-whisper"
        )

    data = Path(args.data)
    wav_scp = read_kaldi_file(data / "wav.scp")
    texts = read_kaldi_file(data / "text")
    durations = {k: float(v) for k, v in read_kaldi_file(data / "utt2dur").items()}
    utt_ids = list(wav_scp)
    if args.n > 0:
        utt_ids = utt_ids[: args.n]
    if args.sort:
        utt_ids = sorted(utt_ids, key=lambda u: durations[u])
    wavs = [sf.read(wav_scp[u], dtype="float32")[0] for u in utt_ids]

    start = time.time()
    speech2text = Speech2Text.from_pretrained(
        model_tag=args.model_tag,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        ctc_weight=args.ctc_weight,
        maxlenratio=args.maxlenratio,
        quantize_s2t_model=args.quantize,
        lang_sym="<eng>",
        task_sym="<asr>",
    )
    load_s = time.time() - start
    n_params = sum(p.numel() for p in speech2text.s2t_model.parameters())

    def decode(batch):
        """Return the 1-best text (without special tokens) of each utterance."""
        if len(batch) == 1:
            return [speech2text(torch.from_numpy(batch[0]))[0][3]]
        speech = torch.zeros(len(batch), max(len(w) for w in batch))
        for i, wav in enumerate(batch):
            speech[i, : len(wav)] = torch.from_numpy(wav)
        lengths = torch.tensor([len(w) for w in batch])
        return [hyps[0][3] for hyps in speech2text.batch_decode(speech, lengths)]

    hyps, times = [], []
    with torch.inference_mode():
        decode(wavs[: args.batch_size])  # warm-up, not timed
        for i in range(0, len(wavs), args.batch_size):
            batch = wavs[i : i + args.batch_size]
            tic = time.perf_counter()
            hyps.extend(decode(batch))
            elapsed = time.perf_counter() - tic
            times.extend([elapsed / len(batch)] * len(batch))
            logging.info(f"{len(hyps)}/{len(wavs)} decoded, batch {elapsed:.1f}s")

    refs_norm = [normalizer(texts[u]) for u in utt_ids]
    hyps_norm = [normalizer(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]
    wer = jiwer_wer([r for r, _ in pairs], [h for _, h in pairs])
    audio_s = sum(durations[u] for u in utt_ids)
    decode_s = sum(times)

    summary = {
        "tag": args.tag,
        "model_tag": args.model_tag,
        "data": data.name,
        "n": len(utt_ids),
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "ctc_weight": args.ctc_weight,
        "maxlenratio": args.maxlenratio,
        "sort": args.sort,
        "quantize": args.quantize,
        "device": args.device,
        "dtype": args.dtype,
        "threads": torch.get_num_threads(),
        "wer": round(100 * wer, 2),
        "rtfx": round(audio_s / decode_s, 2),
        "audio_s": round(audio_s, 1),
        "decode_s": round(decode_s, 1),
        "sec_per_utt": round(decode_s / len(utt_ids), 3),
        "params_M": round(n_params / 1e6),
        "load_s": round(load_s, 1),
        "torch": torch.__version__,
        "machine": platform.machine(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    with out.with_suffix(".jsonl").open("w") as f:
        for u, hyp, ref_n, hyp_n, t in zip(utt_ids, hyps, refs_norm, hyps_norm, times):
            record = {
                "utt_id": u,
                "duration_s": durations[u],
                "time_s": round(t, 3),
                "ref": texts[u],
                "hyp": hyp,
                "ref_norm": ref_n,
                "hyp_norm": hyp_n,
            }
            f.write(json.dumps(record) + "\n")
    logging.info(json.dumps(summary))
    print(
        f"| {data.name} | {args.model_tag} | bs={args.batch_size} beam={args.beam_size}"
        f" maxlenratio={args.maxlenratio:g}{' sorted' if args.sort else ''}"
        f"{' int8' if args.quantize else ''} | {summary['wer']:.2f} | "
        f"{summary['rtfx']:.2f} | {summary['sec_per_utt']:.2f} |"
    )


if __name__ == "__main__":
    main()
