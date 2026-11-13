#!/usr/bin/env python3
"""Find the speech of transcribed recordings by CTC forced alignment.

    python3 local/v2/align.py --dump dump/v2/librispeech/train/shard0 \
        --model_tag espnet/owsm_ctc_v4_1B --revision <commit> --device cuda

For a dump directory of label source "A" (vad_dump.py: wav.scp, text,
utt2lang), each recording is aligned word by word to its transcript with
OWSM-CTC and CTC segmentation (espnet2.bin.s2t_align). The speech is the
union of the word spans, with pauses shorter than --min_pause bridged. A
recording is dropped (and listed in align_dropped.tsv) if its language token
is not in the model, the transcript has no tokens, or a word scores below
--min_word_score. The result replaces espnet_rttm and the speech column of
manifest.tsv; align.tsv keeps every word span and score, align_config.json
the model (tag and commit) and the settings, and .aligned marks the
directory as done. The model is fetched at the given commit, so the labels
are reproducible.
"""

import argparse
import contextlib
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vad_dump import MANIFEST_FIELDS, merge_intervals  # noqa: E402


def model_files(tag: str, revision: str):
    """(config.yaml, model .pth) of `tag` at `revision` from the Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(tag, revision=revision))
    configs = sorted(root.glob("exp/*/config.yaml"))
    models = sorted(root.glob("exp/*/*.pth"))
    if len(configs) != 1 or len(models) != 1:
        raise RuntimeError(f"expected one config and one model in {root}")
    return root, configs[0], models[0]


def speech_of_words(words, min_pause: float):
    """Union of the word spans; pauses shorter than min_pause are bridged."""
    spans = merge_intervals((s, e) for s, e in words)
    out = []
    for s, e in spans:
        if out and s - out[-1][1] < min_pause:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def read_table(path: Path):
    return dict(
        line.rstrip("\n").split(maxsplit=1) for line in path.open() if line.strip()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--model_tag", default="espnet/owsm_ctc_v4_1B")
    parser.add_argument("--revision", required=True, help="model commit (pinned)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min_pause",
        type=float,
        default=0.2,
        help="pauses between words shorter than this (s) count as speech",
    )
    parser.add_argument(
        "--min_word_score",
        type=float,
        default=-10.0,
        help="drop a recording if a word's CTC segmentation score is below this",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    import torch

    from espnet2.bin.s2t_align import CTCSegmentation
    from espnet2.fileio.omniio_audio import OmniIOAudioScpReader

    d = args.dump
    names = ("wav.scp", "utt2num_samples", "utt2category", "espnet_rttm")
    names += ("manifest.tsv", "text", "utt2lang")
    backup = d / "before_align"
    if (d / ".aligned").exists():
        logging.info("%s is already aligned", d)
        return
    if backup.exists():  # an earlier run stopped while rewriting: start from its inputs
        for name in names:
            shutil.copy(backup / name, d / name)
    root, config, model = model_files(args.model_tag, args.revision)
    torch.manual_seed(0)
    # the model's config names its statistics, BPE model and token list relative
    # to the model directory
    with contextlib.chdir(root):
        aligner = CTCSegmentation(
            s2t_train_config=config,
            s2t_model_file=model,
            device=args.device,
            kaldi_style_text=False,
            time_stamps="auto",
        )
    tokens = set(aligner.token_list)
    audio = OmniIOAudioScpReader(d / "wav.scp")
    text = read_table(d / "text")
    lang = read_table(d / "utt2lang")
    num_samples = {k: int(v) for k, v in read_table(d / "utt2num_samples").items()}

    speech, dropped, word_rows, rates = {}, [], [], {}
    for i, rid in enumerate(sorted(text)):
        words = text[rid].split()
        if lang[rid] not in tokens:
            dropped.append((rid, f"language {lang[rid]} not in the model"))
            continue
        rate, x = audio[rid]
        rates[rid] = rate
        aligner.lang_sym = lang[rid]
        try:
            task = aligner(np.asarray(x, dtype=np.float32), words, fs=rate)
        except Exception as e:  # noqa: BLE001 - e.g. a transcript longer than the audio
            dropped.append((rid, f"alignment failed: {type(e).__name__}: {e}"))
            continue
        spans = [(float(s), float(e), float(c)) for s, e, c in task.segments]
        worst = min(c for _, _, c in spans)
        for w, (s, e, c) in zip(words, spans):
            word_rows.append(f"{rid}\t{w}\t{s:.3f}\t{e:.3f}\t{c:.4f}\n")
        if worst < args.min_word_score:
            dropped.append((rid, f"word score {worst:.2f} < {args.min_word_score}"))
            continue
        speech[rid] = speech_of_words([(s, e) for s, e, _ in spans], args.min_pause)
        if i % 1000 == 0:
            logging.info("%d / %d aligned", i, len(text))

    kept = sorted(speech)
    rttm = []
    speech_seconds = {}
    for rid in kept:
        n, fs = num_samples[rid], rates[rid]
        total = 0
        for s, e in speech[rid]:
            s, e = max(0, int(round(s * fs))), min(n, int(round(e * fs)))
            if e > s:
                rttm.append(f"SPEAKER {rid} 1 {s} {e} <NA> <NA> {rid} <NA>\n")
                total += e - s
        rttm.append(f"END {rid} <NA> <NA> {n} <NA> <NA> <NA> <NA>\n")
        speech_seconds[rid] = total / fs

    # keep the unaligned inputs, then rewrite the files without the dropped ones
    backup = d / "before_align"
    backup.mkdir(exist_ok=True)
    for name in (
        "wav.scp",
        "utt2num_samples",
        "utt2category",
        "espnet_rttm",
        "manifest.tsv",
    ):
        shutil.copy(d / name, backup / name)
    keep = set(kept)
    for name in ("wav.scp", "utt2num_samples", "utt2category", "text", "utt2lang"):
        lines = (d / name).read_text().splitlines(keepends=True)
        (d / name).write_text(
            "".join(x for x in lines if x.split(maxsplit=1)[0] in keep)
        )
    (d / "espnet_rttm").write_text("".join(sorted(rttm)))
    with (backup / "manifest.tsv").open() as f, (d / "manifest.tsv").open("w") as g:
        cols = f.readline().rstrip("\n").split("\t")
        g.write("\t".join(MANIFEST_FIELDS) + "\n")
        for line in f:
            row = dict(zip(cols, line.rstrip("\n").split("\t")))
            if row["rec_id"] in keep:
                row["speech_seconds"] = f"{speech_seconds[row['rec_id']]:.3f}"
                g.write("\t".join(row.get(k, "") for k in MANIFEST_FIELDS) + "\n")
    (d / "align.tsv").write_text(
        "rec_id\tword\tstart\tend\tscore\n" + "".join(word_rows)
    )
    (d / "align_dropped.tsv").write_text("".join(f"{r}\t{why}\n" for r, why in dropped))
    import espnet

    config_out = dict(
        model_tag=args.model_tag,
        revision=args.revision,
        model_dir=str(root),
        min_pause=args.min_pause,
        min_word_score=args.min_word_score,
        espnet_version=espnet.__version__,
        aligned=len(kept),
        dropped=len(dropped),
    )
    (d / "align_config.json").write_text(json.dumps(config_out, indent=1) + "\n")
    (d / ".aligned").write_text("")
    logging.info("aligned %d, dropped %d: %s", len(kept), len(dropped), d)


if __name__ == "__main__":
    main()
