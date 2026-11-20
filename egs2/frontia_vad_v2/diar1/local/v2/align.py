#!/usr/bin/env python3
"""Find the speech of transcribed recordings by CTC forced alignment.

    python3 local/v2/align.py --dump dump/v2/librispeech/train/shard0 \
        --model_tag espnet/owsm_ctc_v4_1B --revision <commit> --device cuda

For a dump directory of label source "A" (vad_dump.py: wav.scp, text,
utt2lang), each recording is aligned word by word to its transcript with
OWSM-CTC and CTC segmentation (espnet2.bin.s2t_align). The speech is the
union of the word spans, with pauses shorter than --min_pause bridged. A
word spans either the segment that CTC segmentation gives it
(--word_timing segmentation: up to 0.5 s before its first token, to its last
token; consecutive words nearly abut) or the frames of its own tokens
(--word_timing tokens, ctc_batch.token_word_spans), or the token spans of
torchaudio's CTC forced alignment of the posteriors (--word_timing
forced_align, ctc_batch.forced_align_word_spans), widened by --word_pad on
each side. The word score that --min_word_score tests is CTC segmentation's
confidence (the lowest mean log posterior over windows of the segment, blank
frames included) for segmentation and tokens, and the mean log posterior of
the word's token frames for forced_align. A
transcript written all in capitals is lower-cased for the model (its BPE is
case-sensitive). A recording is dropped (and listed in align_dropped.tsv) if
its language token
is not in the model, the transcript has no tokens, or a word scores below
--min_word_score. The result replaces espnet_rttm and the speech column of
manifest.tsv; align.tsv keeps every word span and score, align_config.json
the model (tag and commit) and the settings, and .aligned marks the
directory as done. The model is fetched at the given commit, so the labels
are reproducible.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
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


def absolute_config(
    root: Path, config: Path, out: Path, use_sdpa: bool = False
) -> Path:
    """Write `config` with its model-relative paths made absolute, to `out`.

    The config names its feature statistics, BPE model and token list relative
    to the model directory; the tokenizer opens its BPE model only when it is
    first used, so the paths must be right whatever the working directory.
    """
    import yaml

    conf = yaml.safe_load(config.read_text())

    def fix(value):
        if isinstance(value, str) and not Path(value).is_absolute():
            if (root / value).exists():
                return str(root / value)
        return value

    for key in ("bpemodel", "token_list"):
        if key in conf:
            conf[key] = fix(conf[key])
    for section in ("normalize_conf", "frontend_conf"):
        if isinstance(conf.get(section), dict):
            conf[section] = {k: fix(v) for k, v in conf[section].items()}
    if use_sdpa:
        # the encoder's attention through torch's scaled_dot_product_attention
        conf.setdefault("encoder_conf", {})["use_sdpa"] = True
    out.write_text(yaml.safe_dump(conf, allow_unicode=True))
    return out


def model_words(words):
    """The words as the model's tokenizer expects them.

    The OWSM BPE is case-sensitive and learned on mixed-case text: a transcript
    written all in capitals (LibriSpeech, ...) would be split into letters and
    align poorly, so it is lower-cased. Other transcripts are kept as written.
    """
    joined = " ".join(words)
    if joined.upper() == joined and joined.lower() != joined:
        return [w.lower() for w in words]
    return words


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


def clip_speech(words, clip_seconds: float, min_pause, pad_start, pad_end):
    """Speech of one transcript clip from its word spans (seconds in the clip).

    The words are joined across pauses shorter than min_pause, each segment is
    widened by pad_start before and pad_end after, and nothing reaches past
    the clip's edges, so the silence around the clip stays non-speech.
    """
    out = []
    for s, e in speech_of_words(words, min_pause):
        s, e = max(0.0, s - pad_start), min(clip_seconds, e + pad_end)
        if e > s:
            out.append((s, e))
    return merge_intervals(out)


def pack_speech(clips, min_pause, pad_start, pad_end):
    """Speech of a recording from its clips, without merging across clips.

    clips: [(start s, end s, word spans in the clip)]; clip_speech of each clip
    at its offset.
    """
    out = []
    for start, end, words in clips:
        out += [
            (start + s, start + e)
            for s, e in clip_speech(words, end - start, min_pause, pad_start, pad_end)
        ]
    return sorted(out)


def read_clips(path: Path):
    """{rec_id: [(clip id, first sample, end sample, text)]} of a dump's clips."""
    out = {}
    for line in path.open():
        rid, cid, s, e, text = line.rstrip("\n").split("\t")
        out.setdefault(rid, []).append((cid, int(s), int(e), text))
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
        default=-15.0,
        help="drop a recording if a word's CTC segmentation score is below this",
    )
    parser.add_argument(
        "--word_timing",
        default="segmentation",
        choices=("segmentation", "tokens", "forced_align"),
        help="word spans from CTC segmentation's segments, from the frames of "
        "each word's tokens on its path (ctc_batch.token_word_spans), or from "
        "torchaudio's CTC forced alignment (ctc_batch.forced_align_word_spans)",
    )
    parser.add_argument(
        "--word_pad",
        type=float,
        default=0.0,
        help="seconds added on each side of every word span",
    )
    parser.add_argument(
        "--clip_mode",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=False,
        help="align and label every transcript clip (dump file clips) on its own: "
        "words joined within a clip only, --pad_start/--pad_end clipped to the "
        "clip's edges, a recording dropped when any of its clips fails",
    )
    parser.add_argument(
        "--pad_start",
        type=float,
        default=0.0,
        help="--clip_mode: seconds of speech added before every segment",
    )
    parser.add_argument(
        "--pad_end",
        type=float,
        default=0.0,
        help="--clip_mode: seconds of speech added after every segment",
    )
    parser.add_argument(
        "--batch_seconds",
        type=float,
        default=3840.0,
        help="audio encoded per GPU forward, in the model's 30 s buffers "
        "(local/v2/ctc_batch.py); 0: one recording at a time, as "
        "espnet2.bin.s2t_align.CTCSegmentation does it",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="processes for the CTC segmentation of the batched path",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=64,
        help="recordings read ahead by the batched path",
    )
    parser.add_argument(
        "--use_sdpa",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=False,
        help="encoder attention through scaled_dot_product_attention",
    )
    parser.add_argument(
        "--tf32",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=False,
        help="allow TF32 matrix multiplication on the GPU",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="encoder precision",
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
    if (d / "clips").exists() or (d / "before_align" / "clips").exists():
        names += ("clips",)
    backup = d / "before_align"
    if (d / ".aligned").exists():
        logging.info("%s is already aligned", d)
        return
    if backup.exists():  # an earlier run stopped while rewriting: start from its inputs
        for name in names:
            shutil.copy(backup / name, d / name)
    root, config, model = model_files(args.model_tag, args.revision)
    torch.manual_seed(0)
    config = absolute_config(root, config, d / "owsm_config.yaml", args.use_sdpa)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    aligner = CTCSegmentation(
        s2t_train_config=config,
        s2t_model_file=model,
        device=args.device,
        dtype=args.dtype,
        kaldi_style_text=False,
        time_stamps="auto",
    )
    tokens = set(aligner.token_list)
    audio = OmniIOAudioScpReader(d / "wav.scp")
    text = read_table(d / "text")
    lang = read_table(d / "utt2lang")
    num_samples = {k: int(v) for k, v in read_table(d / "utt2num_samples").items()}

    speech, dropped, word_rows = {}, [], []
    rates = {}
    todo = []
    for rid in sorted(text):
        if lang[rid] not in tokens:
            dropped.append((rid, f"language {lang[rid]} not in the model"))
        else:
            todo.append(rid)
    from ctc_batch import (
        TooFewFrames,
        forced_align_word_spans,
        token_word_spans,
        word_token_ids,
    )

    # units of alignment: (unit id, rec_id, first sample, end sample or None, words);
    # --clip_mode: every transcript clip of a recording (dump file "clips"),
    # else the whole recording with its whole transcript
    units = []
    if args.clip_mode:
        clips = read_clips(d / "clips")
        for rid in todo:
            for k, (_, s, e, t) in enumerate(clips[rid]):
                units.append((f"{rid}|{k}", rid, s, e, t.split()))
    else:
        units = [(rid, rid, 0, None, text[rid].split()) for rid in todo]
    cache = {}

    def read(rid, s, e):
        if rid not in cache:
            cache.clear()
            rate, x = audio[rid]
            rates[rid] = rate
            cache[rid] = (rate, x)
        rate, x = cache[rid]
        return rate, (x if e is None else x[s:e])

    use_fa = args.word_timing == "forced_align"
    if args.batch_seconds > 0:
        from ctc_batch import align_batched

        items = (
            (u, model_words(w), lang[rid], lambda r=rid, a=s, b=e: read(r, a, b))
            for u, rid, s, e, w in units
        )
        results = align_batched(
            aligner,
            items,
            args.batch_seconds,
            args.workers,
            args.prefetch,
            forced_align=use_fa,
        )
    else:
        results = {}
        for i, (u, rid, s, e, w) in enumerate(units):
            rate, x = read(rid, s, e)
            aligner.lang_sym = lang[rid]
            words = model_words(w)
            try:
                task = aligner(np.asarray(x, dtype=np.float32), words, fs=rate)
                segments = [tuple(map(float, s_)) for s_ in task.segments]
                spans = {
                    "tokens": token_word_spans(
                        task.timings,
                        task.utt_begin_indices,
                        task.config.index_duration_in_seconds,
                    )
                }
                if use_fa:
                    try:
                        spans["forced_align"] = forced_align_word_spans(
                            torch.from_numpy(task.lpz).to(args.device),
                            word_token_ids(task),
                            task.config.blank,
                            task.config.index_duration_in_seconds,
                        )
                    except TooFewFrames as e_:
                        spans["forced_align"] = f"TooFewFrames: {e_}"
                results[u] = (segments, spans, None)
            except Exception as e_:  # noqa: BLE001 - e.g. transcript longer than audio
                results[u] = (None, None, f"{type(e_).__name__}: {e_}")
            if i % 1000 == 0:
                logging.info("%d / %d aligned", i, len(units))
    t_write = time.perf_counter()
    too_few = 0
    by_rec = {}
    for unit in units:
        by_rec.setdefault(unit[1], []).append(unit)
    for rid in todo:
        failed, clip_words, rows = None, [], []
        for k_clip, (u, _, s, e, w) in enumerate(by_rec[rid]):
            segments, spans, error = results[u]
            if error is None and use_fa and isinstance(spans["forced_align"], str):
                error = spans["forced_align"]
                too_few += 1
            if error is not None:
                failed = (
                    f"alignment failed{' in ' + u if args.clip_mode else ''}: {error}"
                )
                break
            tokens = spans["tokens"]
            if args.word_timing == "segmentation":
                timed = segments  # (start, end, CTC segmentation confidence)
            elif args.word_timing == "tokens":
                timed = [(a, b, c) for (a, b), (_, _, c) in zip(tokens, segments)]
            else:  # (start, end, mean log posterior of the word's token frames)
                timed = spans["forced_align"]
            off = s / rates[rid]
            if not args.clip_mode:  # each word widened by --word_pad
                pad = args.word_pad
                timed = [(max(0.0, a - pad), b + pad, c) for a, b, c in timed]
            worst = min(c for _, _, c in timed)
            for word, (a, b, c), g, k in zip(w, timed, segments, tokens):
                rows.append(
                    f"{rid}\t{word}\t{off + a:.3f}\t{off + b:.3f}\t{c:.4f}"
                    f"\t{off + g[0]:.3f}\t{off + g[1]:.3f}"
                    f"\t{off + k[0]:.3f}\t{off + k[1]:.3f}\t{k_clip}\n"
                )
            if worst < args.min_word_score:
                failed = f"word score {worst:.2f} < {args.min_word_score}" + (
                    f" in {u}" if args.clip_mode else ""
                )
                break
            end = (num_samples[rid] if e is None else e) / rates[rid]
            clip_words.append((off, end, [(a, b) for a, b, _ in timed]))
        word_rows += rows
        if failed:
            dropped.append((rid, failed))
            continue
        if args.clip_mode:
            speech[rid] = pack_speech(
                clip_words, args.min_pause, args.pad_start, args.pad_end
            )
        else:
            speech[rid] = speech_of_words(clip_words[0][2], args.min_pause)

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
    # every input (also text and utt2lang, which are filtered below), so that a
    # rerun after an interruption starts from the inputs again
    backup.mkdir(exist_ok=True)
    for name in names:
        shutil.copy(d / name, backup / name)
    keep = set(kept)
    keyed = ("wav.scp", "utt2num_samples", "utt2category", "text", "utt2lang")
    for name in keyed + (("clips",) if "clips" in names else ()):
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
        "rec_id\tword\tstart\tend\tscore\tseg_start\tseg_end\ttok_start\ttok_end"
        "\tclip\n" + "".join(word_rows)
    )
    (d / "align_dropped.tsv").write_text(
        "".join(f"{r}\t{why}\n" for r, why in sorted(dropped))
    )
    import espnet

    config_out = dict(
        model_tag=args.model_tag,
        revision=args.revision,
        model_dir=str(root),
        min_pause=args.min_pause,
        min_word_score=args.min_word_score,
        word_timing=args.word_timing,
        clip_mode=args.clip_mode,
        pad_start=args.pad_start,
        pad_end=args.pad_end,
        forced_align_too_few_frames=too_few if use_fa else None,
        word_pad=args.word_pad,
        dtype=args.dtype,
        use_sdpa=args.use_sdpa,
        tf32=args.tf32,
        batch_seconds=args.batch_seconds,
        espnet_version=espnet.__version__,
        aligned=len(kept),
        dropped=len(dropped),
    )
    (d / "align_config.json").write_text(json.dumps(config_out, indent=1) + "\n")
    (d / ".aligned").write_text("")
    logging.info("writing the outputs: %.0f s", time.perf_counter() - t_write)
    logging.info("aligned %d, dropped %d: %s", len(kept), len(dropped), d)
    if use_fa:
        logging.info("too few frames for forced alignment: %d recordings", too_few)


if __name__ == "__main__":
    main()
