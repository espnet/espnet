#!/usr/bin/env python3
"""Time one training step of an ASR config, split into its parts.

An SSL epoch here ran past 38 minutes where the log-mel config takes 95 s, and
the trainer's own counters had not been written yet, so this measures the pieces
directly: the data pipeline on CPU, then the frontend, the rest of the model,
and the backward pass on GPU.  conf/train_asr.yaml is measured as a control --
its epoch time is known, so if the numbers reproduce it they can be trusted for
the SSL configs too.
"""

import argparse
import time

import numpy as np
import torch

from espnet2.tasks.asr import ASRTask

N_BATCH = 564  # batches per epoch, from the trainer's sampler


def build(cfg):
    argv = [
        "--config",
        cfg,
        "--token_list",
        "data/ko_token_list/bpe_unigram2000/tokens.txt",
        "--token_type",
        "bpe",
        "--bpemodel",
        "data/ko_token_list/bpe_unigram2000/bpe.model",
        "--use_preprocessor",
        "true",
        "--non_linguistic_symbols",
        "data/nlsyms.txt",
        "--cleaner",
        "none",
        "--g2p",
        "none",
        "--output_dir",
        "exp_tags2/.bench",
        "--frontend_conf",
        "fs=16k",
    ]
    args = ASRTask.get_parser().parse_args(argv)
    return ASRTask.build_model(args), args


def timed(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--utts", type=int, default=13)
    ap.add_argument("--batch_bins", type=int, default=4000000)
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()

    print(f"########## {a.config}")
    print(f"  torch {torch.__version__}  threads={torch.get_num_threads()}")
    model, args = build(a.config)
    model = model.cuda().train()
    torch.cuda.reset_peak_memory_stats()

    # the sampler packs batch_bins padded samples into ~utts utterances
    per = a.batch_bins // a.utts
    speech = torch.randn(a.utts, per, device="cuda")
    slens = torch.full((a.utts,), per, dtype=torch.long, device="cuda")
    tl = 58  # mean target length on this data
    text = torch.randint(3, 1999, (a.utts, tl), device="cuda")
    tlens = torch.full((a.utts,), tl, dtype=torch.long, device="cuda")
    print(
        f"  batch: {a.utts} x {per / 16000:.1f}s = {a.utts * per / 16000:.0f}s audio, "
        f"targets {tl} tokens"
    )

    scaler = torch.amp.GradScaler("cuda")

    def frontend_only():
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model.frontend(speech, slens)

    def fwd():
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model(speech, slens, text, tlens)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = model(speech, slens, text, tlens)[0]
        scaler.scale(loss).backward()

    for name, fn in (
        ("frontend fwd", frontend_only),
        ("model fwd", fwd),
        ("model fwd+bwd", fwd_bwd),
    ):
        try:
            s = timed(fn, a.iters)
            print(
                f"  {name:16} {s * 1000:8.1f} ms/step"
                f" -> epoch {s * N_BATCH / 60:6.1f} min"
            )
        except RuntimeError as e:
            print(f"  {name:16} FAILED: {str(e)[:120]}")

    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # CPU side: the preprocessor is what the dataloader workers run per utterance
    pre = ASRTask.build_preprocess_fn(args, train=True)
    if pre is not None:
        wav = np.random.randn(int(13.1 * 16000)).astype(np.float32)
        t0 = time.perf_counter()
        for i in range(a.utts):
            pre(f"u{i}", {"speech": wav, "text": "네 그렇습니다"})
        dt = time.perf_counter() - t0
        print(
            f"  preprocess       {dt * 1000:8.1f} ms/batch ({a.utts} utts) "
            f"->  epoch {dt * N_BATCH / 60:6.1f} min on one worker"
        )
    else:
        print("  preprocess       (none)")


if __name__ == "__main__":
    main()
