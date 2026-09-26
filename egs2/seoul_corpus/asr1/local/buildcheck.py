#!/usr/bin/env python3
"""Build an ASR model from a config and run one forward pass.

Catches the things that only surface once the frontend is real -- its output
dimension against the preencoder's input_size, and whether the encoder's
subsampling leaves more frames than target tokens -- without waiting for a GPU.
"""

import sys

import torch

from espnet2.tasks.asr import ASRTask


def main():
    cfg = sys.argv[1]
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
        "exp_tags2/.buildcheck",
        "--frontend_conf",
        "fs=16k",
    ]
    args = ASRTask.get_parser().parse_args(argv)
    model = ASRTask.build_model(args)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  params: {total / 1e6:.1f} M total, {train / 1e6:.1f} M trainable, "
        f"{(total - train) / 1e6:.1f} M frozen"
    )
    print(f"  frontend output_size: {model.frontend.output_size()}")

    model.eval()
    with torch.no_grad():
        # 5 s and 3 s, the shape a real batch has after sorting by length
        speech = torch.randn(2, 80000)
        speech_lengths = torch.tensor([80000, 48000])
        text = torch.tensor([[5, 100, 200, 8, 300], [5, 100, 200, -1, -1]])
        text_lengths = torch.tensor([5, 3])
        enc, enc_lens = model.encode(speech, speech_lengths)
        print(
            f"  encoder out: {tuple(enc.shape)} lengths={enc_lens.tolist()} "
            f"({enc.shape[1] / (80000 / 16000) :.1f} frames/s)"
        )
        loss, stats, _ = model(speech, speech_lengths, text, text_lengths)
    print(
        f"  forward OK: loss={float(loss):.2f} "
        f"loss_ctc={stats.get('loss_ctc')} loss_att={stats.get('loss_att')}"
    )


if __name__ == "__main__":
    main()
