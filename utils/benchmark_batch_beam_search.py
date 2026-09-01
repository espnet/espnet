#!/usr/bin/env python3
"""Measure the speedup of decoding several utterances in one beam search.

This is a synthetic benchmark: it builds a randomly initialized attention
decoder plus CTC and decodes random encoder outputs, so the hypotheses are
meaningless but the amount of work per decoding step is representative. It
reports the wall time of
:class:`espnet2.legacy.nets.batch_beam_search.BatchBeamSearch` given one
utterance at a time and given batches of increasing size, and checks that
they produce the same best hypothesis for every utterance.

Example::

    python utils/benchmark_batch_beam_search.py --device cuda --varied
"""

import argparse
import time

import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--vocab", type=int, default=5000, help="Vocabulary size")
    parser.add_argument(
        "--dim", type=int, default=512, help="Encoder output / decoder size"
    )
    parser.add_argument("--layers", type=int, default=6, help="Decoder blocks")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--units", type=int, default=2048, help="Feed-forward units")
    parser.add_argument("--beam", type=int, default=10, help="Beam size")
    parser.add_argument("--nutt", type=int, default=16, help="Utterances to decode")
    parser.add_argument(
        "--frames", type=int, default=300, help="Encoder output length of the longest"
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Output tokens to generate per utterance"
    )
    parser.add_argument("--ctc", type=float, default=0.3, help="CTC decoding weight")
    parser.add_argument(
        "--varied",
        action="store_true",
        help="Sweep the encoder lengths from --frames down to half of it, "
        "instead of giving every utterance the same length",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch.set_num_threads (0 to leave as is)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser


def main(cmd=None):
    """Run the benchmark."""
    args = get_parser().parse_args(cmd)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.device == "cuda":
        synchronize = torch.cuda.synchronize
    elif args.device == "mps":
        synchronize = torch.mps.synchronize
    else:

        def synchronize():
            return None

    decoder = (
        TransformerDecoder(
            vocab_size=args.vocab,
            encoder_output_size=args.dim,
            num_blocks=args.layers,
            attention_heads=args.heads,
            linear_units=args.units,
        )
        .to(device)
        .eval()
    )
    # NOTE: `BeamSearch.__init__` drops any scorer whose weight is 0, so the
    # length bonus needs a non-zero weight to be part of what is measured.
    scorers = {"decoder": decoder, "length_bonus": LengthBonus(args.vocab)}
    weights = {"decoder": 1.0 - args.ctc, "length_bonus": 0.1}
    if args.ctc > 0:
        ctc = CTC(odim=args.vocab, encoder_output_size=args.dim).to(device).eval()
        scorers["ctc"] = CTCPrefixScorer(ctc=ctc, eos=args.vocab - 1)
        weights["ctc"] = args.ctc

    common = dict(
        vocab_size=args.vocab,
        weights=weights,
        scorers=scorers,
        sos=args.vocab - 1,
        eos=args.vocab - 1,
        beam_size=args.beam,
        pre_beam_score_key="full",
    )
    beam_search = BatchBeamSearch(**common).to(device).eval()

    if args.varied:
        step = (args.frames // 2) / max(1, args.nutt - 1)
        lengths = [args.frames - int(step * i) for i in range(args.nutt)]
    else:
        lengths = [args.frames] * args.nutt
    encs = [torch.randn(n, args.dim, device=device) for n in lengths]
    # a constant max output length keeps the number of decoding steps the same
    # for both implementations, which is the least favourable case for batching
    maxlenratio = -args.steps

    def pad(subset):
        max_len = max(e.size(0) for e in subset)
        out = torch.zeros(len(subset), max_len, args.dim, device=device)
        for i, e in enumerate(subset):
            out[i, : e.size(0)] = e
        lens = torch.tensor([e.size(0) for e in subset], device=device)
        return out, lens

    def run_reference():
        return [
            beam_search(x=e, maxlenratio=maxlenratio, minlenratio=0.0) for e in encs
        ]

    def run_batched(batch_size):
        out = []
        for i in range(0, len(encs), batch_size):
            x, x_lengths = pad(encs[i : i + batch_size])
            out += beam_search(
                x=x, x_lengths=x_lengths, maxlenratio=maxlenratio, minlenratio=0.0
            )
        return out

    def timeit(fn):
        with torch.no_grad():
            fn()  # warm-up
            synchronize()
            start = time.perf_counter()
            out = fn()
            synchronize()
        return time.perf_counter() - start, out

    print(
        f"device={args.device} vocab={args.vocab} dim={args.dim} "
        f"layers={args.layers} beam={args.beam} nutt={args.nutt} "
        f"frames={lengths[0]}..{lengths[-1]} out_steps={args.steps} ctc={args.ctc}"
    )
    base, ref_out = timeit(run_reference)
    print(
        f"  one utterance at a time : {base:7.2f} s "
        f"{base / args.nutt * 1000:8.1f} ms/utt  {1.0:5.2f}x"
    )
    batch_size = 1
    while batch_size <= args.nutt:
        elapsed, out = timeit(lambda bs=batch_size: run_batched(bs))
        match = sum(
            r[0].yseq.tolist() == n[0].yseq.tolist() for r, n in zip(ref_out, out)
        )
        print(
            f"  batched, batch={batch_size:3d}      : {elapsed:7.2f} s "
            f"{elapsed / args.nutt * 1000:8.1f} ms/utt  {base / elapsed:5.2f}x  "
            f"best-hyp match {match}/{args.nutt}"
        )
        batch_size *= 2


if __name__ == "__main__":
    main()
