#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np


def main():
    if args.log is not None:
        with open(args.log) as f:
            logs = json.load(f)
        val_scores = []
        for log in logs:
            if log["epoch"] > args.max_epoch:
                continue

            if args.metric == "acc":
                if "validation/main/acc" in log.keys():
                    val_scores += [[log["epoch"], log["validation/main/acc"]]]
            elif args.metric == "perplexity":
                if "val_perplexity" in log.keys():
                    val_scores += [[log["epoch"], 1 / log["val_perplexity"]]]
            elif args.metric == "loss":
                if "validation/main/loss" in log.keys():
                    val_scores += [[log["epoch"], -log["validation/main/loss"]]]
            elif args.metric == "bleu":
                if "validation/main/bleu" in log.keys():
                    val_scores += [[log["epoch"], log["validation/main/bleu"]]]
            elif args.metric == "cer":
                if "validation/main/cer" in log.keys():
                    val_scores += [[log["epoch"], -log["validation/main/cer"]]]
            elif args.metric == "cer_ctc":
                if "validation/main/cer_ctc" in log.keys():
                    val_scores += [[log["epoch"], -log["validation/main/cer_ctc"]]]
            else:
                # Keep original order for compatibility
                if "validation/main/acc" in log.keys():
                    val_scores += [[log["epoch"], log["validation/main/acc"]]]
                elif "val_perplexity" in log.keys():
                    val_scores += [[log["epoch"], 1 / log["val_perplexity"]]]
                elif "validation/main/loss" in log.keys():
                    val_scores += [[log["epoch"], -log["validation/main/loss"]]]

        if len(val_scores) == 0:
            raise ValueError("%s is not found in log." % args.metric)
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores[:, -1])
        sorted_val_scores = val_scores[sort_idx][::-1]
        print("metric: %s" % args.metric)
        print("best val scores = " + str(sorted_val_scores[: args.num, 1]))
        print(
            "selected epochs = "
            + str(sorted_val_scores[: args.num, 0].astype(np.int64))
        )
        last = [
            os.path.dirname(args.snapshots[0]) + "/snapshot.ep.%d" % (int(epoch))
            for epoch in sorted_val_scores[: args.num, 0]
        ]
    else:
        last = sorted(args.snapshots, key=os.path.getmtime)
        last = last[-args.num :]
    print("average over", last)
    avg = None

    if args.backend == "pytorch":
        import torch

        # sum
        for path in last:
            states = torch.load(path, map_location=torch.device("cpu"))["model"]
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    avg[k] += states[k]

        # average
        for k in avg.keys():
            if avg[k] is not None:
                if avg[k].is_floating_point():
                    avg[k] /= args.num
                else:
                    avg[k] //= args.num

        torch.save(avg, args.out)

    elif args.backend == "chainer":
        # sum
        for path in last:
            states = np.load(path)
            if avg is None:
                keys = [x.split("main/")[1] for x in states if "model" in x]
                avg = dict()
                for k in keys:
                    avg[k] = states["updater/model:main/{}".format(k)]
            else:
                for k in keys:
                    avg[k] += states["updater/model:main/{}".format(k)]
        # average
        for k in keys:
            if avg[k] is not None:
                avg[k] /= args.num
        np.savez_compressed(args.out, **avg)
        os.rename("{}.npz".format(args.out), args.out)  # numpy save with .npz extension
    else:
        raise ValueError("Incorrect type of backend")


def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--backend", default="chainer", type=str)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    parser.add_argument(
        "--metric",
        default="",
        type=str,
        nargs="?",
        choices=["acc", "bleu", "cer", "cer_ctc", "loss", "perplexity"],
    )
    parser.add_argument(
        "--max-epoch",
        default=10000000,
        type=int,
        nargs="?",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
