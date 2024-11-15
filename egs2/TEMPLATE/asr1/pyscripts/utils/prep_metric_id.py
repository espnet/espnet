#!/usr/bin/env python3

import argparse
import logging

from espnet2.fileio.metric_scp import MetricReader
from espnet2.fileio.read_text import read_2columns_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare metric ID")
    parser.add_argument("metric_scp", type=str, help="metric.scp information")
    parser.add_argument("metric2id", type=str, help="output metric2id")
    parser.add_argument("--metric2type", type=str, default=None, help="metric type")
    parser.add_argument("--reading_size", type=int, default=-1, help="reading size (for efficient loading)")
    args = parser.parse_args()

    if args.metric2type is not None:
        metric2type = dict(read_2columns_text(args.metric2type))
        with open(args.metric2id, "w") as f:
            for iter, (key, value) in enumerate(metric2type.items()):
                f.write(f"{key}\n")
    else:
        metric_reader = MetricReader(args.metric_scp)
        reading_size = args.reading_size if args.reading_size > 0 else len(metric_reader)
        metric2id = set()
        id = 0
        with open(args.metric2id, "w") as f:
            row_num = 0
            for key, metric in metric_reader.items():
                for k, v in metric.items():
                    if k in metric2id:
                        continue
                    metric2id.add(k)
                    f.write(f"{k}\n")
                    id += 1
                row_num += 1
                if row_num > reading_size:
                    print(f"Reading size reached {reading_size}, stop reading.")
        
