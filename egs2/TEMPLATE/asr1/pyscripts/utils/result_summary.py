#!/usr/bin/env python3

import argparse
import json
import logging

import matplotlib.pyplot as plt
from pathlib import Path

from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse --metric and --score_dir options"
    )

    parser.add_argument(
        "--all_eval_results",
        type=Path,
        nargs="+",
        help="utterance-level result files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="the directory for output results",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="metrics to analyze",
    )
    parser.add_argument(
        "--nbest",
        type=int,
        default=1,
        help="the expected number of samples for each utterance",
    )
    parser.add_argument(
        "--cross_rerank",
        type=str2bool,
        default=False,
        help="If true, rerank by each metric and evaluate by the others",
    )
    parser.add_argument(
        "--draw_picture",
        type=str2bool,
        default=True,
        help="If true, draw picture for this rerank",
    )

    args = parser.parse_args()
    return args


def main(args):

    # (1) parse all results
    stats_dict = {}
    for score_file in args.all_eval_results:
        for line in open(score_file):
            line = line.strip().replace("'", '"')
            line = json.loads(line)
            example_name = line.pop("key")
            utt_name = example_name.split("_sample")[0]
            
            if "weight" in line:
                weight = line["weight"]
                line.pop("weight")
            else:
                weight = 1.0

            # dict arch: utt_name -> example_name -> metric -> (value, weight)
            if utt_name not in stats_dict:
                stats_dict[utt_name] = {}
            for key, value in line.items():
                if example_name not in stats_dict[utt_name]:
                    stats_dict[utt_name][example_name] = {}
                stats_dict[utt_name][example_name][key] = (value, weight)

    # (2) validate all utterances:
    for utt_name in stats_dict.keys():
        other_sample = next(iter(stats_dict[utt_name].values()))

        # (2.1) add dummy examples until reach `nbest` examples
        count = 0
        while len(stats_dict[utt_name]) < args.nbest:
            example_name = utt_name + f"_pad{count}"
            stats_dict[utt_name][example_name] = {}
            for metric in args.metrics:
                weight = other_sample[metric][1]
                worst_value = worse_result(metric)
                stats_dict[utt_name][example_name][metric] = (worst_value, weight)
            count += 1
            logging.info(f"add dummy example {example_name}")

        # (2.2) add dummy metric for each example
        for example_name in stats_dict[utt_name].keys():
            for metric in args.metrics:
                if metric not in stats_dict[utt_name][example_name]:
                    weight = other_sample[metric][1]
                    worst_value = worse_result(metric)
                    stats_dict[utt_name][example_name][metric] = (worst_value, weight)
                    logging.info(f"add dummy metric {metric} to example {example_name}")

        # (2.3) Additionally, add the overall score as an additional metric
        add_overall_score(stats_dict[utt_name])

    # (3) rerank based on each metric
    if args.cross_rerank:
        metric_combos = [(a, b) for a in args.metrics for b in args.metrics]
    else:
        metric_combos = [(a, a) for a in args.metrics]

    metric_combos += [("overall_score", a) for a in args.metrics]

    for rerank_metric, report_metric in metric_combos:
        analyze_one_metric(
            stats_dict,
            rerank_metric,
            report_metric,
            args.nbest,
            args.draw_picture,
            args.output_dir,
        )

def analyze_one_metric(
    stats_dict,
    rerank_metric,
    report_metric,
    nbest,
    draw_picture,
    output_dir,
):
    weights = [0.0 for _ in range(nbest)]
    accum = [0.0 for _ in range(nbest)]

    for utt_dict in stats_dict.values():
        example_names = list(utt_dict.keys())
        example_names.sort(
            key=lambda x: utt_dict[x][rerank_metric][0],
            reverse=sort_reverse(rerank_metric),
        )

        for idx, example_name in enumerate(example_names):
            score, weight = utt_dict[example_name][report_metric]
            weights[idx] += weight
            accum[idx] += score

    print(f"Metric: {report_metric}, Rerank by {rerank_metric}")
    result_list = [a / c for a, c in zip(accum, weights)]
    for idx, v in enumerate(result_list):
        print(f"rank: {idx} | value: {v}")
    print(f"Average: {sum(accum) / sum(weights)}")

    if draw_picture:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        draw(
            lists=[result_list],
            path=output_dir / "images" / f"{report_metric}-{rerank_metric}.png",
            title=f"{report_metric}, reranked by {rerank_metric}",
            xlabel="rank",
            ylabel=report_metric,
        )


def draw(
    lists,
    path,
    title="",
    xlabel="",
    ylabel="",
):
    plt.clf()
    for i, data in enumerate(lists):
        plt.plot(data, label=f"Line {i+1}", marker="v")
        for j, value in enumerate(data):
            plt.text(value, j, f"{value:.2f}", fontsize=9, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)


def worse_result(metric):
    if metric == "spk_similarity":
        return 0.0

    elif metric == "utmos":
        return 0.0

    elif metric == "wer":
        return 1.0

    else:
        raise NotImplementedError(f"{metric}")


def sort_reverse(metric):
    # True: the higher the better
    # False: the lower the better
    if metric == "spk_similarity":
        return True

    elif metric == "utmos":
        return True

    elif metric == "wer":
        return False
    
    elif metric == "overall_score":
        return False

    else:
        raise NotImplementedError(f"{metric}")

def draw_line_chart(lists, title="", xlabel="", ylabel="", path=None):
    for i, data in enumerate(lists):
        plt.plot(data, label=f"Line {i+1}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if path:
        plt.savefig(path)


def add_overall_score(utt_dict):
    example_names = list(utt_dict.keys())
    metrics = list(utt_dict[example_names[0]].keys())

    scores = {name: 0.0 for name in example_names}

    for metric in metrics:
        name_values = [(name, utt_dict[name][metric]) for name in example_names]
        name_values.sort(key=lambda x: x[1], reverse=sort_reverse(metric))

        for score, (name, _) in enumerate(name_values):
            scores[name] += score
    
    for name, score in scores.items():
        utt_dict[name]["overall_score"] = (score, 1.0) # add weight

if __name__ == "__main__":
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    args = parse_arguments()
    main(args)
