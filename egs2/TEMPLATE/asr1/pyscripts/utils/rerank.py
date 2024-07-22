#!/usr/bin/env python3

import argparse
import json
import logging

import matplotlib.pyplot as plt
from pathlib import Path

from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse --metric and --score_dir options")
    
    parser.add_argument(
        '--all_eval_results', 
        type=Path,
        nargs="+",
        help='utterance-level result files',
    )
    parser.add_argument(
        '--output_dir', 
        type=Path,
        help='the directory for output results',
    )
    parser.add_argument(
        '--metrics', 
        type=str,
        nargs="+",
        help='metrics to analyze',
    )
    parser.add_argument(
        '--nbest', 
        type=int,
        default=1,
        help='the expected number of samples for each utterance',
    )
    parser.add_argument(
        '--cross_rerank', 
        type=str2bool,
        default=False,
        help='If true, rerank by each metric and evaluate by the others',
    )
    parser.add_argument(
        '--draw_picture', 
        type=str2bool,
        default=True,
        help='If true, draw picture for this rerank',
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
            
            # dict arch: utt_name -> example_name -> metric -> value
            if utt_name not in stats_dict:
                stats_dict[utt_name] = {}
            for key, value in line.items():
                if example_name not in stats_dict[utt_name]:
                    stats_dict[utt_name][example_name] = {}
                stats_dict[utt_name][example_name][key] = value
            
    # (2) validate all utterances:
    for utt_name in stats_dict.keys():
        # (2.1) add dummy examples until reach `nbest` examples
        count = 0
        while len(stats_dict[utt_name]) < args.nbest:
            example_name = utt_name + f"_pad{count}"
            stats_dict[utt_name][example_name] = {}
            for metric in args.metrics:
                if metric in ["word_count", "edit_distance"]:
                    word_count = stats_dict[utt_name][f"{utt_name}_sample0"]["word_count"]
                    stats_dict[utt_name][example_name][metric] = word_count
                else:
                    stats_dict[utt_name][example_name][metric] = worse_result(metric)
            count += 1
            logging.info(f"add dummy example {example_name}")
        
        # (2.2) add dummy metric for each example
        for example_name in stats_dict[utt_name].keys():
            for metric in args.metrics:
                if metric not in stats_dict[utt_name][example_name]:
                    if metric in ["word_count", "edit_distance"]:
                        word_count = stats_dict[utt_name][f"{utt_name}_sample0"]["word_count"]
                        stats_dict[utt_name][example_name][metric] = word_count
                    else:
                        stats_dict[utt_name][example_name][metric] = worse_result(metric)
                    logging.info(f"add dummy metric {metric} to example {example_name}")
    
    # (3) rerank based on each metric
    if args.cross_rerank:
        metric_combos = [(a, b) for a in args.metrics for b in args.metrics]
    else:
        metric_combos = [(a, a) for a in args.metrics]
    
    for rerank_metric, report_metric in metric_combos:
        if report_metric == "word_count" or rerank_metric == "word_count":
            continue
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
    count = [0.0 for _ in range(nbest)]
    accum = [0.0 for _ in range(nbest)]

    for utt_dict in stats_dict.values():
        example_names = list(utt_dict.keys())
        example_names.sort(
            key=lambda x: utt_dict[x][rerank_metric],
            reverse=sort_reverse(rerank_metric)
        )

        for idx, example_name in enumerate(example_names):
            if report_metric in ["spk_similarity", "utmos"]:
                count[idx] += 1
                accum[idx] += utt_dict[example_name][report_metric]
            elif report_metric in ['edit_distance']:
                # should always have word count
                count[idx] += utt_dict[example_name]['word_count']
                accum[idx] += utt_dict[example_name][report_metric]
    
    print(f"Metric: {report_metric}, Rerank by {rerank_metric}")
    result_list = [a / c for a, c in zip(accum, count)]
    for idx, v in enumerate(result_list):
        print(f"rank: {idx} | value: {v}")
    print(f"Average: {sum(accum) / sum(count)}")
    
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
        plt.plot(data, label=f'Line {i+1}', marker='v')
        for j, value in enumerate(data):
            plt.text(value, j, f'{value:.2f}', fontsize=9, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)

def worse_result(metric):
    if metric == "spk_similarity":
        return 0.0
    
    elif metric == "utmos":
        return 0.0
    
    elif metric == "edit_distance":
        return 10000 # dummy
    
    elif metric == "word_count":
        return 10000 # dummy
    
    else:
        raise NotImplementedError(f"{metric}")

def sort_reverse(metric):
    # True: the higher the better
    # False: the lower the better
    if metric == "spk_similarity":
        return True
    
    elif metric == "utmos":
        return True
    
    elif metric == "edit_distance":
        return False
    
    elif metric == "word_count":
        return False
    
    else:
        raise NotImplementedError

import matplotlib.pyplot as plt

def draw_line_chart(
    lists, 
    title="", 
    xlabel="", 
    ylabel="", 
    path=None
):
    for i, data in enumerate(lists):
        plt.plot(data, label=f'Line {i+1}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    if path:
        plt.savefig(path)


if __name__ == "__main__":
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    args = parse_arguments()
    main(args)