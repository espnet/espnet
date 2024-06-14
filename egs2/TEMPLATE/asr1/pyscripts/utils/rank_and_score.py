#!/usr/bin/env python3

import argparse
import editdistance

from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse --metric and --score_dir options")
    parser.add_argument('--metric', type=str, required=True, help='The metric to be used')
    parser.add_argument('--score_dir', type=Path, required=True, help='The directory for scores')

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    input_dir = args.score_dir
    output_dir = input_dir / "rank_and_score"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    if metric == "wer":
        ref_file = "ref.trn"
        hyp_file = "hyp.trn"
    else:
        raise NotImplementedError
    
    stats_dict = {}

    # (1) parse reference
    for line in open(input_dir / ref_file):
        example_name, content = parse_trn(line)
        example_name = example_name.split("_sample")[0]

        if example_name not in stats_dict:
            stats_dict[example_name] = {"ref": content}
        else:
            assert stats_dict[example_name]["ref"] == content
    
    # (2) parse hypotheses
    for line in open(input_dir / hyp_file):
        example_name, content = parse_trn(line)
        example_name, example_id = example_name.split("_sample")

        if example_name not in stats_dict:
            raise ValueError(f"example {example_name} is not in reference file")
        
        if "hyp" not in stats_dict[example_name]:
            stats_dict[example_name]["hyp"] = []
        
        stats_dict[example_name]["hyp"].append([example_id, content])
    
    nbest = len(stats_dict[example_name]["hyp"])
    assert [len(stats_dict[example_name]["hyp"]) == nbest for example_name in stats_dict.keys()]

    # (3) compare and add the score
    for example_name in stats_dict.keys():
        ref = stats_dict[example_name]["ref"]
        hyps = stats_dict[example_name]["hyp"]

        for idx, hyp in enumerate(hyps):
            score = pair_scoring(ref, hyp[1], metric)
            stats_dict[example_name]["hyp"][idx].append(score)
        
        stats_dict[example_name]["hyp"].sort(key=lambda x: x[2])
    
    # (4) collect stats
    writer = open(output_dir / "rank_scoring_result", 'w')
    for rank in range(nbest):
        num, den = 0, 0
        for example_name in stats_dict.keys():
            score = stats_dict[example_name]["hyp"][rank][2]

            if metric == "wer":
                num += score
                den += len(stats_dict[example_name]["ref"].split())
        
        tot_score = num /  den
        writer.write(f"Rank {rank} | Metric: {metric} | total_score: {tot_score}\n")
        

def pair_scoring(ref, hyp, metric):
    # the number of errors
    if metric == "wer":
        return wer_scoring(ref, hyp)
    
def wer_scoring(ref, hyp):
    return editdistance.eval(ref.split(), hyp.split())

def parse_trn(line):
    elems = line.strip().split()
    content = " ".join(elems[:-1])
    example_name = elems[-1].lstrip("(").rstrip(")")

    return example_name, content


if __name__ == "__main__":
    main()