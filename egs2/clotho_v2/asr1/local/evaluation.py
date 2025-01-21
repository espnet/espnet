"""Wrapper around aac_metrics (https://github.com/Labbeti/aac-metrics)
to evaluate the audio captioning predictions."""

import argparse
from string import punctuation

from aac_metrics import Evaluate

strip_punct_table = str.maketrans("", "", punctuation)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--decode_file",
    required=True,
    help="Path to file containing predictions.",
)
parser.add_argument(
    "--split",
    default="evaluation",
    choices=["evaluation", "validation"],
    help="Split. Could be either evaluation or validation. Default: evaluation",
)

args = parser.parse_args()


def _parse_and_read_file(file_path: str):
    read_data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                uttid, prediction = line.strip().split(maxsplit=1)
                read_data[uttid] = (
                    prediction.strip().lower().translate(strip_punct_table)
                )
    except Exception as exc:
        raise ValueError(f"Error reading file {file_path}") from exc
    if len(read_data) == 0:
        raise ValueError(f"File {file_path} is empty")
    return read_data


def _convert_to_evaluation_format(predictions, references):
    candidates = []
    mult_references = []
    for key in predictions.keys():
        candidates.append(predictions[key])
        if key not in references:
            raise ValueError(f"Key {key} not found in references")
        mult_references.append(references[key])
    return candidates, mult_references


def _evaluate(decode_fname: str, split: str):
    # load all predictions from decode_file
    predictions = _parse_and_read_file(decode_fname)

    # modify ref file pattern based on split
    ref_file_pattern = "data/{split}/text{suffix}"
    ref_file_paths = []
    if split == "evaluation":
        ref_file_paths = [
            ref_file_pattern.format(split="evaluation", suffix=f"_spk{i}")
            for i in range(1, 6)
        ]
    elif split == "validation":
        ref_file_paths = [ref_file_pattern.format(split="validation", suffix="")]
    else:
        raise ValueError(f"Invalid split: {split}")

    # load all references
    references = {}
    for ref_file_path in ref_file_paths:
        reference_ = _parse_and_read_file(ref_file_path)
        # add reference_ values to reference based on keys
        for key, value in reference_.items():
            if key in references:
                references[key].append(value)
            else:
                references[key] = [value]

    candidates, mult_references = _convert_to_evaluation_format(predictions, references)

    evaluate = Evaluate(
        metrics=[
            "spider",
            "fense",
            "meteor",
            "rouge_l",
            "spice",
            "spider_fl",
            "cider_d",
        ]
    )
    corpus_scores, _ = evaluate(candidates, mult_references)

    header_str = f" Split: {split} Evaluation over {len(candidates)} predictions. "
    header_str_l = len(header_str)
    print("=" * header_str_l)
    print(header_str)
    print("=" * header_str_l)
    for metric, value in corpus_scores.items():
        print(f" {metric:<20}: {value} ")
    print("=" * header_str_l)

    # Write results in a file
    results_file = decode_fname + ".result"
    with open(results_file, "w") as result_f:
        header_str = f" Split: {split} Evaluation over {len(candidates)} predictions. "
        header_str_l = len(header_str)
        print("=" * header_str_l, file=result_f)
        print(header_str, file=result_f)
        print("=" * header_str_l, file=result_f)
        for metric, value in corpus_scores.items():
            print(f" {metric:<20}: {value} ", file=result_f)
        print("=" * header_str_l, file=result_f)


_evaluate(args.decode_file, args.split)
