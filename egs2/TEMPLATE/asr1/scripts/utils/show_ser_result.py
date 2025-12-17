import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Format SER results in Markdown")
    parser.add_argument(
        "--metrics", type=str, required=True, help="Path to metrics.txt"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., inference_xxx/test)",
    )
    return parser.parse_args()


def show_ser_results(metrics_path, dataset_name):
    with open(metrics_path, "r") as f:
        lines = f.readlines()

    # Header
    print("|---|---|---|---|---|---|")
    print("| dataset | Label | Prec | Recall | F1 | Support |")

    # Parse lines
    in_report = False
    for line in lines:
        line = line.strip()
        if line.startswith("Detailed classification report"):
            in_report = True
            continue
        if not in_report:
            continue
        if line == "":
            continue

        # Match report line: label  precision  recall  f1-score support
        match = re.match(r"^(\S+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)$", line)
        if match:
            label, prec, recall, f1, support = match.groups()
            print(
                f"| {dataset_name} | {label} | {prec} | {recall} | {f1} | {support} |"
            )


if __name__ == "__main__":
    args = parse_args()
    show_ser_results(args.metrics, args.dataset)
