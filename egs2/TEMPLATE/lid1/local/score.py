import argparse
from collections import defaultdict

# score.py


def read_file(file_path):
    """
    Reads a file and returns a dictionary with key and lid.
    Each line in the file should be in the format: key lid
    """
    data = {} 
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                key, lid = parts
                data[key] = lid
            else:
                raise ValueError(f"Invalid line format: {line}")
    return data


def score(pred_file, target_file, results_file):
    """
    Calculates the accuracy by comparing the predicted and target lids.
    """
    pred_data = read_file(pred_file)
    target_data = read_file(target_file)

    if set(pred_data.keys()) != set(target_data.keys()):
        raise ValueError("Keys in pred and target files do not match.")

    total = len(pred_data)
    correct = 0
    incorrect_predictions = {}

    # Count error frequencies: target->predicted pairs
    error_count = defaultdict(int)

    langs = list(set(target_data.values()))
    lang_correct = {lang: 0 for lang in langs}
    lang_total = {lang: 0 for lang in langs}

    for key in pred_data:
        lang_total[target_data[key]] += 1
        if pred_data[key] == target_data[key]:
            correct += 1
            lang_correct[target_data[key]] += 1
        else:
            incorrect_predictions[key] = (pred_data[key], target_data[key])
            # Count the error pair
            error_pair = f"{target_data[key]}->{pred_data[key]}"
            error_count[error_pair] += 1

    accuracy = correct / total
    accuracy_per_lang = {lang: lang_correct[lang] / lang_total[lang] for lang in langs}
    macro_accuracy = sum(accuracy_per_lang.values()) / len(langs)

    # Write results to file
    with open(results_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Macro Accuracy: {macro_accuracy:.2%}\n")
        f.write("Accuracy per Language:\n")
        for lang, acc in accuracy_per_lang.items():
            f.write(f"{lang}: {acc:.2%}\n")

        # Add error frequency statistics
        if error_count:
            f.write("\nError Frequency (Target->Predicted):\n")
            # Sort by frequency (descending) and then by target-pred pair name
            sorted_errors = sorted(error_count.items(), key=lambda x: (-x[1], x[0]))
            for error_pair, count in sorted_errors:
                f.write(f"{error_pair}: {count}\n")

        f.write("\nDetailed Incorrect Predictions:\n")
        if incorrect_predictions:
            for key, (pred, target) in incorrect_predictions.items():
                f.write(f"Key: {key}, Target: {target}, Predicted: {pred}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate accuracy of predictions.")
    parser.add_argument("--pred_lids", required=True, help="Path to the predict lids.")
    parser.add_argument(
        "--target_lids", required=True, help="Path to the target (ground-truth) lids."
    )
    parser.add_argument("--results", required=True, help="Path to the results file.")
    args = parser.parse_args()

    score(args.pred_lids, args.target_lids, args.results)
