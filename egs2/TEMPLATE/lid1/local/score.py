import argparse
import logging
from collections import defaultdict

# score.py


def read_utt2lang_file(file_path):
    """
    Reads a file and returns a dictionary with key and lid.
    Each line in the file should be in the format: key lid
    """
    data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split()
                if len(parts) == 2:
                    key, lid = parts
                    data[key] = lid
                else:
                    logging.warning(
                        f"Line {line_num} in {file_path} has {len(parts)} parts, "
                        f"expected 2: {line}"
                    )
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")

    return data


def read_lang2utt_file(file_path):
    """
    Reads a file and returns a dictionary with key and lid.
    Each line in the file should be in the format: lid keys, key is uttid
    """
    data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    lid = parts[0]
                    keys = parts[1:]
                    data[lid] = keys
                else:
                    logging.warning(
                        f"Line {line_num} in {file_path} has {len(parts)} parts, "
                        f"expected >= 2: {line}"
                    )
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")

    return data


def score(pred_file, target_file, train_lang2utt, results_file):
    """
    Calculates the accuracy by comparing the predicted and target lids.
    """
    pred_data = read_utt2lang_file(pred_file)
    target_data = read_utt2lang_file(target_file)
    train_lang2utt_data = read_lang2utt_file(train_lang2utt)

    if set(pred_data.keys()) != set(target_data.keys()):
        raise ValueError("Keys in pred and target files do not match.")

    total = len(pred_data)
    correct = 0
    incorrect_predictions = {}

    # Count error frequencies: target->predicted pairs
    error_count = defaultdict(int)

    # Get lanugages the system trained on
    # all langs the system trained
    all_langs = list(set(train_lang2utt_data.keys()))
    # langs in the test set, target langs is sub or equal to all_langs
    target_langs = list(set(target_data.values()))
    lang_correct = {lang: 0 for lang in all_langs}
    lang_total = {lang: 0 for lang in all_langs}

    # For precision and recall calculations
    true_positives = {lang: 0 for lang in all_langs}
    false_positives = {lang: 0 for lang in all_langs}
    false_negatives = {lang: 0 for lang in all_langs}

    for key in pred_data:
        target_lang = target_data[key]
        pred_lang = pred_data[key]

        lang_total[target_lang] += 1

        if pred_lang == target_lang:
            correct += 1
            lang_correct[target_lang] += 1
            true_positives[target_lang] += 1
        else:
            incorrect_predictions[key] = (pred_lang, target_lang)
            # Count the error pair
            error_pair = f"{target_lang}->{pred_lang}"
            error_count[error_pair] += 1
            # False negative for target language
            false_negatives[target_lang] += 1
            # False positive for predicted language
            false_positives[pred_lang] += 1

    accuracy = correct / total
    accuracy_per_lang = {
        lang: lang_correct[lang] / lang_total[lang]
        for lang in target_langs
        if lang_total[lang] > 0
    }
    macro_accuracy = (
        sum(accuracy_per_lang.values()) / len(target_langs) if target_langs else 0.0
    )

    # Calculate precision and recall for each language
    precision_per_lang = {}
    recall_per_lang = {}
    f1_per_lang = {}

    # Calculate precision, recall, and F1 for each language in the test set
    for lang in target_langs:
        # Precision = TP / (TP + FP)
        if true_positives[lang] + false_positives[lang] > 0:
            precision_per_lang[lang] = true_positives[lang] / (
                true_positives[lang] + false_positives[lang]
            )
        else:
            precision_per_lang[lang] = 0.0

        # Recall = TP / (TP + FN)
        if true_positives[lang] + false_negatives[lang] > 0:
            recall_per_lang[lang] = true_positives[lang] / (
                true_positives[lang] + false_negatives[lang]
            )
        else:
            recall_per_lang[lang] = 0.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision_per_lang[lang] + recall_per_lang[lang] > 0:
            f1_per_lang[lang] = (
                2
                * (precision_per_lang[lang] * recall_per_lang[lang])
                / (precision_per_lang[lang] + recall_per_lang[lang])
            )
        else:
            f1_per_lang[lang] = 0.0

    # Calculate macro-averaged precision, recall, and F1 for test languages
    macro_precision = (
        sum(precision_per_lang.values()) / len(target_langs) if target_langs else 0.0
    )
    macro_recall = (
        sum(recall_per_lang.values()) / len(target_langs) if target_langs else 0.0
    )
    macro_f1 = sum(f1_per_lang.values()) / len(target_langs) if target_langs else 0.0

    # Calculate overall (micro-averaged) precision, recall, and F1
    # for all trained languages
    total_tp = sum(true_positives[lang] for lang in all_langs)
    total_fp = sum(false_positives[lang] for lang in all_langs)
    total_fn = sum(false_negatives[lang] for lang in all_langs)

    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
    else:
        overall_precision = 0.0

    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
    else:
        overall_recall = 0.0

    if overall_precision + overall_recall > 0:
        overall_f1 = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall)
        )
    else:
        overall_f1 = 0.0

    # Write results to file
    with open(results_file, "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.2%}\n")
        f.write(f"Overall Precision: {overall_precision:.2%}\n")
        f.write(f"Overall Recall: {overall_recall:.2%}\n")
        f.write(f"Overall F1: {overall_f1:.2%}\n")
        f.write(f"\nMacro Accuracy across Languages: {macro_accuracy:.2%}\n")
        f.write(f"Macro Precision across Languages: {macro_precision:.2%}\n")
        f.write(f"Macro Recall across Languages: {macro_recall:.2%}\n")
        f.write(f"Macro F1 across Languages: {macro_f1:.2%}\n")
        f.write("\nPer-Language Metrics:\n")
        for lang in target_langs:
            f.write(f"{lang}:\n")
            f.write(f"  Accuracy: {accuracy_per_lang[lang]:.2%}\n")
            f.write(f"  Precision: {precision_per_lang[lang]:.2%}\n")
            f.write(f"  Recall: {recall_per_lang[lang]:.2%}\n")
            f.write(f"  F1: {f1_per_lang[lang]:.2%}\n")

        # Add error frequency statistics
        if error_count:
            f.write("\nError Frequency (Target->Prediction):\n")
            # Sort by frequency (descending) and then by target-pred pair name
            sorted_errors = sorted(error_count.items(), key=lambda x: (-x[1], x[0]))
            for error_pair, count in sorted_errors:
                f.write(f"{error_pair}: {count}\n")

        f.write("\nDetailed Incorrect Predictions:\n")
        if incorrect_predictions:
            for key, (pred, target) in incorrect_predictions.items():
                f.write(f"Key: {key}, Target: {target}, Prediction: {pred}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate accuracy of predictions.")
    parser.add_argument("--pred_lids", required=True, help="Path to the predict lids.")
    parser.add_argument(
        "--target_lids", required=True, help="Path to the target (ground-truth) lids."
    )
    parser.add_argument(
        "--train_lang2utt", required=True, help="Path to the train lang2utt."
    )
    parser.add_argument("--results", required=True, help="Path to the results file.")
    args = parser.parse_args()

    score(args.pred_lids, args.target_lids, args.train_lang2utt, args.results)
