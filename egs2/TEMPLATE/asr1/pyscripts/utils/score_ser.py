import argparse

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    recall_score,
)


def load_labels(file_path):
    labels = {}
    with open(file_path) as f:
        for line in f:
            key, label = line.strip().split()
            labels[key] = int(label)
    return labels


def main(args):
    ref = load_labels(args.ref)
    hyp = load_labels(args.hyp)

    # Match keys
    keys = sorted(set(ref.keys()) & set(hyp.keys()))
    y_true = [ref[k] for k in keys]
    y_pred = [hyp[k] for k in keys]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    uar = recall_score(y_true, y_pred, average="macro")

    print(f"WA (Accuracy): {acc:.4f}")
    print(f"UAR: {uar:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True, help="Path to reference file")
    parser.add_argument(
        "--hyp", type=str, required=True, help="Path to hypothesis file"
    )
    args = parser.parse_args()
    main(args)
