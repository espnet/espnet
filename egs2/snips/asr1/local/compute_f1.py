import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Macro-F1 for intent classification")
parser.add_argument("--hyp_trn", required=True, help="hyp.trn file path")
parser.add_argument("--ref_trn", required=True, help="ref.trn file path")
args = parser.parse_args()
print(args)


def compute_precision_recall_f1(count_metrics):
    tp = count_metrics["TP"]
    fp = count_metrics["FP"]
    fn = count_metrics["FN"]
    precision = 0.0 if tp == 0 else float(tp) / float(tp + fp)
    recall = 0.0 if tp == 0 else float(tp) / float(tp + fn)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


if __name__ == "__main__":
    metrics = defaultdict()
    with open(args.hyp_trn, "r") as hyp, open(args.ref_trn, "r") as ref:
        for line_hyp, line_ref in zip(hyp, ref):
            line_hyp, line_ref = line_hyp.split(), line_ref.split()
            assert line_hyp[-1] == line_ref[-1]
            predicted_intent, actual_intent = line_hyp[0], line_ref[0]
            if predicted_intent not in metrics:
                metrics[predicted_intent] = {"TP": 0, "FP": 0, "FN": 0}
            if actual_intent not in metrics:
                metrics[actual_intent] = {"TP": 0, "FP": 0, "FN": 0}
            if predicted_intent == actual_intent:
                metrics[predicted_intent]["TP"] += 1
            else:
                metrics[predicted_intent]["FP"] += 1
                metrics[actual_intent]["FN"] += 1
    f1_lists = [compute_precision_recall_f1(val_dict) for val_dict in metrics.values()]
    macro_f1 = sum(f1_lists) / len(f1_lists)
    print(f"The Macro-F1 score: {macro_f1}")
