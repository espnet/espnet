import argparse

from sklearn.metrics import classification_report


def main(args):
    txt_path = f"{args.dir}/lid.trn"
    with open(txt_path, "r", encoding="utf-8") as f:
        correct, total = 0, 0
        y_true, y_pred = [], []
        for line in f:
            if line == "\n":
                continue
            [gt, pred, name] = line.strip().split("\t")
            y_true.append(gt)
            y_pred.append(pred)
            if pred == gt:
                correct += 1
            total += 1

        with open(f"{args.dir}/scores.txt", "w") as f:
            f.write(f"Acc: {correct / total * 100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()
    main(args)
