import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, required=True, help="Path to training config")
    parser.add_argument("--eval_config", type=str, required=True, help="Path to evaluation config")
    parser.add_argument("--train_tokenizer", action="store_true", help="Enable training tokenizer")
    parser.add_argument("--collect_stats", action="store_true", help="Enable collect_stats step")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode in evaluation")
    args = parser.parse_args()

    # Step 1: Train
    train_cmd = [
        "python", "train.py",
        "--config", args.train_config,
    ]
    if args.train_tokenizer:
        train_cmd.append("--train_tokenizer")
    if args.collect_stats:
        train_cmd.append("--collect_stats")

    print("==> Starting training...")
    subprocess.run(train_cmd, check=True)
    print("==> Training completed.")

    # Step 2: Evaluate
    eval_cmd = [
        "python", "evaluate.py",
        "--config", args.eval_config,
    ]
    if args.no_resume:
        eval_cmd.append("--no_resume")

    print("==> Starting evaluation...")
    subprocess.run(eval_cmd, check=True)
    print("==> Evaluation completed.")


if __name__ == "__main__":
    main()
