import argparse
import subprocess
import shlex


def run_command(cmd, dry_run=False):
    print(f">>> {shlex.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="train.yaml")
    parser.add_argument("--eval_config", type=str, default="evaluate.yaml")
    parser.add_argument("--stage", type=str,
                        choices=["create_dataset", "train", 
                                 "evaluate", "all"],
                        default="all")

    # Common options
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print commands without executing")

    # Create dataset options
    parser.add_argument("--input_dir", type=str, help="Directory of raw input data")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save processed data")

    # Train options
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--collect_stats", action="store_true")
    parser.add_argument("--train_args", type=str, default="")

    # Evaluate options
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--eval_args", type=str, default="")

    args = parser.parse_args()

    # Stage: create dataset
    if args.stage in ("create_dataset", "all"):
        if not args.input_dir or not args.output_dir:
            raise ValueError("--input_dir and --output_dir must be specified for"
                             "stage=create_dataset or all")
        create_cmd = [
            "python", "dataset/create_dataset.py",
            "--input_dir", args.input_dir,
            "--output_dir", args.output_dir,
        ]
        print("==> Starting dataset creation...")
        run_command(create_cmd, dry_run=args.dry_run)
        print("==> Dataset creation completed.")

    # Stage: train
    if args.stage in ("train", "all"):
        train_cmd = [
            "python", "train.py",
            "--config", args.train_config,
        ]
        if args.train_tokenizer:
            train_cmd.append("--train_tokenizer")
        if args.collect_stats:
            train_cmd.append("--collect_stats")
        train_cmd += shlex.split(args.train_args)
        print("==> Starting training...")
        run_command(train_cmd, dry_run=args.dry_run)
        print("==> Training completed.")

    # Stage: evaluate
    if args.stage in ("evaluate", "all"):
        eval_cmd = [
            "python", "evaluate.py",
            "--config", args.eval_config,
        ]
        if args.no_resume:
            eval_cmd.append("--no_resume")
        eval_cmd += shlex.split(args.eval_args)
        print("==> Starting evaluation...")
        run_command(eval_cmd, dry_run=args.dry_run)
        print("==> Evaluation completed.")


if __name__ == "__main__":
    main()
