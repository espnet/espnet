import argparse
import torch
from pathlib import Path


def convert_checkpoint(checkpoint_path, output_dir, metric_name):
    """Convert a single Lightning checkpoint to the averaged format.

    Args:
        checkpoint_path: Path to the checkpoint
        output_dir: Directory where the converted checkpoint will be saved
        metric_name: Name of the metric to use in the output filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]

    # Remove extra prefix in model keys
    new_state_dict = {
        k.removeprefix("model."): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }

    # If no keys start with "model.", keep the original state_dict
    if not new_state_dict:
        print("Warning: No keys with 'model.' prefix found. Using original state dict.")
        new_state_dict = state_dict

    avg_ckpt_path = output_dir / f"{metric_name.replace('/', '.')}.ave_1best.pth"
    print(f"Saving converted checkpoint to {avg_ckpt_path}")
    torch.save(new_state_dict, avg_ckpt_path)
    return avg_ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a single checkpoint to averaged format"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where converted checkpoint will be saved",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="val_loss",
        help="Metric name used for the output filename",
    )

    args = parser.parse_args()

    avg_path = convert_checkpoint(
        args.checkpoint_path, args.output_dir, args.metric_name
    )
    print(f"Successfully created converted checkpoint at {avg_path}")


if __name__ == "__main__":
    main()
