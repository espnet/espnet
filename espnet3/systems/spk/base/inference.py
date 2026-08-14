"""Speaker inference entrypoints for ESPnet3."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from espnet2.bin.spk_inference import (  # Reuse mature espnet2 speaker inference logic
    Speech2Embedding,
    get_parser as _get_parser,
    inference as _inference,
    main as _main,
)


def get_inference_parser():
    """Return the speaker-inference CLI parser."""
    return _get_parser()


def infer(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: int | str,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
):
    """Run speaker-embedding inference.

    Args:
        output_dir: Output directory for generated embeddings and SCP files.
        batch_size: Inference batch size.
        dtype: Floating-point dtype used during inference.
        ngpu: Number of GPUs to use.
        seed: Random seed.
        num_workers: Data loader worker count.
        log_level: Logging verbosity level.
        data_path_and_name_and_type: Input data triplets.
        key_file: Optional key file limiting processed utterances.
        train_config: Optional training config path.
        model_file: Optional model parameter path.
        model_tag: Optional pretrained model tag.

    Returns:
        Return value from ``espnet2.bin.spk_inference.inference``.
    """
    return _inference(
        output_dir=output_dir,
        batch_size=batch_size,
        dtype=dtype,
        ngpu=ngpu,
        seed=seed,
        num_workers=num_workers,
        log_level=log_level,
        data_path_and_name_and_type=data_path_and_name_and_type,
        key_file=key_file,
        train_config=train_config,
        model_file=model_file,
        model_tag=model_tag,
    )


def main_inference(cmd: Optional[Sequence[str]] = None) -> Any:
    """CLI-compatible speaker inference entrypoint."""
    return _main(cmd=cmd)
