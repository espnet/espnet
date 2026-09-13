"""Export portable BEATs checkpoints from ESPnet3 training runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from espnet2.beats.generate_beats_checkpoint import convert_checkpoint

logger = logging.getLogger(__name__)

# Monitors tried in order: loss first, because BEATs accuracy is noisy and
# tokenizer training reports no accuracy.
_MONITOR_SUFFIXES = ("valid.loss", "valid.acc")


def resolve_best_checkpoints(exp_dir: str | Path) -> List[Path]:
    """Resolve the top-K checkpoints of an ESPnet3 experiment directory.

    ESPnet3 keeps the ``best_model_criterion`` top-K models as
    ``epoch<E>_step<S>_<monitor>.ckpt`` (weights only) and deletes the others,
    so the files on disk are exactly the current top-K. They are used instead
    of ``<monitor>.ave_<K>best.pth``, which is written before Lightning updates
    the top-K and therefore misses the last validation.

    Args:
        exp_dir: Training ``exp_dir`` of an encoder or tokenizer run.

    Returns:
        List[Path]: Checkpoints of the first monitor in
        ``valid/loss``, ``valid/acc`` that has any, sorted by name.

    Raises:
        FileNotFoundError: If no top-K checkpoint exists, typically because
            training has not validated yet or ``best_model_criterion`` monitors
            neither ``valid/loss`` nor ``valid/acc``.
    """
    exp_root = Path(exp_dir)
    for suffix in _MONITOR_SUFFIXES:
        checkpoints = sorted(exp_root.glob(f"epoch*_step*_{suffix}.ckpt"))
        if checkpoints:
            return checkpoints
    raise FileNotFoundError(
        f"No top-K checkpoint under {exp_root}: expected "
        "epoch<E>_step<S>_valid.loss.ckpt or epoch<E>_step<S>_valid.acc.ckpt."
    )


def export_beats_checkpoint(
    exp_dir: str | Path,
    output_path: str | Path,
    checkpoint_paths: List[str | Path] | None = None,
) -> Path:
    """Average top-K checkpoints into a portable BEATs checkpoint.

    Used at the end of the SSL ``train`` and ``train_tokenizer`` stages. The
    output has the ``{"model": state_dict, "cfg": config}`` layout expected by
    ``BeatsEncoder(beats_ckpt_path=...)`` (tokenizer teacher, downstream
    fine-tuning) and ``BeatsTokenizer(beats_tokenizer_ckpt_path=...)``
    (tokenization). Only ``encoder.*`` weights are kept, cast to float32.

    Args:
        exp_dir: Training ``exp_dir``. Must contain ``config.yaml`` written by
            the training stage (``save_espnet_config``).
        output_path: Destination ``.pt`` file. Overwritten if it exists.
        checkpoint_paths: Lightning checkpoints to average. When ``None``,
            :func:`resolve_best_checkpoints` picks them.

    Returns:
        Path: ``output_path``.

    Raises:
        FileNotFoundError: If ``config.yaml`` or the checkpoints are missing.
    """
    exp_root = Path(exp_dir)
    config_path = exp_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    checkpoints = [
        Path(path)
        for path in (
            checkpoint_paths
            if checkpoint_paths is not None
            else resolve_best_checkpoints(exp_root)
        )
    ]
    output = Path(output_path)
    logger.info(
        "Exporting BEATs checkpoint from %d checkpoint(s) -> %s: %s",
        len(checkpoints),
        output,
        [str(path) for path in checkpoints],
    )
    convert_checkpoint(
        espnet_model_checkpoint_paths=[str(path) for path in checkpoints],
        espnet_model_config_path=str(config_path),
        output_path=str(output),
        deepspeed_checkpoint=False,
        lightning_checkpoint=True,
    )
    if len(checkpoints) > 1:
        # convert_checkpoint names averaged outputs `<K>avg.<name>`.
        output.with_name(f"{len(checkpoints)}avg.{output.name}").replace(output)
    return output
