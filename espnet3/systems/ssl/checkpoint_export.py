"""Export portable BEATs checkpoints from ESPnet3 training runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from espnet2.beats.generate_beats_checkpoint import convert_checkpoint

logger = logging.getLogger(__name__)

# Monitors tried in order: loss first, because BEATs accuracy is noisy and
# tokenizer training reports no accuracy.
_MONITORS = ("valid/loss", "valid/acc")


def resolve_best_checkpoints(trainer: Any) -> List[Path]:
    """Resolve the top-K checkpoints kept by a finished training run.

    The checkpoints come from the run's own ``ModelCheckpoint`` callbacks
    (``best_k_models``), not from a directory listing: a run restarted in the
    same ``exp_dir`` without resuming leaves the previous run's checkpoints in
    place, and those must never be averaged into this run's export. They are
    used instead of ``<monitor>.ave_<K>best.pth``, which is written before
    Lightning updates the top-K and therefore misses the last validation.

    Args:
        trainer: Trainer returned by :func:`espnet3.systems.base.training.train`,
            or a ``lightning.Trainer``.

    Returns:
        List[Path]: Checkpoints of the first monitor in ``valid/loss``,
        ``valid/acc`` that kept any, sorted by name.

    Raises:
        FileNotFoundError: If no monitored checkpoint was kept, typically
            because training has not validated yet or ``best_model_criterion``
            monitors neither ``valid/loss`` nor ``valid/acc``.
    """
    lightning_trainer = getattr(trainer, "trainer", trainer)
    kept = {
        callback.monitor: callback.best_k_models
        for callback in getattr(lightning_trainer, "checkpoint_callbacks", [])
        if getattr(callback, "monitor", None) and callback.best_k_models
    }
    for monitor in _MONITORS:
        if monitor in kept:
            return sorted(Path(path) for path in kept[monitor])
    raise FileNotFoundError(
        "This training run kept no checkpoint for "
        f"{' or '.join(_MONITORS)}; monitored checkpoints: {sorted(kept)}."
    )


def export_beats_checkpoint(
    exp_dir: str | Path,
    output_path: str | Path,
    trainer: Any = None,
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
        trainer: Trainer that just finished this run, used to resolve the
            checkpoints to average. Required unless ``checkpoint_paths`` is
            given.
        checkpoint_paths: Explicit Lightning checkpoints to average, bypassing
            :func:`resolve_best_checkpoints`.

    Returns:
        Path: ``output_path``.

    Raises:
        FileNotFoundError: If ``config.yaml`` is missing or the run kept no
            monitored checkpoint.
        ValueError: If neither ``trainer`` nor ``checkpoint_paths`` is given.
    """
    if trainer is None and checkpoint_paths is None:
        raise ValueError("export_beats_checkpoint needs trainer or checkpoint_paths.")
    exp_root = Path(exp_dir)
    config_path = exp_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    checkpoints = [
        Path(path)
        for path in (
            checkpoint_paths
            if checkpoint_paths is not None
            else resolve_best_checkpoints(trainer)
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
