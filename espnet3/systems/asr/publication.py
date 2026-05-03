"""Publication helpers for the ASR system."""

from __future__ import annotations

from pathlib import Path


def _resolve_stats_pack_paths(system) -> list[Path]:
    stats_dir = getattr(system.training_config, "stats_dir", None)
    if not stats_dir:
        return []
    train_dir = Path(stats_dir) / "train"
    return sorted(path for path in train_dir.glob("*.npz") if path.is_file())


def get_pack_model_artifacts(system) -> dict:
    """Return ASR-specific pack-model artifacts.

    Collects the trained model checkpoint, espnet2 train config, tokenizer
    directory, and stats archives from the experiment directory.

    Args:
        system: ASRSystem instance with a populated ``training_config``.

    Returns:
        dict: Artifact dict with keys ``files``, ``yaml_files``,
            and ``copy_paths``.

    Examples:
        >>> artifacts = get_pack_model_artifacts(asr_system)
        >>> "asr_model_file" in artifacts["files"]
        True
    """
    files: dict[str, str] = {}
    yaml_files: dict[str, str] = {}
    copy_paths: list[Path] = []

    training_cfg = system.training_config
    exp_dir = Path(training_cfg.exp_dir)

    train_config = exp_dir / "config.yaml"
    model_file = exp_dir / "last.ckpt"
    if train_config.exists():
        yaml_files["asr_train_config"] = str(train_config)
    if model_file.exists():
        files["asr_model_file"] = str(model_file)

    tokenizer_cfg = getattr(training_cfg, "tokenizer", None)
    if tokenizer_cfg is not None:
        save_path = getattr(tokenizer_cfg, "save_path", None)
        if save_path:
            copy_paths.append(Path(save_path))

    copy_paths.extend(_resolve_stats_pack_paths(system))

    data_dir = getattr(training_cfg, "data_dir", None)
    if data_dir:
        data_tokenizer = Path(data_dir) / "tokenizer"
        if data_tokenizer.exists():
            copy_paths.append(data_tokenizer)

    return {"files": files, "yaml_files": yaml_files, "copy_paths": copy_paths}
