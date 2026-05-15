"""Helpers for preparing runner configs before stage execution.

This module contains runner-oriented config logic that sits above raw config
loading. The functions here are intended to be called by `run.py`-style entry
points after `training_config`, `inference_config`, and `metrics_config` have
been loaded, but before those configs are resolved and passed into a system.
"""

from __future__ import annotations

import logging
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

_TRAINING_CONTEXT_KEYS = (
    "exp_tag",
    "exp_dir",
)
_INFERENCE_METRICS_CONTEXT_KEYS = (
    "exp_tag",
    "exp_dir",
    "inference_dir",
)


def _is_missing_or_empty(value) -> bool:
    """Return whether a config value should be treated as absent.

    This helper normalizes the common "missing" checks used by the runner
    config propagation logic. The check is intentionally conservative: `None`
    and empty strings are considered absent, while other values are kept.

    Args:
        value: Config value to inspect.

    Returns:
        bool: `True` when the value should be treated as missing.

    Notes:
        This helper is intentionally private because it codifies runner-local
        behavior rather than a general-purpose OmegaConf rule.

    Examples:
        `_is_missing_or_empty(None)` returns `True`.
        `_is_missing_or_empty("train_debug")` returns `False`.
    """
    return value is None or (isinstance(value, str) and not value.strip())


def _has_exp_identity(config: DictConfig) -> bool:
    """Return whether a config can resolve experiment output paths on its own.

    Standalone inference or metrics configs are allowed only when they already
    carry enough experiment identity to derive output paths without relying on
    `training_config`. This helper currently treats either `exp_tag` or a
    concrete `exp_dir` as sufficient.

    Args:
        config (DictConfig): Inference or metrics config to inspect.

    Returns:
        bool: `True` if the config has a usable `exp_tag` or standalone
        `exp_dir`.

    Notes:
        `exp_dir` values that still contain `${exp_tag}` are not considered
        standalone because they still depend on another missing key.

    Examples:
        `OmegaConf.create({"exp_tag": "train_debug"})` returns `True`.
        `OmegaConf.create({"exp_dir": "${exp_tag}/foo"})` returns `False`.
    """
    exp_tag = config.get("exp_tag")
    if not _is_missing_or_empty(exp_tag):
        return True

    exp_dir = config.get("exp_dir")
    if not isinstance(exp_dir, str) or not exp_dir.strip():
        return False
    return "${exp_tag}" not in exp_dir and "None" not in exp_dir


def _copy_config_context(
    *,
    source: DictConfig,
    target: DictConfig,
    keys: Sequence[str],
    source_name: str,
    target_name: str,
    log: logging.Logger,
) -> None:
    """Copy selected runner context keys from one config into another.

    The runner uses this helper to keep runtime configs aligned when one stage
    determines output locations consumed by a later stage. Missing keys are
    inserted with an info log, while conflicting existing values are
    overwritten with a warning so the operator can see which config took
    precedence.

    Args:
        source (DictConfig): Source config that provides authoritative values.
        target (DictConfig): Destination config to mutate in place.
        keys (Sequence[str]): Config keys to copy from source to target.
        source_name (str): Human-readable source name for logging.
        target_name (str): Human-readable destination name for logging.
        log (logging.Logger): Logger used for insert/overwrite messages.

    Returns:
        None: The target config is updated in place.

    Raises:
        This function does not raise exceptions.

    Notes:
        This helper intentionally stays generic so the runner can reuse the
        same logging and overwrite behavior for both training-derived
        experiment identity and inference-derived output locations.

    Examples:
        When `target.exp_tag` is missing, the helper logs an `INFO` insert.
        When `target.exp_tag` differs, the helper logs a `WARNING` overwrite.
    """
    for key in keys:
        if key not in source:
            continue

        source_value = source.get(key)
        if _is_missing_or_empty(source_value):
            continue

        current_value = target.get(key) if key in target else None
        if key not in target or _is_missing_or_empty(current_value):
            target[key] = source_value
            log.info("Inserted %s.%s from %s", target_name, key, source_name)
            continue

        if current_value != source_value:
            log.warning(
                "Overriding %s.%s with %s value: %r -> %r",
                target_name,
                key,
                source_name,
                current_value,
                source_value,
            )
            target[key] = source_value


def apply_training_experiment_context(
    *,
    training_config: DictConfig | None,
    inference_config: DictConfig | None,
    metrics_config: DictConfig | None,
    log: logging.Logger,
) -> None:
    """Apply runner context propagation across training, inference, and metrics.

    Runner entry points call this after loading configs and before resolving
    interpolations. When `training_config` is available, its `exp_tag` and
    `exp_dir` are treated as the source of truth for inference and metrics.
    When both `inference_config` and `metrics_config` are present,
    `metrics_config` also inherits `inference_dir` from `inference_config` so
    measurement follows the same output location as inference.

    Args:
        training_config (DictConfig | None): Training config selected for the
            current run. When `None`, this function is a no-op.
        inference_config (DictConfig | None): Inference config to patch in
            place when present.
        metrics_config (DictConfig | None): Metrics config to patch in place
            when present.
        log (logging.Logger): Logger used for insert/overwrite messages.

    Returns:
        None: Provided configs are mutated in place.

    Raises:
        This function does not raise exceptions.

    Notes:
        This helper is intentionally runner-oriented. It does not load configs;
        it only normalizes already-loaded configs before stage execution.

    Examples:
        Insert missing experiment identity from training:

        ```python
        training_config = OmegaConf.create(
            {"exp_tag": "train_asr_rnn", "exp_dir": "./exp/train_asr_rnn"}
        )
        inference_config = OmegaConf.create(
            {"inference_dir": "${exp_dir}/inference"}
        )
        apply_training_experiment_context(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=None,
            log=logging.getLogger("example"),
        )
        # Logs:
        #   INFO Inserted inference_config.exp_tag from training_config
        #   INFO Inserted inference_config.exp_dir from training_config
        # Result:
        #   inference_config.exp_tag == "train_asr_rnn"
        #   inference_config.exp_dir == "./exp/train_asr_rnn"
        ```

        Override conflicting experiment identity in inference:

        ```python
        training_config = OmegaConf.create(
            {"exp_tag": "train_asr_rnn", "exp_dir": "./exp/train_asr_rnn"}
        )
        inference_config = OmegaConf.create(
            {"exp_tag": "manual_tag", "exp_dir": "./exp/manual_tag"}
        )
        apply_training_experiment_context(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=None,
            log=logging.getLogger("example"),
        )
        # Logs:
        #   WARNING Overriding inference_config.exp_tag with
        #   training_config value: 'manual_tag' -> 'train_asr_rnn'
        #   WARNING Overriding inference_config.exp_dir with
        #   training_config value: './exp/manual_tag' -> './exp/train_asr_rnn'
        ```

        Align metrics with a custom inference output directory:

        ```python
        inference_config = OmegaConf.create(
            {
                "exp_tag": "eval_debug",
                "exp_dir": "./exp/eval_debug",
                "inference_dir": "./custom/eval_outputs",
            }
        )
        metrics_config = OmegaConf.create({"inference_dir": None})
        apply_training_experiment_context(
            training_config=None,
            inference_config=inference_config,
            metrics_config=metrics_config,
            log=logging.getLogger("example"),
        )
        # Logs:
        #   INFO Inserted metrics_config.exp_tag from inference_config
        #   INFO Inserted metrics_config.exp_dir from inference_config
        #   INFO Inserted metrics_config.inference_dir from inference_config
        # Result:
        #   metrics_config.inference_dir == "./custom/eval_outputs"
        #   metrics_config.exp_dir == "./exp/eval_debug"
        #   metrics_config.exp_tag == "eval_debug"
        ```

        Leave standalone inference unchanged when no training config is given:

        ```python
        inference_config = OmegaConf.create(
            {"exp_tag": "whisper_eval", "exp_dir": "./exp/whisper_eval"}
        )
        apply_training_experiment_context(
            training_config=None,
            inference_config=inference_config,
            metrics_config=None,
            log=logging.getLogger("example"),
        )
        # No logs are emitted and inference_config is unchanged.
        ```
    """
    if training_config is not None:
        if inference_config is not None:
            _copy_config_context(
                source=training_config,
                target=inference_config,
                keys=_TRAINING_CONTEXT_KEYS,
                source_name="training_config",
                target_name="inference_config",
                log=log,
            )
        if metrics_config is not None:
            _copy_config_context(
                source=training_config,
                target=metrics_config,
                keys=_TRAINING_CONTEXT_KEYS,
                source_name="training_config",
                target_name="metrics_config",
                log=log,
            )

    if inference_config is not None and metrics_config is not None:
        _copy_config_context(
            source=inference_config,
            target=metrics_config,
            keys=_INFERENCE_METRICS_CONTEXT_KEYS,
            source_name="inference_config",
            target_name="metrics_config",
            log=log,
        )


def validate_experiment_context(
    *,
    training_config: DictConfig | None,
    inference_config: DictConfig | None,
    metrics_config: DictConfig | None,
    stages_to_run: Sequence[str],
) -> None:
    """Validate that runtime configs have enough experiment identity.

    This helper enforces the two supported runner modes:

    1. Training-backed mode, where `training_config` is present and provides
       experiment identity for inference and metrics.
    2. Standalone inference/metrics mode, where the runtime config must define
       its own `exp_tag` or concrete `exp_dir`.

    Args:
        training_config (DictConfig | None): Training config selected for the
            current run.
        inference_config (DictConfig | None): Inference config for the `infer`
            stage.
        metrics_config (DictConfig | None): Metrics config for the `measure`
            stage.
        stages_to_run (Sequence[str]): Resolved stage names requested by the
            runner.

    Returns:
        None: Validation succeeds silently.

    Raises:
        ValueError: If `infer` or `measure` is requested without
            `training_config` and the corresponding runtime config does not
            define `exp_tag` or a concrete `exp_dir`.

    Notes:
        Validation is stage-aware. For example, a missing metrics identity is
        ignored when `measure` is not in `stages_to_run`.

    Examples:
        Allow standalone inference when the config already defines its
        experiment identity:

        ```python
        inference_config = OmegaConf.create(
            {"exp_tag": "whisper_eval", "exp_dir": "./exp/whisper_eval"}
        )
        validate_experiment_context(
            training_config=None,
            inference_config=inference_config,
            metrics_config=None,
            stages_to_run=["infer"],
        )
        # Succeeds because standalone inference can derive exp_dir.
        ```

        Reject standalone inference when experiment identity is missing:

        ```python
        inference_config = OmegaConf.create(
            {"inference_dir": "${exp_dir}/inference"}
        )
        validate_experiment_context(
            training_config=None,
            inference_config=inference_config,
            metrics_config=None,
            stages_to_run=["infer"],
        )
        # Raises:
        #   ValueError: infer stage requires --training_config or a
        #   standalone inference_config with exp_tag/exp_dir.
        ```

        Accept training-backed inference even when inference_config does not
        define `exp_tag`:

        ```python
        training_config = OmegaConf.create(
            {"exp_tag": "train_asr_rnn", "exp_dir": "./exp/train_asr_rnn"}
        )
        inference_config = OmegaConf.create(
            {"inference_dir": "${exp_dir}/inference"}
        )
        validate_experiment_context(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=None,
            stages_to_run=["infer"],
        )
        # Succeeds because training-backed mode is active.
        ```
    """
    if training_config is not None:
        return

    if "infer" in stages_to_run and inference_config is not None:
        if not _has_exp_identity(inference_config):
            raise ValueError(
                "infer stage requires --training_config or a standalone "
                "inference_config with exp_tag/exp_dir."
            )

    if "measure" in stages_to_run and metrics_config is not None:
        if not _has_exp_identity(metrics_config):
            raise ValueError(
                "measure stage requires --training_config or a standalone "
                "metrics_config with exp_tag/exp_dir."
            )


def resolve_loaded_configs(*configs: DictConfig | None) -> None:
    """Resolve a set of already-loaded configs in place.

    Runner entry points use `load_and_merge_config(..., resolve=False)` so they
    can patch experiment identity before OmegaConf interpolations are
    evaluated. This helper performs the final resolution step once all runtime
    adjustments are complete.

    Args:
        *configs (DictConfig | None): Configs to resolve. `None` entries are
            ignored.

    Returns:
        None: Provided configs are resolved in place.

    Raises:
        omegaconf.errors.OmegaConfBaseException: Propagated if interpolation
            resolution fails for any config.

    Notes:
        Resolution happens independently for each config in the order passed by
        the caller.

    Examples:
        `resolve_loaded_configs(training, inference, metrics)` resolves all
        three configs in place.
        `resolve_loaded_configs(None, inference)` resolves only inference.
    """
    for config in configs:
        if config is not None:
            OmegaConf.resolve(config)
