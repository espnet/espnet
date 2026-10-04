"""Calculate metrics entrypoint for hypothesis/reference outputs."""

import json
import logging
from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.utils.logging_utils import log_component
from espnet3.utils.scp_utils import get_class_path, load_scp_paths

logger = logging.getLogger(__name__)


def _resolve_test_sets(metrics_config: DictConfig) -> list[str]:
    """Return the test-set names to score for the measurement stage."""
    dataset = getattr(metrics_config, "dataset", None)
    test_config = getattr(dataset, "test", None) if dataset is not None else None
    if test_config:
        return [t.name for t in test_config]

    inference_dir = Path(metrics_config.inference_dir)
    test_sets = sorted(
        entry.name
        for entry in inference_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    if not test_sets:
        raise ValueError(
            "No test sets found. Specify `metrics_config.dataset.test` or place "
            f"test-set subdirectories under inference_dir: {inference_dir}"
        )
    logger.info(
        "Resolved test sets from inference_dir: %s",
        test_sets,
    )
    return test_sets


DATASET_PREFIX = "dataset:"


def _dataset_column_scp(
    inference_config: DictConfig | None,
    inference_dir: Path,
    test_name: str,
    column: str,
    idx_key: str,
) -> Path:
    """Write a test set's column as ``<inference_dir>/<test>/dataset/<column>.scp``.

    The reference a metric compares against is the dataset's own column, so
    it is read from the test set here, at scoring time, rather than copied
    by the ``infer`` stage: the inference directory holds what the model
    produced, and ``dataset/`` beside it what the data said. The file is
    kept, so a second ``measure`` run does not read the set again; delete
    it after changing the dataset.

    Args:
        inference_config: The inference config, for the test set definition
            (``dataset``, ``provider``) - the same the ``infer`` stage used.
        inference_dir: The inference directory.
        test_name: The test set.
        column: The dataset field to write, such as ``text``.
        idx_key: The item field holding the utterance id; the item's index
            when absent, as the ``infer`` stage does.

    Returns:
        The written ``.scp``, one ``<id> <value>`` line per item in dataset
        order.

    Raises:
        ValueError: If no inference config was given, or an item's column is
            not a scalar.
    """
    path = inference_dir / test_name / "dataset" / f"{column}.scp"
    if path.exists():
        logger.info("Reusing %s", path)
        return path
    if inference_config is None:
        raise ValueError(
            f"reading `{DATASET_PREFIX}{column}` needs the inference config for "
            "the test set definition; run measure through the system, or pass "
            "inference_config"
        )
    config = OmegaConf.create(OmegaConf.to_container(inference_config, resolve=True))
    with open_dict(config):
        config.test_set = test_name
    provider_target = getattr(getattr(config, "provider", None), "_target_", None)
    provider_cls = get_class(provider_target) if provider_target else InferenceProvider
    dataset = provider_cls.build_dataset(config)
    logger.info(
        "Writing %s from the %s test set (%d items)", path, test_name, len(dataset)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(len(dataset)):
            item = dataset[idx]
            if column not in item:
                raise KeyError(
                    f"test set {test_name!r} item {idx} has no {column!r}; "
                    f"it has {sorted(item)}"
                )
            value = item[column]
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"`{DATASET_PREFIX}{column}` must be a scalar column, "
                    f"got {type(value).__name__} at item {idx}"
                )
            handle.write(f"{item.get(idx_key, str(idx))} {value}\n")
    return path


def _resolve_inputs(
    inputs,
    metrics_config: DictConfig,
    test_name: str,
    inference_config: DictConfig | None,
) -> dict[str, Path]:
    """Map metric input aliases to files.

    An input is an ``.scp`` file the ``infer`` stage wrote, or a
    ``dataset:<column>`` column of the test set, written on demand.
    """
    input_map = {k: k for k in inputs} if isinstance(inputs, list) else dict(inputs)
    inference_dir = Path(metrics_config.inference_dir)
    from_scp = {
        a: f for a, f in input_map.items() if not str(f).startswith(DATASET_PREFIX)
    }
    data = (
        load_scp_paths(inference_dir, test_name, inputs=from_scp, file_suffix=".scp")
        if from_scp
        else {}
    )
    idx_key = (
        inference_config.get("idx_key", "utt_id")
        if inference_config is not None
        else "utt_id"
    )
    for alias, source in input_map.items():
        if str(source).startswith(DATASET_PREFIX):
            data[alias] = _dataset_column_scp(
                inference_config,
                inference_dir,
                test_name,
                str(source)[len(DATASET_PREFIX) :],
                idx_key,
            )
    return data


def measure(metrics_config: DictConfig, inference_config: DictConfig | None = None):
    """Compute metrics for each test set and write a metrics JSON file.

    Test sets are resolved in the following order:

        1. If ``metrics_config.dataset.test`` is defined, use the configured
           ``name`` fields as-is.
        2. Otherwise, scan ``metrics_config.inference_dir`` and treat each
           non-hidden subdirectory as a test set.

    Example:
        If ``inference_dir`` contains:

        .. code-block:: text

            exp/my_run/inference/
              test-clean/
              test-other/

        then ``measure()`` scores both ``test-clean`` and ``test-other``
        when ``metrics_config.dataset.test`` is omitted.

    A metric's inputs are ``.scp`` files the ``infer`` stage wrote, named by
    alias (``hyp_key: text`` reads ``<test_name>/text.scp``), or a column of
    the test set itself, named ``dataset:<column>`` (``ref_key: dataset:text``
    reads the transcript from the data and writes it to
    ``<test_name>/dataset/text.scp`` on first use). The reference is the
    data's, so it comes from the data, not from what inference wrote.

    Args:
        metrics_config: Omegaconf configuration with inference and metric settings.
        inference_config: The inference config, needed for ``dataset:<column>``
            inputs; the system passes its own.

    Returns:
        Nested dict keyed by metric class path and test set name.

    Raises:
        ValueError: If no test sets can be resolved from either
            ``metrics_config.dataset.test`` or ``metrics_config.inference_dir``.
    """
    test_sets = _resolve_test_sets(metrics_config)
    results = {}
    assert hasattr(metrics_config, "metrics"), "Please specify `metrics`!"

    for idx, metric_config in enumerate(metrics_config.metrics):
        metric = instantiate(metric_config.metric)
        if not isinstance(metric, BaseMetric):
            raise TypeError(f"{type(metric)} is not a valid BaseMetric instance")

        log_component(
            logger,
            kind="Metric",
            label=str(idx),
            obj=metric,
            max_depth=2,
        )
        results[get_class_path(metric)] = {}
        for test_name in test_sets:
            if hasattr(metric_config, "inputs"):
                inputs = OmegaConf.to_container(metric_config.inputs, resolve=True)
            else:
                ref_key = getattr(metric, "ref_key", None)
                hyp_key = getattr(metric, "hyp_key", None)
                if ref_key is None or hyp_key is None:
                    raise ValueError(
                        f"Metric {get_class_path(metric)} requires inputs in config"
                    )
                inputs = [ref_key, hyp_key]
            data = _resolve_inputs(inputs, metrics_config, test_name, inference_config)
            metric_result = metric(data, test_name, metrics_config.inference_dir)
            results[get_class_path(metric)].update({test_name: metric_result})

    out_path = Path(metrics_config.inference_dir) / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
