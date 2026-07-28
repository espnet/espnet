"""Calculate metrics entrypoint for hypothesis/reference outputs."""

import json
import logging
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.metrics.base_metric import BaseMetric
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


def measure(metrics_config: DictConfig):
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

    Args:
        metrics_config: Omegaconf configuration with inference and metric settings.

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
            data = load_scp_paths(
                inference_dir=Path(metrics_config.inference_dir),
                test_name=test_name,
                inputs=inputs,
                file_suffix=".scp",
            )
            metric_result = metric(data, test_name, metrics_config.inference_dir)
            results[get_class_path(metric)].update({test_name: metric_result})

    out_path = Path(metrics_config.inference_dir) / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
