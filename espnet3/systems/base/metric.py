"""Calculate metrics entrypoint for hypothesis/reference outputs."""

import json
import logging
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.metrics.abs_metric import AbsMetric
from espnet3.utils.logging_utils import log_component
from espnet3.utils.scp_utils import get_class_path, load_scp_fields

logger = logging.getLogger(__name__)


def measure(measure_config: DictConfig):
    """Compute metrics for each test set and write a metrics JSON file.

    Args:
        measure_config: Omegaconf configuration with inference and metric settings.

    Returns:
        Nested dict keyed by metric class path and test set name.
    """
    test_sets = [t.name for t in measure_config.dataset.test]
    results = {}
    assert hasattr(measure_config, "metrics"), "Please specify `metrics`!"

    for idx, metric_config in enumerate(measure_config.metrics):
        metric = instantiate(metric_config.metric)
        if not isinstance(metric, AbsMetric):
            raise TypeError(f"{type(metric)} is not a valid AbsMetric instance")

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
            data = load_scp_fields(
                inference_dir=Path(measure_config.inference_dir),
                test_name=test_name,
                inputs=inputs,
                file_suffix=".scp",
            )
            metric_result = metric(data, test_name, measure_config.inference_dir)
            results[get_class_path(metric)].update({test_name: metric_result})

    out_path = Path(measure_config.inference_dir) / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
