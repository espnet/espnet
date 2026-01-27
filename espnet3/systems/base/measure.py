"""Metric measurement entrypoint for hypothesis/reference outputs."""

import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.metrics.abs_metric import AbsMetrics
from espnet3.utils.scp_utils import get_class_path, load_scp_fields


def measure(config: DictConfig):
    """Compute metrics for each test set and write a measures JSON file.

    Args:
        config: Hydra/omegaconf configuration with inference and metric settings.

    Returns:
        Nested dict keyed by metric class path and test set name.
    """
    test_sets = [t.name for t in config.dataset.test]
    results = {}
    assert hasattr(config, "metrics"), "Please specify metrics!"

    for metric_cfg in config.metrics:
        metric = instantiate(metric_cfg.metric)
        if not isinstance(metric, AbsMetrics):
            raise TypeError(f"{type(metric)} is not a valid AbsMetrics instance")

        results[get_class_path(metric)] = {}
        for test_name in test_sets:
            if hasattr(metric_cfg, "inputs"):
                inputs = OmegaConf.to_container(metric_cfg.inputs, resolve=True)
            else:
                ref_key = getattr(metric, "ref_key", None)
                hyp_key = getattr(metric, "hyp_key", None)
                if ref_key is None or hyp_key is None:
                    raise ValueError(
                        f"Metric {get_class_path(metric)} requires inputs in config"
                    )
                inputs = [ref_key, hyp_key]
            data = load_scp_fields(
                infer_dir=Path(config.infer_dir),
                test_name=test_name,
                inputs=inputs,
                file_suffix=".scp",
            )
            metric_result = metric(data, test_name, config.infer_dir)
            results[get_class_path(metric)].update({test_name: metric_result})

    out_path = Path(config.infer_dir) / "measures.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
