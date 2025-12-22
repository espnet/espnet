import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.metrics.abs_metric import AbsMetrics
from espnet3.utils.scp_utils import get_class_path, load_scp_fields


def score(config: DictConfig):
    test_sets = [t.name for t in config.dataset.test]
    results = {}
    assert hasattr(config, "metrics"), "Please specify metrics!"

    for metric_cfg in config.metrics:
        apply_to = metric_cfg.get("apply_to", None)
        assert apply_to is not None, "Please set 'apply_to' to specify test set"

        metric = instantiate(metric_cfg.metric)
        if not isinstance(metric, AbsMetrics):
            raise TypeError(f"{type(metric)} is not a valid AbsMetrics instance")

        results[get_class_path(metric)] = {}
        for test_name in test_sets:
            if test_name not in apply_to:
                continue

            inputs = OmegaConf.to_container(metric_cfg.inputs, resolve=True)
            data = load_scp_fields(
                decode_dir=Path(config.decode_dir),
                test_name=test_name,
                inputs=inputs,
                file_suffix=".scp",
            )
            metric_result = metric(data, test_name, config.decode_dir)
            results[get_class_path(metric)].update({test_name: metric_result})

    out_path = Path(config.decode_dir) / "scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
