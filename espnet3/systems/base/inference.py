"""Inference entrypoint for ESPnet3 systems."""

import logging
import time
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.inference_runner import _load_output_fn

logger = logging.getLogger(__name__)


def _flatten_results(results):
    flat = []
    for item in results:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _collect_scp_lines(results, idx_key: str, hyp_keys, ref_keys):
    scp_lines = {}
    hyp_keys = list(hyp_keys) if isinstance(hyp_keys, (list, tuple)) else [hyp_keys]
    ref_keys = list(ref_keys) if isinstance(ref_keys, (list, tuple)) else [ref_keys]
    list_sizes = {key: None for key in (*hyp_keys, *ref_keys)}

    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                f"Expected dict output, got {type(result).__name__}: {result}"
            )

        idx_value = result[idx_key]
        if isinstance(idx_value, (list, tuple)):
            raise TypeError(
                f"'{idx_key}' must be a scalar, got {type(idx_value).__name__}"
            )

        for field_key in (*ref_keys, *hyp_keys):
            value = result[field_key]
            if isinstance(value, (list, tuple)):
                if list_sizes[field_key] is None:
                    list_sizes[field_key] = len(value)
                elif list_sizes[field_key] != len(value):
                    raise ValueError(
                        f"List length mismatch for '{field_key}': "
                        f"expected {list_sizes[field_key]}, got {len(value)}"
                    )
                for i, entry in enumerate(value):
                    if isinstance(entry, (list, tuple)):
                        raise TypeError(f"Nested list is not allowed for '{field_key}'")
                    scp_key = f"{field_key}{i}"
                    scp_lines.setdefault(scp_key, []).append(f"{idx_value} {entry}")
            else:
                if list_sizes[field_key] is not None:
                    raise TypeError(
                        f"'{field_key}' must be a list when list outputs are used"
                    )
                scp_lines.setdefault(field_key, []).append(f"{idx_value} {value}")

    return scp_lines


def infer(config: DictConfig):
    """Run inference over all configured test sets and write SCP files.

    Args:
        config: Hydra/omegaconf configuration with dataset and inference settings.
    """
    start = time.perf_counter()
    set_parallel(config.parallel)

    test_sets = [test_set.name for test_set in config.dataset.test]
    assert len(test_sets) > 0, "No test set found in dataset"
    assert len(test_sets) == len(set(test_sets)), "Duplicate test key found."

    logger.info(
        "Starting inference | inference_dir=%s test_sets=%s",
        getattr(config, "inference_dir", None),
        test_sets,
    )

    for test_name in test_sets:
        logger.info("===> Processing test set: %s", test_name)
        config.test_set = test_name

        output_fn_path = getattr(config, "output_fn", None)
        if not output_fn_path:
            raise RuntimeError("infer_config.output_fn must be set.")

        _load_output_fn(output_fn_path)

        input_key = getattr(config, "input_key", None)
        if input_key is None:
            raise RuntimeError("infer_config.input_key must be set.")

        if isinstance(input_key, (list, tuple)) and not input_key:
            raise RuntimeError("infer_config.input_key must not be empty.")

        output_keys = getattr(config, "output_keys", None)
        if output_keys is not None:
            if isinstance(output_keys, str):
                output_keys = [output_keys]
            elif not isinstance(output_keys, (list, tuple)):
                output_keys = list(output_keys)
            if not output_keys:
                raise RuntimeError("infer_config.output_keys must not be empty.")

        idx_key = getattr(config, "idx_key", "uttid")

        batch_size = getattr(config, "batch_size", None)
        provider_config = getattr(config, "provider", None)
        if provider_config is None:
            raise RuntimeError("infer_config.provider must be set.")
        raw_params = getattr(provider_config, "params", {}) or {}
        if OmegaConf.is_config(raw_params):
            provider_params = OmegaConf.to_container(raw_params, resolve=True)
        else:
            provider_params = dict(raw_params)

        provider_params["input_key"] = input_key
        provider_params["output_fn_path"] = output_fn_path

        provider = instantiate(
            provider_config,
            infer_config=config,
            params=provider_params,
            _recursive_=False,
        )

        hyp_keys = output_keys if output_keys is not None else []
        runner_config = getattr(config, "runner", None)
        if runner_config is None:
            raise RuntimeError("infer_config.runner must be set.")

        runner_kwargs = {
            "provider": provider,
            "async_mode": False,
            "idx_key": idx_key,
            "hyp_key": hyp_keys,
            "batch_size": batch_size,
        }
        runner = instantiate(runner_config, **runner_kwargs)
        if not hasattr(runner, "idx_key"):
            raise TypeError(
                f"{type(runner).__name__} must provide inference runner attributes"
            )
        dataset_length = len(provider.build_dataset(config))
        logger.info("===> Processing %d samples..", dataset_length)
        out = runner(list(range(dataset_length)))
        if out is None:
            raise RuntimeError("Async inference is not supported in this entrypoint.")
        # Runner can return nested lists. normalize to flat list.
        results = _flatten_results(out)
        if output_keys is None:
            if not results:
                raise RuntimeError("No inference results available.")
            first = results[0]
            output_keys = [key for key in first.keys() if key != runner.idx_key]
            if not output_keys:
                raise RuntimeError("No output keys found in inference results.")

        # Convert output dicts into per-key SCP lines (uttid + value).
        scp_lines = _collect_scp_lines(
            results,
            idx_key=runner.idx_key,
            hyp_keys=output_keys,
            ref_keys=[],
        )

        # create scp files
        output_dir = Path(config.inference_dir) / test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, lines in scp_lines.items():
            with open(output_dir / f"{key}.scp", "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        logger.info(
            "Finished test set %s | outputs=%s",
            test_name,
            output_dir,
        )

    logger.info("Inference finished in %.2fs", time.perf_counter() - start)
